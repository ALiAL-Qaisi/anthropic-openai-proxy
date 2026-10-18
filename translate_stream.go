package main

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// ---------------------------------------------------------------------------
// Streaming translation: a flat OpenAI delta stream -> Anthropic's
// block-structured event stream.
//
// The two formats disagree on three things that drive this state machine:
//
//  1. Anthropic content blocks are opened/closed explicitly and indexed
//     sequentially across ALL blocks, while OpenAI's `tool_calls[].index`
//     restarts at 0 independently of any text. With GLM emitting text *and* a
//     tool call, OpenAI tool index 0 must become Anthropic block index 1, so
//     indices are allocated here (nextIndex) and mapped, never passed through.
//
//  2. Anthropic keeps exactly one block open at a time, so any open block must
//     be closed before the next one starts.
//
//  3. Tool arguments stream as fragments of a JSON *string* that the client
//     accumulates; `id`/`name` normally arrive in the first fragment for an
//     index. GLM sends name+arguments in a single delta, but the generic
//     contract allows them to be split, so tool blocks are opened lazily: args
//     are buffered until a name is known, then flushed.
//
//  4. Because OpenAI may interleave fragments for several tool indices while
//     Anthropic permits only one open block, exactly one tool "owns" the open
//     block and streams live; fragments for any other index are buffered and
//     emitted in full once that block closes. Closing a tool block early just
//     because another index appeared would strand its remaining fragments and
//     yield unparseable input JSON.
// ---------------------------------------------------------------------------

type sseWriter struct {
	w   http.ResponseWriter
	f   http.Flusher
	lg  *reqLogger
	err error
}

func (s *sseWriter) send(event string, payload any) {
	if s.err != nil {
		return // client already gone; stop trying to write
	}
	b, err := json.Marshal(payload)
	if err != nil {
		s.lg.errorf("marshalling SSE event %q: %v", event, err)
		return
	}
	if _, werr := fmt.Fprintf(s.w, "event: %s\ndata: %s\n\n", event, b); werr != nil {
		s.err = werr
		s.lg.errorf("writing SSE event %q: %v", event, werr)
		return
	}
	s.f.Flush()
	s.lg.debugf(dbgOut, "event: %s | data: %s", event, b)
}

type toolStream struct {
	anthIndex int
	id        string
	name      string
	pending   string          // fragments received before the block could open
	accum     strings.Builder // everything emitted, validated as JSON at close
	open      bool
	closed    bool
}

type streamState struct {
	sse   *sseWriter
	lg    *reqLogger
	cfg   *config
	model string
	msgID string

	started  bool // message_start emitted
	finished bool // message_delta emitted

	nextIndex int // next Anthropic block index to allocate

	textOpen  bool
	textIndex int

	tools       map[int]*toolStream // keyed by upstream tool_calls[].index
	toolOrder   []int
	streamTool  int // upstream index owning the open tool block, or -1
	toolsOpened int // tool blocks actually emitted (excludes dropped ones)

	stopReason string
	usage      *oaiUsage
	inputEst   int
	outChars   int
}

func newStreamState(sse *sseWriter, lg *reqLogger, cfg *config, inputEst int) *streamState {
	return &streamState{
		sse:        sse,
		lg:         lg,
		cfg:        cfg,
		model:      cfg.upstreamModel,
		msgID:      newMessageID(),
		tools:      map[int]*toolStream{},
		streamTool: -1,
		inputEst:   inputEst,
	}
}

// streamResponse drives the whole upstream SSE read loop.
func (p *proxy) streamResponse(w http.ResponseWriter, r *http.Request, lg *reqLogger, resp *http.Response, areq *anthropicRequest) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		lg.errorf("ResponseWriter does not implement http.Flusher; cannot stream")
		p.writeAnthropicError(w, lg, http.StatusInternalServerError, "api_error", "Server cannot stream responses.")
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no") // defeat proxy buffering if any
	w.WriteHeader(http.StatusOK)
	flusher.Flush()

	sse := &sseWriter{w: w, f: flusher, lg: lg}
	st := newStreamState(sse, lg, p.cfg, estimateTokens(approxRequestChars(areq)))

	reader := bufio.NewReaderSize(resp.Body, 64*1024)

	for {
		// Stop promptly if Claude Code hung up.
		if err := r.Context().Err(); err != nil {
			lg.debugf(dbgUpIn, "client disconnected (%v); abandoning stream", err)
			return
		}

		line, err := reader.ReadString('\n')
		if len(line) > 0 {
			if stop := st.handleRawLine(strings.TrimRight(line, "\r\n")); stop {
				return
			}
		}
		if err != nil {
			if !errors.Is(err, io.EOF) {
				lg.errorf("reading upstream stream: %v", err)
				// Headers are already sent, so the only way to report this is an
				// in-band error event.
				if !st.finished {
					st.ensureStarted()
					st.closeOpenBlock()
					sse.send("error", anthropicErrorEnvelope{
						Type:  "error",
						Error: anthropicErrorBody{Type: "api_error", Message: "Upstream stream failed: " + err.Error()},
					})
				}
			} else {
				lg.debugf(dbgUpIn, "upstream stream ended (EOF)")
			}
			// Always leave the client with a well-formed, terminated stream.
			st.finalize()
			return
		}
	}
}

// handleRawLine processes one raw upstream SSE line. It returns true when the
// stream is complete and the caller should stop reading.
func (s *streamState) handleRawLine(line string) bool {
	if line == "" {
		return false // SSE event separator
	}
	s.lg.debugf(dbgUpIn, "%s", line)

	if strings.HasPrefix(line, ":") {
		return false // comment / keep-alive
	}
	if strings.HasPrefix(line, "event:") || strings.HasPrefix(line, "id:") || strings.HasPrefix(line, "retry:") {
		return false // upstream field we don't need
	}
	if !strings.HasPrefix(line, "data:") {
		s.lg.debugf(dbgXlate, "ignoring unrecognized upstream SSE line: %q", line)
		return false
	}

	payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
	if payload == "" {
		return false
	}
	if payload == "[DONE]" {
		s.finalize()
		return true
	}

	var chunk oaiResponse
	if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
		s.lg.warnf("could not parse upstream stream chunk as JSON; skipping it: %v", err)
		return false
	}

	// An error can arrive mid-stream after a 200.
	if chunk.Error != nil && chunk.Error.Message != "" {
		msg := chunk.Error.Message
		if len(chunk.Error.Code) > 0 {
			msg = fmt.Sprintf("%s (code %s)", msg, strings.Trim(string(chunk.Error.Code), `"`))
		}
		s.lg.errorf("upstream reported an error mid-stream: %s", msg)
		s.ensureStarted()
		s.closeOpenBlock()
		s.sse.send("error", anthropicErrorEnvelope{
			Type:  "error",
			Error: anthropicErrorBody{Type: "api_error", Message: "Upstream provider error: " + msg},
		})
		return true
	}

	s.handleChunk(&chunk)
	return false
}

func (s *streamState) handleChunk(chunk *oaiResponse) {
	if chunk.Model != "" {
		s.model = chunk.Model
	}
	if chunk.Usage != nil {
		s.usage = chunk.Usage // GLM sends real usage in the final chunk
	}

	// message_start goes out on the first chunk, before any block decisions.
	s.ensureStarted()

	if len(chunk.Choices) == 0 {
		return
	}
	choice := chunk.Choices[0]

	if delta := choice.Delta; delta != nil {
		// Chain-of-thought: dropped, and crucially must NOT open a text block —
		// GLM streams all of its reasoning_content before any real content.
		if !isJSONNull(delta.ReasoningContent) {
			if rc := rawToText(delta.ReasoningContent); rc != "" {
				s.lg.debugf(dbgXlate, "dropping `reasoning_content` fragment (%d chars) — no Anthropic equivalent", len(rc))
			}
		}

		// The final chunk carries content:"" alongside finish_reason, so an
		// empty string must never open or extend a text block.
		if text := rawToText(delta.Content); text != "" {
			s.onText(text)
		}

		for _, tc := range delta.ToolCalls {
			s.onToolCall(tc)
		}
	}

	if choice.FinishReason != "" {
		s.onFinish(choice.FinishReason)
	}
}

func (s *streamState) ensureStarted() {
	if s.started {
		return
	}
	s.started = true
	// input_tokens is an estimate here: GLM only reports real usage in its
	// final chunk, and message_start must go out first. The real numbers are
	// sent later in message_delta.
	s.sse.send("message_start", evtMessageStart{
		Type: "message_start",
		Message: anthropicResponse{
			ID:           s.msgID,
			Type:         "message",
			Role:         "assistant",
			Model:        s.model,
			Content:      []anthropicRespBlock{},
			StopReason:   nil,
			StopSequence: nil,
			Usage:        anthropicUsage{InputTokens: s.inputEst, OutputTokens: 0},
		},
	})
}

func (s *streamState) onText(text string) {
	if s.finished {
		s.lg.debugf(dbgXlate, "discarding text received after finish_reason")
		return
	}
	// Text after a tool call belongs in a fresh block.
	if s.streamTool >= 0 {
		s.closeOpenBlock()
	}
	if !s.textOpen {
		s.textIndex = s.nextIndex
		s.nextIndex++
		s.textOpen = true
		empty := ""
		s.sse.send("content_block_start", evtContentBlockStart{
			Type:         "content_block_start",
			Index:        s.textIndex,
			ContentBlock: startBlock{Type: "text", Text: &empty},
		})
	}
	s.outChars += len(text)
	s.sse.send("content_block_delta", evtContentBlockDelta{
		Type:  "content_block_delta",
		Index: s.textIndex,
		Delta: textDeltaBody{Type: "text_delta", Text: text},
	})
}

func (s *streamState) onToolCall(tc oaiToolCall) {
	if s.finished {
		s.lg.debugf(dbgXlate, "discarding tool_call delta received after finish_reason")
		return
	}
	idx := tc.Index // absent in JSON decodes to 0, which is the correct default
	ts, exists := s.tools[idx]
	if !exists {
		ts = &toolStream{}
		s.tools[idx] = ts
		s.toolOrder = append(s.toolOrder, idx)
	}

	if tc.ID != "" {
		ts.id = tc.ID
	}
	if tc.Function.Name != "" {
		ts.name = tc.Function.Name
	}
	if tc.Function.Arguments != "" {
		if ts.closed {
			// The block was already closed (interrupted by text), so these
			// bytes can no longer be appended to it.
			s.lg.warnf("tool %q (upstream index %d) received %d more argument byte(s) after its block closed; they cannot be appended and are dropped: %q",
				ts.name, idx, len(tc.Function.Arguments), tc.Function.Arguments)
			return
		}
		ts.pending += tc.Function.Arguments
	}
	if ts.closed {
		return
	}

	// Only the tool owning the open block streams live. Fragments for any other
	// index stay buffered until that block closes (see note 4 above) — closing
	// early would strand this index's remaining fragments.
	if s.streamTool == -1 || s.streamTool == idx {
		s.flushTool(idx, ts)
	}
}

// flushTool opens the tool block once a name is known, then emits any buffered
// argument fragments. Only ever called for the index that owns (or is about to
// own) the open block.
func (s *streamState) flushTool(idx int, ts *toolStream) {
	if ts.closed {
		return
	}
	if !ts.open {
		if ts.name == "" {
			// Can't emit content_block_start without a name; keep buffering.
			return
		}
		s.closeOpenBlock()
		ts.anthIndex = s.nextIndex
		s.nextIndex++
		if ts.id == "" {
			ts.id = newToolUseID()
			s.lg.debugf(dbgXlate, "upstream tool_call index %d had no id; generated %s", idx, ts.id)
		}
		ts.open = true
		s.streamTool = idx
		s.toolsOpened++
		s.sse.send("content_block_start", evtContentBlockStart{
			Type:  "content_block_start",
			Index: ts.anthIndex,
			ContentBlock: startBlock{
				Type:  "tool_use",
				ID:    ts.id,
				Name:  ts.name,
				Input: json.RawMessage(`{}`),
			},
		})
	}
	if ts.pending != "" {
		s.outChars += len(ts.pending)
		ts.accum.WriteString(ts.pending)
		s.sse.send("content_block_delta", evtContentBlockDelta{
			Type:  "content_block_delta",
			Index: ts.anthIndex,
			Delta: jsonDeltaBody{Type: "input_json_delta", PartialJSON: ts.pending},
		})
		ts.pending = ""
	}
}

// closeOpenBlock closes whichever block is currently open, if any.
func (s *streamState) closeOpenBlock() {
	if s.textOpen {
		s.sse.send("content_block_stop", evtContentBlockStop{Type: "content_block_stop", Index: s.textIndex})
		s.textOpen = false
		return
	}
	if s.streamTool >= 0 {
		ts := s.tools[s.streamTool]
		if ts != nil {
			// The accumulated fragments are what the client will parse, so an
			// invalid total is worth flagging loudly.
			if args := ts.accum.String(); args != "" && !json.Valid([]byte(args)) {
				s.lg.warnf("accumulated arguments for tool %q are not valid JSON (%d bytes); the client will fail to parse them: %q",
					ts.name, len(args), args)
			}
			s.sse.send("content_block_stop", evtContentBlockStop{Type: "content_block_stop", Index: ts.anthIndex})
			ts.open = false
			ts.closed = true
		}
		s.streamTool = -1
	}
}

func (s *streamState) onFinish(finish string) {
	if s.finished {
		return
	}
	s.stopReason = mapStopReason(s.lg, finish)

	// Close whatever is open, then drain the tools that were buffered while
	// another index owned the block. Each is opened, flushed and closed in turn
	// (Anthropic allows only one open block), in upstream arrival order.
	s.closeOpenBlock()
	for _, idx := range s.toolOrder {
		ts := s.tools[idx]
		if ts == nil || ts.closed || ts.open {
			continue
		}
		if ts.name == "" {
			// A tool whose name never arrived cannot be a valid tool_use block.
			s.lg.warnf("upstream tool_call index %d never provided a function name; dropping it (buffered args: %q)", idx, ts.pending)
			ts.closed = true
			continue
		}
		s.flushTool(idx, ts)
		s.closeOpenBlock()
	}

	// Anthropic requires at least one content block, and every block must be
	// emitted before message_delta — so this has to happen here, not later.
	if s.nextIndex == 0 {
		s.lg.debugf(dbgXlate, "upstream produced no content at all; emitting an empty text block")
		empty := ""
		s.sse.send("content_block_start", evtContentBlockStart{
			Type:         "content_block_start",
			Index:        0,
			ContentBlock: startBlock{Type: "text", Text: &empty},
		})
		s.sse.send("content_block_stop", evtContentBlockStop{Type: "content_block_stop", Index: 0})
		s.nextIndex = 1
	}

	// Claude Code runs a tool based on stop_reason alone.
	if s.toolsOpened > 0 && s.stopReason != "tool_use" {
		s.lg.debugf(dbgXlate, "upstream finish_reason %q arrived with tool call(s); forcing stop_reason \"tool_use\"", finish)
		s.stopReason = "tool_use"
	}

	usage := anthropicUsage{InputTokens: s.inputEst, OutputTokens: estimateTokens(s.outChars)}
	if s.usage != nil {
		usage = anthropicUsage{InputTokens: s.usage.PromptTokens, OutputTokens: s.usage.CompletionTokens}
	} else {
		s.lg.debugf(dbgXlate, "upstream reported no usage in the stream; estimated input=%d output=%d (chars/4)",
			usage.InputTokens, usage.OutputTokens)
	}

	s.finished = true
	s.sse.send("message_delta", evtMessageDelta{
		Type: "message_delta",
		Delta: messageDeltaBody{
			StopReason:   strPtr(s.stopReason),
			StopSequence: nil,
		},
		Usage: usage,
	})
}

// finalize guarantees a well-formed, terminated Anthropic stream regardless of
// how the upstream stream ended — [DONE], EOF, or a read error.
func (s *streamState) finalize() {
	s.ensureStarted()
	if !s.finished {
		if s.stopReason == "" {
			s.lg.debugf(dbgXlate, "upstream stream ended without a finish_reason; defaulting stop_reason to \"end_turn\"")
		}
		// onFinish emits the empty-content block (if needed) and message_delta.
		s.onFinish("")
	}
	s.sse.send("message_stop", evtMessageStop{Type: "message_stop"})
}
