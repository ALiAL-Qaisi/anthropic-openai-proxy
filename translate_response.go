package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

// mapStopReason converts an OpenAI finish_reason to an Anthropic stop_reason.
// Unknown values fall back to "end_turn" with a warning.
func mapStopReason(lg *reqLogger, finish string) string {
	switch finish {
	case "stop", "stop_sequence":
		return "end_turn"
	case "length", "max_tokens":
		return "max_tokens"
	case "tool_calls", "function_call":
		return "tool_use"
	case "":
		return "end_turn"
	default:
		lg.warnf("unmapped upstream finish_reason %q; falling back to stop_reason \"end_turn\"", finish)
		return "end_turn"
	}
}

// nonStreamResponse reads a complete upstream JSON body and replies with an
// Anthropic-format message.
func (p *proxy) nonStreamResponse(w http.ResponseWriter, lg *reqLogger, resp *http.Response, areq *anthropicRequest) {
	raw, err := io.ReadAll(io.LimitReader(resp.Body, maxRequestBody))
	if err != nil {
		lg.errorf("reading upstream body: %v", err)
		p.writeAnthropicError(w, lg, http.StatusBadGateway, "api_error", "Could not read upstream response: "+err.Error())
		return
	}
	lg.debugBlob(dbgUpIn, "upstream response body", raw)

	var ores oaiResponse
	if err := json.Unmarshal(raw, &ores); err != nil {
		lg.errorf("parsing upstream body: %v", err)
		p.writeAnthropicError(w, lg, http.StatusBadGateway, "api_error",
			"Upstream returned a body that is not valid chat-completions JSON: "+err.Error())
		return
	}

	// Some providers return HTTP 200 with an error object in the body.
	if ores.Error != nil && ores.Error.Message != "" {
		lg.errorf("upstream returned an error object in a 200 body: %s", ores.Error.Message)
		p.writeAnthropicError(w, lg, http.StatusBadGateway, "api_error",
			"Upstream provider error: "+extractUpstreamError(raw))
		return
	}

	if len(ores.Choices) == 0 {
		lg.errorf("upstream response had no choices")
		p.writeAnthropicError(w, lg, http.StatusBadGateway, "api_error", "Upstream response contained no choices.")
		return
	}

	choice := ores.Choices[0]
	msg := choice.Message
	if msg == nil {
		msg = &oaiRespMessage{}
	}

	ares := anthropicResponse{
		ID:           newMessageID(),
		Type:         "message",
		Role:         "assistant",
		Model:        firstNonEmpty(ores.Model, p.cfg.upstreamModel),
		Content:      []anthropicRespBlock{},
		StopSequence: nil,
	}

	// GLM returns its chain-of-thought separately; Anthropic has no equivalent
	// block for a non-thinking request, so it is dropped.
	if !isJSONNull(msg.ReasoningContent) {
		if rc := rawToText(msg.ReasoningContent); rc != "" {
			lg.debugf(dbgXlate, "dropping upstream `reasoning_content` (%d chars) — no Anthropic equivalent in a non-thinking response", len(rc))
		}
	}

	// Text first, then any tool_use blocks — the order the model produced them.
	if text := rawToText(msg.Content); text != "" {
		ares.Content = append(ares.Content, anthropicRespBlock{Type: "text", Text: &text})
	}

	for i, tc := range msg.ToolCalls {
		input := json.RawMessage(`{}`)
		if tc.Function.Arguments != "" {
			// `arguments` is a JSON-encoded *string* upstream; Anthropic wants a
			// real object in `input`.
			if json.Valid([]byte(tc.Function.Arguments)) {
				input = json.RawMessage(tc.Function.Arguments)
			} else {
				lg.warnf("tool_calls[%d] (%s) arguments are not valid JSON; substituting {}. raw=%q",
					i, tc.Function.Name, tc.Function.Arguments)
			}
		}
		id := tc.ID
		if id == "" {
			id = newToolUseID()
			lg.debugf(dbgXlate, "tool_calls[%d] had no id; generated %s", i, id)
		}
		ares.Content = append(ares.Content, anthropicRespBlock{
			Type:  "tool_use",
			ID:    id,
			Name:  tc.Function.Name,
			Input: input,
		})
	}

	// Anthropic requires at least one content block.
	if len(ares.Content) == 0 {
		lg.debugf(dbgXlate, "upstream produced neither content nor tool calls; emitting an empty text block")
		empty := ""
		ares.Content = append(ares.Content, anthropicRespBlock{Type: "text", Text: &empty})
	}

	stop := mapStopReason(lg, choice.FinishReason)
	// Claude Code decides whether to execute a tool from stop_reason alone, so
	// tool calls must always report "tool_use" even if the provider said "stop".
	if len(msg.ToolCalls) > 0 && stop != "tool_use" {
		lg.debugf(dbgXlate, "upstream finish_reason %q accompanied %d tool call(s); forcing stop_reason \"tool_use\"",
			choice.FinishReason, len(msg.ToolCalls))
		stop = "tool_use"
	}
	ares.StopReason = strPtr(stop)

	if ores.Usage != nil {
		ares.Usage = anthropicUsage{
			InputTokens:  ores.Usage.PromptTokens,
			OutputTokens: ores.Usage.CompletionTokens,
		}
	} else {
		in := estimateTokens(approxRequestChars(areq))
		outChars := 0
		for _, b := range ares.Content {
			if b.Text != nil {
				outChars += len(*b.Text)
			}
			outChars += len(b.Input)
		}
		ares.Usage = anthropicUsage{InputTokens: in, OutputTokens: estimateTokens(outChars)}
		lg.debugf(dbgXlate, "upstream reported no usage; estimated input=%d output=%d (chars/4)", ares.Usage.InputTokens, ares.Usage.OutputTokens)
	}

	body, err := json.Marshal(ares)
	if err != nil {
		lg.errorf("marshalling Anthropic response: %v", err)
		p.writeAnthropicError(w, lg, http.StatusInternalServerError, "api_error", "Could not encode response: "+err.Error())
		return
	}

	lg.debugBlob(dbgOut, "Anthropic response body", body)
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Content-Length", fmt.Sprintf("%d", len(body)))
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write(body)
}

func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if v != "" {
			return v
		}
	}
	return ""
}
