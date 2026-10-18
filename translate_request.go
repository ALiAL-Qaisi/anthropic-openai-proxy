package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
)

// blockJoiner separates the text of adjacent Anthropic content blocks when they
// have to be collapsed into one upstream string. Anthropic treats each block as
// its own unit, so a bare concatenation would run the last sentence of one
// block into the first word of the next ("You are Claude Code.Be terse.").
const blockJoiner = "\n\n"

// translateRequest converts an Anthropic Messages API request into an
// OpenAI-compatible chat/completions request.
//
// Shape differences handled here:
//   - `system` is a top-level field on Anthropic but a leading message upstream.
//     It may be a bare string OR an array of text blocks (Claude Code sends the
//     latter, with cache_control attached).
//   - Anthropic nests tool calls and tool results as content *blocks* inside
//     user/assistant messages; OpenAI puts tool calls on the assistant message
//     and tool results in their own `role:"tool"` messages.
//   - `tools[].input_schema` becomes `tools[].function.parameters`.
//   - The client's model name is discarded in favour of UPSTREAM_MODEL.
func translateRequest(cfg *config, lg *reqLogger, in *anthropicRequest) (*oaiRequest, error) {
	out := &oaiRequest{
		Model:       cfg.upstreamModel,
		Stream:      in.Stream,
		MaxTokens:   in.MaxTokens,
		Temperature: in.Temperature,
		TopP:        in.TopP,
		Stop:        in.StopSequences,
		Messages:    make([]oaiMessage, 0, len(in.Messages)+1),
	}

	if in.Model != "" && in.Model != cfg.upstreamModel {
		lg.debugf(dbgXlate, "overriding requested model %q with UPSTREAM_MODEL %q", in.Model, cfg.upstreamModel)
	}

	// Parameters with no OpenAI-compatible equivalent: drop, never fail.
	if !isJSONNull(in.Thinking) {
		lg.debugf(dbgXlate, "dropping Anthropic `thinking` config (unsupported upstream): %s", compact(in.Thinking))
	}
	if in.TopK != nil {
		lg.debugf(dbgXlate, "dropping `top_k`=%d (no OpenAI-compatible equivalent)", *in.TopK)
	}
	if !isJSONNull(in.Metadata) {
		lg.debugf(dbgXlate, "dropping `metadata` (not forwarded upstream): %s", compact(in.Metadata))
	}

	// --- system prompt -> leading system message ---
	if !isJSONNull(in.System) {
		sys := rawToText(in.System)
		if sys != "" {
			if in.System[0] == '[' {
				lg.debugf(dbgXlate, "flattened array-form `system` (%d bytes) into a single system message", len(in.System))
			}
			out.Messages = append(out.Messages, oaiMessage{Role: "system", Content: sys})
		} else {
			lg.debugf(dbgXlate, "`system` present but yielded no text; skipping: %s", compact(in.System))
		}
	}

	// --- messages ---
	for mi, m := range in.Messages {
		if isJSONNull(m.Content) {
			lg.debugf(dbgXlate, "messages[%d] (role %s) has null/empty content; skipping", mi, m.Role)
			continue
		}

		// Fast path: content is a plain string.
		var s string
		if err := json.Unmarshal(m.Content, &s); err == nil {
			if strings.TrimSpace(s) == "" {
				lg.debugf(dbgXlate, "messages[%d] (role %s) is an empty string; skipping", mi, m.Role)
				continue
			}
			out.Messages = append(out.Messages, oaiMessage{Role: m.Role, Content: s})
			continue
		}

		var blocks []anthropicBlock
		if err := json.Unmarshal(m.Content, &blocks); err != nil {
			return nil, fmt.Errorf("messages[%d].content is neither a string nor an array of content blocks: %w", mi, err)
		}

		converted, err := convertBlockMessage(lg, mi, m.Role, blocks)
		if err != nil {
			return nil, err
		}
		out.Messages = append(out.Messages, converted...)
	}

	if len(out.Messages) == 0 {
		return nil, fmt.Errorf("no translatable messages in request")
	}

	// --- tools ---
	for ti, t := range in.Tools {
		if t.Name == "" {
			// Server-side tool definitions (e.g. {"type":"web_search_20250305"})
			// carry no name/schema we can express as a function.
			lg.debugf(dbgXlate, "dropping tools[%d] with no name (type=%q); not expressible as an OpenAI function", ti, t.Type)
			continue
		}
		params := t.InputSchema
		if isJSONNull(params) {
			params = json.RawMessage(`{"type":"object","properties":{}}`)
			lg.debugf(dbgXlate, "tools[%d] (%s) had no input_schema; substituting an empty object schema", ti, t.Name)
		}
		out.Tools = append(out.Tools, oaiTool{
			Type: "function",
			Function: oaiToolFunc{
				Name:        t.Name,
				Description: t.Description,
				Parameters:  params,
			},
		})
	}

	// --- tool_choice ---
	if !isJSONNull(in.ToolChoice) {
		if tc, ok := translateToolChoice(lg, in.ToolChoice); ok {
			out.ToolChoice = tc
		}
	}

	return out, nil
}

// convertBlockMessage turns one Anthropic block-array message into one or more
// OpenAI messages.
//
// For an assistant turn, text blocks are concatenated and tool_use blocks
// become tool_calls on that single message.
//
// For a user turn, each tool_result block must become its own `role:"tool"`
// message. Text and tool_results are emitted in their original relative order,
// as specified.
func convertBlockMessage(lg *reqLogger, mi int, role string, blocks []anthropicBlock) ([]oaiMessage, error) {
	var out []oaiMessage

	var textParts []string
	var images []oaiImageContent
	var toolCalls []oaiToolCall
	sawToolResult := false

	// flushPending emits the accumulated text/images as a single message of the
	// current role, preserving position relative to tool messages.
	flushPending := func() {
		if len(textParts) == 0 && len(images) == 0 {
			return
		}
		text := strings.Join(textParts, blockJoiner)
		msg := oaiMessage{Role: role}
		if len(images) > 0 {
			// Multimodal: content must be an array of parts.
			parts := make([]any, 0, len(images)+1)
			if strings.TrimSpace(text) != "" {
				parts = append(parts, oaiTextContent{Type: "text", Text: text})
			}
			for _, img := range images {
				parts = append(parts, img)
			}
			msg.Content = parts
		} else {
			if strings.TrimSpace(text) == "" {
				textParts = nil
				return
			}
			msg.Content = text
		}
		out = append(out, msg)
		textParts = nil
		images = nil
	}

	for bi, b := range blocks {
		switch b.Type {
		case "text":
			textParts = append(textParts, b.Text)

		case "tool_use":
			// A tool call the model previously made.
			if role != "assistant" {
				lg.debugf(dbgXlate, "messages[%d].content[%d]: tool_use in a %q message (expected assistant); forwarding as a tool call anyway", mi, bi, role)
			}
			args := "{}"
			if !isJSONNull(b.Input) {
				args = compact(b.Input)
			}
			id := b.ID
			if id == "" {
				id = newToolUseID()
				lg.debugf(dbgXlate, "messages[%d].content[%d]: tool_use had no id; generated %s", mi, bi, id)
			}
			toolCalls = append(toolCalls, oaiToolCall{
				ID:       id,
				Type:     "function",
				Function: oaiFunc{Name: b.Name, Arguments: args},
			})

		case "tool_result":
			// A result Claude Code already produced -> its own `tool` message.
			sawToolResult = true
			if len(textParts) > 0 || len(images) > 0 {
				// Spec says preserve original order. Worth flagging: a strict
				// OpenAI provider requires `tool` messages to directly follow
				// the assistant message that made the calls, so text placed
				// before a tool_result can be rejected upstream.
				lg.debugf(dbgXlate, "messages[%d].content[%d]: plain content precedes a tool_result; emitting it as a separate %q message first (original order preserved, but some providers reject this ordering)", mi, bi, role)
				flushPending()
			}
			content := flattenToolResult(lg, mi, bi, b.Content)
			if b.IsError && content != "" {
				lg.debugf(dbgXlate, "messages[%d].content[%d]: tool_result is_error=true; upstream has no error flag, prefixing the text", mi, bi)
				content = "Error: " + content
			}
			if content == "" {
				content = "(no output)"
			}
			if b.ToolUseID == "" {
				lg.debugf(dbgXlate, "messages[%d].content[%d]: tool_result has no tool_use_id; upstream may reject it", mi, bi)
			}
			out = append(out, oaiMessage{
				Role:       "tool",
				ToolCallID: b.ToolUseID,
				Content:    content,
			})

		case "image":
			if img, ok := imageToOpenAI(b.Source); ok {
				lg.debugf(dbgXlate, "messages[%d].content[%d]: mapped Anthropic image block to an OpenAI image_url part (best effort; model may not accept it)", mi, bi)
				images = append(images, img)
			} else {
				lg.warnf("messages[%d].content[%d]: unusable image block (source type %q); dropped", mi, bi, sourceType(b.Source))
			}

		case "thinking", "redacted_thinking":
			lg.debugf(dbgXlate, "messages[%d].content[%d]: dropping %q block (no upstream equivalent)", mi, bi, b.Type)

		default:
			lg.warnf("messages[%d].content[%d]: unknown block type %q; dropped", mi, bi, b.Type)
		}
	}

	// An assistant turn carries its text and tool calls on one message.
	if role == "assistant" && len(toolCalls) > 0 {
		msg := oaiMessage{Role: "assistant", ToolCalls: toolCalls}
		if text := strings.Join(textParts, blockJoiner); strings.TrimSpace(text) != "" {
			msg.Content = text
		}
		if len(images) > 0 {
			lg.warnf("messages[%d]: dropping %d image block(s) from an assistant tool-call message", mi, len(images))
		}
		out = append(out, msg)
		return out, nil
	}

	flushPending()

	if len(toolCalls) > 0 {
		// Non-assistant role that nonetheless had tool_use blocks.
		out = append(out, oaiMessage{Role: "assistant", ToolCalls: toolCalls})
	}

	if len(out) == 0 && !sawToolResult {
		lg.debugf(dbgXlate, "messages[%d] (role %s) produced no upstream message (all %d block(s) dropped)", mi, role, len(blocks))
	}
	return out, nil
}

// flattenToolResult reduces tool_result content (string, or array of blocks) to
// the single string upstream expects.
func flattenToolResult(lg *reqLogger, mi, bi int, raw json.RawMessage) string {
	if isJSONNull(raw) {
		return ""
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s
	}

	var blocks []anthropicBlock
	if err := json.Unmarshal(raw, &blocks); err == nil {
		parts := make([]string, 0, len(blocks))
		for _, b := range blocks {
			switch b.Type {
			case "text":
				parts = append(parts, b.Text)
			case "image":
				parts = append(parts, "[image omitted]")
			default:
				parts = append(parts, compact(mustMarshal(b)))
			}
		}
		lg.debugf(dbgXlate, "messages[%d].content[%d]: flattened non-string tool_result content (%d block(s)) into a string", mi, bi, len(blocks))
		return strings.Join(parts, blockJoiner)
	}

	lg.debugf(dbgXlate, "messages[%d].content[%d]: tool_result content was neither a string nor a block array; forwarding raw JSON", mi, bi)
	return compact(raw)
}

// imageToOpenAI best-effort maps an Anthropic image source to an OpenAI
// image_url part (base64 payloads become data: URLs).
func imageToOpenAI(src *anthropicImageSource) (oaiImageContent, bool) {
	if src == nil {
		return oaiImageContent{}, false
	}
	switch src.Type {
	case "base64":
		if src.Data == "" {
			return oaiImageContent{}, false
		}
		mt := src.MediaType
		if mt == "" {
			mt = "image/png"
		}
		return oaiImageContent{
			Type:     "image_url",
			ImageURL: oaiImageURL{URL: fmt.Sprintf("data:%s;base64,%s", mt, src.Data)},
		}, true
	case "url":
		if src.URL == "" {
			return oaiImageContent{}, false
		}
		return oaiImageContent{Type: "image_url", ImageURL: oaiImageURL{URL: src.URL}}, true
	default:
		return oaiImageContent{}, false
	}
}

func sourceType(src *anthropicImageSource) string {
	if src == nil {
		return "<nil>"
	}
	return src.Type
}

// translateToolChoice maps Anthropic's {"type":"auto"|"any"|"tool"|"none"} onto
// OpenAI's "auto"|"required"|{"type":"function",...}|"none".
func translateToolChoice(lg *reqLogger, raw json.RawMessage) (any, bool) {
	var tc struct {
		Type string `json:"type"`
		Name string `json:"name"`
	}
	if err := json.Unmarshal(raw, &tc); err != nil {
		lg.debugf(dbgXlate, "could not parse tool_choice %s; dropping", compact(raw))
		return nil, false
	}
	switch tc.Type {
	case "auto":
		return "auto", true
	case "any":
		lg.debugf(dbgXlate, `mapped tool_choice "any" to upstream "required"`)
		return "required", true
	case "none":
		return "none", true
	case "tool":
		if tc.Name == "" {
			lg.debugf(dbgXlate, `tool_choice type "tool" without a name; dropping`)
			return nil, false
		}
		return map[string]any{
			"type":     "function",
			"function": map[string]string{"name": tc.Name},
		}, true
	default:
		lg.debugf(dbgXlate, "unknown tool_choice type %q; dropping", tc.Type)
		return nil, false
	}
}

func compact(raw json.RawMessage) string {
	var buf bytes.Buffer
	if err := json.Compact(&buf, raw); err != nil {
		return string(raw)
	}
	return buf.String()
}

func mustMarshal(v any) json.RawMessage {
	b, err := json.Marshal(v)
	if err != nil {
		return json.RawMessage(`{}`)
	}
	return b
}
