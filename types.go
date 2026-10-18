package main

import "encoding/json"

// ---------------------------------------------------------------------------
// Anthropic Messages API — what Claude Code sends us and expects back.
// ---------------------------------------------------------------------------

// anthropicRequest is the inbound POST /v1/messages body.
//
// `System` and `Messages[].Content` are kept as json.RawMessage because both
// are polymorphic: a plain string or an array of content blocks. Claude Code in
// particular sends `system` as an array of text blocks carrying cache_control,
// even though the simple docs show a bare string.
type anthropicRequest struct {
	Model         string             `json:"model"`
	MaxTokens     int                `json:"max_tokens"`
	System        json.RawMessage    `json:"system,omitempty"`
	Messages      []anthropicMessage `json:"messages"`
	Tools         []anthropicTool    `json:"tools,omitempty"`
	ToolChoice    json.RawMessage    `json:"tool_choice,omitempty"`
	Stream        bool               `json:"stream,omitempty"`
	Temperature   *float64           `json:"temperature,omitempty"`
	TopP          *float64           `json:"top_p,omitempty"`
	TopK          *int               `json:"top_k,omitempty"`
	StopSequences []string           `json:"stop_sequences,omitempty"`
	Thinking      json.RawMessage    `json:"thinking,omitempty"`
	Metadata      json.RawMessage    `json:"metadata,omitempty"`
}

type anthropicMessage struct {
	Role    string          `json:"role"`
	Content json.RawMessage `json:"content"`
}

// anthropicBlock is the union of every content block type we accept inbound.
type anthropicBlock struct {
	Type string `json:"type"`

	// type: text
	Text string `json:"text,omitempty"`

	// type: tool_use
	ID    string          `json:"id,omitempty"`
	Name  string          `json:"name,omitempty"`
	Input json.RawMessage `json:"input,omitempty"`

	// type: tool_result
	ToolUseID string          `json:"tool_use_id,omitempty"`
	Content   json.RawMessage `json:"content,omitempty"`
	IsError   bool            `json:"is_error,omitempty"`

	// type: image
	Source *anthropicImageSource `json:"source,omitempty"`

	// type: thinking / redacted_thinking
	Thinking string `json:"thinking,omitempty"`
}

type anthropicImageSource struct {
	Type      string `json:"type"` // "base64" | "url"
	MediaType string `json:"media_type,omitempty"`
	Data      string `json:"data,omitempty"`
	URL       string `json:"url,omitempty"`
}

type anthropicTool struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	InputSchema json.RawMessage `json:"input_schema,omitempty"`
	Type        string          `json:"type,omitempty"`
}

// --- outbound (response) shapes ---

// anthropicRespBlock is a content block in a completed message.
type anthropicRespBlock struct {
	Type string `json:"type"`
	// Pointer so a text block can carry an explicit "text":"" (what real
	// Anthropic sends for empty output) while tool_use blocks omit the field.
	Text  *string         `json:"text,omitempty"`
	ID    string          `json:"id,omitempty"`
	Name  string          `json:"name,omitempty"`
	Input json.RawMessage `json:"input,omitempty"`
}

type anthropicUsage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// anthropicResponse is both the non-streaming reply body and the `message`
// envelope inside a streaming `message_start` event.
//
// StopReason/StopSequence are pointers so they serialize as JSON null (which
// `message_start` requires) rather than "".
type anthropicResponse struct {
	ID           string               `json:"id"`
	Type         string               `json:"type"`
	Role         string               `json:"role"`
	Model        string               `json:"model"`
	Content      []anthropicRespBlock `json:"content"`
	StopReason   *string              `json:"stop_reason"`
	StopSequence *string              `json:"stop_sequence"`
	Usage        anthropicUsage       `json:"usage"`
}

// --- streaming event shapes ---

type evtMessageStart struct {
	Type    string            `json:"type"`
	Message anthropicResponse `json:"message"`
}

// startBlock mirrors anthropicRespBlock but with Text as a *string so that a
// text block start emits `"text":""` instead of dropping the field.
type startBlock struct {
	Type  string          `json:"type"`
	Text  *string         `json:"text,omitempty"`
	ID    string          `json:"id,omitempty"`
	Name  string          `json:"name,omitempty"`
	Input json.RawMessage `json:"input,omitempty"`
}

type evtContentBlockStart struct {
	Type         string     `json:"type"`
	Index        int        `json:"index"`
	ContentBlock startBlock `json:"content_block"`
}

type textDeltaBody struct {
	Type string `json:"type"` // "text_delta"
	Text string `json:"text"`
}

type jsonDeltaBody struct {
	Type        string `json:"type"` // "input_json_delta"
	PartialJSON string `json:"partial_json"`
}

type evtContentBlockDelta struct {
	Type  string `json:"type"`
	Index int    `json:"index"`
	Delta any    `json:"delta"`
}

type evtContentBlockStop struct {
	Type  string `json:"type"`
	Index int    `json:"index"`
}

type messageDeltaBody struct {
	StopReason   *string `json:"stop_reason"`
	StopSequence *string `json:"stop_sequence"`
}

type evtMessageDelta struct {
	Type  string           `json:"type"`
	Delta messageDeltaBody `json:"delta"`
	Usage anthropicUsage   `json:"usage"`
}

type evtMessageStop struct {
	Type string `json:"type"`
}

// --- error shape ---

type anthropicErrorBody struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

type anthropicErrorEnvelope struct {
	Type  string             `json:"type"`
	Error anthropicErrorBody `json:"error"`
}

// ---------------------------------------------------------------------------
// OpenAI-compatible chat completions — what Z.ai speaks.
// ---------------------------------------------------------------------------

type oaiRequest struct {
	Model       string       `json:"model"`
	Messages    []oaiMessage `json:"messages"`
	Tools       []oaiTool    `json:"tools,omitempty"`
	ToolChoice  any          `json:"tool_choice,omitempty"`
	Stream      bool         `json:"stream,omitempty"`
	MaxTokens   int          `json:"max_tokens,omitempty"`
	Temperature *float64     `json:"temperature,omitempty"`
	TopP        *float64     `json:"top_p,omitempty"`
	Stop        []string     `json:"stop,omitempty"`
}

// oaiMessage.Content is `any` so it can be a string, an array (multimodal), or
// omitted entirely for an assistant turn that is purely tool calls.
type oaiMessage struct {
	Role       string        `json:"role"`
	Content    any           `json:"content,omitempty"`
	ToolCalls  []oaiToolCall `json:"tool_calls,omitempty"`
	ToolCallID string        `json:"tool_call_id,omitempty"`
}

type oaiToolCall struct {
	Index    int     `json:"index,omitempty"`
	ID       string  `json:"id,omitempty"`
	Type     string  `json:"type,omitempty"`
	Function oaiFunc `json:"function"`
}

type oaiFunc struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

type oaiTool struct {
	Type     string      `json:"type"`
	Function oaiToolFunc `json:"function"`
}

type oaiToolFunc struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

// oaiTextContent / oaiImageContent are used only when a user message carries an
// image and therefore needs array-form content.
type oaiTextContent struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type oaiImageURL struct {
	URL string `json:"url"`
}

type oaiImageContent struct {
	Type     string      `json:"type"`
	ImageURL oaiImageURL `json:"image_url"`
}

// --- upstream response / stream chunk ---

// oaiResponse covers both the non-streaming body and a single stream chunk;
// Message is set in the former, Delta in the latter.
type oaiResponse struct {
	ID      string      `json:"id"`
	Model   string      `json:"model"`
	Object  string      `json:"object"`
	Choices []oaiChoice `json:"choices"`
	Usage   *oaiUsage   `json:"usage,omitempty"`
	Error   *oaiError   `json:"error,omitempty"`
}

type oaiChoice struct {
	Index        int             `json:"index"`
	FinishReason string          `json:"finish_reason"`
	Message      *oaiRespMessage `json:"message,omitempty"`
	Delta        *oaiRespMessage `json:"delta,omitempty"`
}

// oaiRespMessage serves as both `message` and `delta`. Content is RawMessage
// because providers send a string, null, or (rarely) an array of parts — a
// hard-typed string would make one odd chunk fail the whole response.
type oaiRespMessage struct {
	Role             string          `json:"role,omitempty"`
	Content          json.RawMessage `json:"content,omitempty"`
	ReasoningContent json.RawMessage `json:"reasoning_content,omitempty"`
	ToolCalls        []oaiToolCall   `json:"tool_calls,omitempty"`
}

type oaiUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// oaiError.Code is RawMessage because Z.ai sends a quoted string ("1305")
// while OpenAI proper sends a bare int or null.
type oaiError struct {
	Code    json.RawMessage `json:"code,omitempty"`
	Message string          `json:"message"`
	Type    string          `json:"type,omitempty"`
}
