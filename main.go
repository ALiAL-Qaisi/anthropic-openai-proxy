package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"strings"
	"time"
)

const maxRequestBody = 128 << 20 // 128 MiB — Claude Code contexts get large.

type proxy struct {
	cfg          *config
	streamClient *http.Client // no overall deadline: SSE responses are long-lived
	jsonClient   *http.Client
}

func main() {
	log.SetFlags(log.Ldate | log.Ltime | log.Lmicroseconds)

	cfg, err := loadConfig()
	if err != nil {
		log.Printf("FATAL: %v", err)
		log.Printf("Set it and retry, e.g.:  UPSTREAM_API_KEY=<your-z.ai-key> ./anthropic-openai-proxy")
		os.Exit(1)
	}

	transport := &http.Transport{
		Proxy:                 http.ProxyFromEnvironment,
		DialContext:           (&net.Dialer{Timeout: 15 * time.Second, KeepAlive: 30 * time.Second}).DialContext,
		TLSHandshakeTimeout:   15 * time.Second,
		ResponseHeaderTimeout: 120 * time.Second,
		IdleConnTimeout:       90 * time.Second,
		MaxIdleConnsPerHost:   16,
		ForceAttemptHTTP2:     true,
	}

	p := &proxy{
		cfg:          cfg,
		streamClient: &http.Client{Transport: transport}, // Timeout 0 = unlimited
		jsonClient:   &http.Client{Transport: transport, Timeout: 10 * time.Minute},
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/v1/messages", p.handleMessages)
	mux.HandleFunc("/v1/messages/count_tokens", p.handleCountTokens)
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"status":"ok"}`))
	})
	mux.HandleFunc("/", p.handleNotFound)

	srv := &http.Server{
		Addr:              ":" + cfg.port,
		Handler:           mux,
		ReadHeaderTimeout: 30 * time.Second,
		// No WriteTimeout: it would cut long streaming responses short.
	}

	log.Printf("anthropic-openai-proxy listening on http://localhost:%s", cfg.port)
	log.Printf("  upstream : %s/chat/completions", cfg.upstreamBaseURL)
	log.Printf("  model    : %s (forced on every request)", cfg.upstreamModel)
	log.Printf("  api key  : present (%d chars, never logged)", len(cfg.upstreamAPIKey))
	log.Printf("  debug    : %v", cfg.debug)
	log.Printf("Point Claude Code at it with ANTHROPIC_BASE_URL=http://localhost:%s", cfg.port)

	if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Printf("FATAL: server stopped: %v", err)
		os.Exit(1)
	}
}

// ---------------------------------------------------------------------------
// Anthropic-shaped error replies
// ---------------------------------------------------------------------------

// errorTypeForStatus maps an HTTP status onto Anthropic's error `type` taxonomy
// so Claude Code's own error handling (retry vs. surface) behaves sanely.
func errorTypeForStatus(status int) string {
	switch {
	case status == http.StatusBadRequest:
		return "invalid_request_error"
	case status == http.StatusUnauthorized:
		return "authentication_error"
	case status == http.StatusForbidden:
		return "permission_error"
	case status == http.StatusNotFound:
		return "not_found_error"
	case status == http.StatusRequestEntityTooLarge:
		return "request_too_large"
	case status == http.StatusTooManyRequests:
		return "rate_limit_error"
	case status == 529:
		return "overloaded_error"
	case status >= 500:
		return "api_error"
	default:
		return "api_error"
	}
}

// writeAnthropicError emits {"type":"error","error":{...}} — the shape Claude
// Code expects for any failure. Never a bare Go error string or a panic.
func (p *proxy) writeAnthropicError(w http.ResponseWriter, lg *reqLogger, status int, errType, msg string) {
	if errType == "" {
		errType = errorTypeForStatus(status)
	}
	body, err := json.Marshal(anthropicErrorEnvelope{
		Type:  "error",
		Error: anthropicErrorBody{Type: errType, Message: msg},
	})
	if err != nil {
		body = []byte(`{"type":"error","error":{"type":"api_error","message":"internal error"}}`)
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_, _ = w.Write(body)
	if lg != nil {
		lg.debugBlob(dbgOut, fmt.Sprintf("error response (HTTP %d)", status), body)
	}
}

func (p *proxy) handleNotFound(w http.ResponseWriter, r *http.Request) {
	log.Printf("%s %s -> 404 (unhandled path)", r.Method, r.URL.Path)
	p.writeAnthropicError(w, nil, http.StatusNotFound, "not_found_error",
		fmt.Sprintf("Unrecognized path %q. This proxy implements POST /v1/messages.", r.URL.Path))
}

// ---------------------------------------------------------------------------
// POST /v1/messages
// ---------------------------------------------------------------------------

func (p *proxy) handleMessages(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	lg := newReqLogger(p.cfg.debug)

	if r.Method != http.MethodPost {
		p.writeAnthropicError(w, lg, http.StatusMethodNotAllowed, "invalid_request_error",
			"Method not allowed; use POST /v1/messages.")
		log.Printf("%s %s -> 405", r.Method, r.URL.Path)
		return
	}

	// Note which auth header the client sent, but never its value — and never
	// use it upstream. Upstream auth always comes from UPSTREAM_API_KEY.
	lg.debugf(dbgIn, "%s %s | x-api-key: %s | authorization: %s | anthropic-version: %s",
		r.Method, r.URL.Path,
		presence(r.Header.Get("x-api-key")),
		presence(r.Header.Get("Authorization")),
		valueOrAbsent(r.Header.Get("anthropic-version")))

	raw, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxRequestBody))
	if err != nil {
		lg.errorf("reading request body: %v", err)
		p.writeAnthropicError(w, lg, http.StatusBadRequest, "invalid_request_error",
			"Could not read request body: "+err.Error())
		log.Printf("POST %s -> 400 (unreadable body)", r.URL.Path)
		return
	}
	lg.debugBlob(dbgIn, "Anthropic request body", raw)

	var areq anthropicRequest
	if err := json.Unmarshal(raw, &areq); err != nil {
		lg.errorf("parsing Anthropic request: %v", err)
		p.writeAnthropicError(w, lg, http.StatusBadRequest, "invalid_request_error",
			"Request body is not valid JSON for the Messages API: "+err.Error())
		log.Printf("POST %s -> 400 (bad JSON)", r.URL.Path)
		return
	}
	if len(areq.Messages) == 0 {
		p.writeAnthropicError(w, lg, http.StatusBadRequest, "invalid_request_error",
			"`messages` must contain at least one message.")
		log.Printf("POST %s -> 400 (empty messages)", r.URL.Path)
		return
	}

	oreq, err := translateRequest(p.cfg, lg, &areq)
	if err != nil {
		lg.errorf("translating request: %v", err)
		p.writeAnthropicError(w, lg, http.StatusBadRequest, "invalid_request_error",
			"Could not translate request for the upstream provider: "+err.Error())
		log.Printf("POST %s -> 400 (translation failed)", r.URL.Path)
		return
	}

	outBody, err := json.Marshal(oreq)
	if err != nil {
		lg.errorf("marshalling upstream request: %v", err)
		p.writeAnthropicError(w, lg, http.StatusInternalServerError, "api_error",
			"Could not encode upstream request: "+err.Error())
		return
	}

	url := p.cfg.upstreamBaseURL + "/chat/completions"
	lg.debugf(dbgUpOut, "POST %s | Authorization: Bearer <redacted> (present, %d chars)",
		url, len(p.cfg.upstreamAPIKey))
	lg.debugBlob(dbgUpOut, "upstream request body", outBody)

	upReq, err := http.NewRequestWithContext(r.Context(), http.MethodPost, url, bytes.NewReader(outBody))
	if err != nil {
		lg.errorf("building upstream request: %v", err)
		p.writeAnthropicError(w, lg, http.StatusInternalServerError, "api_error",
			"Could not build upstream request: "+err.Error())
		return
	}
	upReq.Header.Set("Content-Type", "application/json")
	upReq.Header.Set("Authorization", "Bearer "+p.cfg.upstreamAPIKey)
	upReq.Header.Set("Accept", acceptFor(areq.Stream))

	client := p.jsonClient
	if areq.Stream {
		client = p.streamClient
	}

	resp, err := client.Do(upReq)
	if err != nil {
		// Connection refused, DNS failure, bad UPSTREAM_BASE_URL, TLS error…
		lg.errorf("upstream request failed: %v", err)
		p.writeAnthropicError(w, lg, http.StatusBadGateway, "api_error",
			fmt.Sprintf("Upstream request to %s failed: %v", url, err))
		log.Printf("POST %s -> upstream unreachable (%v) in %s", r.URL.Path, err, time.Since(start).Round(time.Millisecond))
		return
	}
	defer resp.Body.Close()

	lg.debugf(dbgUpIn, "HTTP %d | content-type: %s", resp.StatusCode, resp.Header.Get("Content-Type"))

	// Any non-2xx is translated into an Anthropic error before we commit to a
	// streaming response, so the client always gets a clean JSON error body.
	if resp.StatusCode < 200 || resp.StatusCode > 299 {
		errBody, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		lg.debugBlob(dbgUpIn, "upstream error body", errBody)
		msg := extractUpstreamError(errBody)
		status := resp.StatusCode
		if status < 400 || status > 599 {
			status = http.StatusBadGateway
		}
		p.writeAnthropicError(w, lg, status, errorTypeForStatus(status),
			fmt.Sprintf("Upstream provider returned HTTP %d: %s", resp.StatusCode, msg))
		log.Printf("POST %s -> upstream %d in %s", r.URL.Path, resp.StatusCode, time.Since(start).Round(time.Millisecond))
		return
	}

	if areq.Stream {
		p.streamResponse(w, r, lg, resp, &areq)
		log.Printf("POST %s -> upstream %d (stream) in %s", r.URL.Path, resp.StatusCode, time.Since(start).Round(time.Millisecond))
		return
	}

	p.nonStreamResponse(w, lg, resp, &areq)
	log.Printf("POST %s -> upstream %d in %s", r.URL.Path, resp.StatusCode, time.Since(start).Round(time.Millisecond))
}

func acceptFor(stream bool) string {
	if stream {
		return "text/event-stream"
	}
	return "application/json"
}

func presence(v string) string {
	if v == "" {
		return "absent"
	}
	return fmt.Sprintf("present (%d chars, value not logged)", len(v))
}

func valueOrAbsent(v string) string {
	if v == "" {
		return "absent"
	}
	return v
}

// extractUpstreamError pulls a human message out of whatever error shape the
// provider used, falling back to the raw body.
func extractUpstreamError(body []byte) string {
	if len(body) == 0 {
		return "(empty response body)"
	}
	var env struct {
		Error   *oaiError `json:"error"`
		Msg     string    `json:"msg"`
		Message string    `json:"message"`
	}
	if err := json.Unmarshal(body, &env); err == nil {
		if env.Error != nil && env.Error.Message != "" {
			if len(env.Error.Code) > 0 {
				return fmt.Sprintf("%s (code %s)", env.Error.Message, strings.Trim(string(env.Error.Code), `"`))
			}
			return env.Error.Message
		}
		if env.Message != "" {
			return env.Message
		}
		if env.Msg != "" {
			return env.Msg
		}
	}
	s := strings.TrimSpace(string(body))
	if len(s) > 800 {
		s = s[:800] + "…(truncated)"
	}
	return s
}

// ---------------------------------------------------------------------------
// POST /v1/messages/count_tokens
//
// Not in the Messages-translation spec, but Claude Code calls this endpoint to
// size a context before sending it. Z.ai exposes no token-counting API, so we
// answer with the same chars/4 estimate used elsewhere — a missing endpoint
// here shows up as a visible error in Claude Code.
// ---------------------------------------------------------------------------

func (p *proxy) handleCountTokens(w http.ResponseWriter, r *http.Request) {
	lg := newReqLogger(p.cfg.debug)
	if r.Method != http.MethodPost {
		p.writeAnthropicError(w, lg, http.StatusMethodNotAllowed, "invalid_request_error", "Method not allowed; use POST.")
		return
	}
	raw, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxRequestBody))
	if err != nil {
		p.writeAnthropicError(w, lg, http.StatusBadRequest, "invalid_request_error", "Could not read request body: "+err.Error())
		return
	}
	lg.debugBlob(dbgIn, "count_tokens request body", raw)

	var areq anthropicRequest
	if err := json.Unmarshal(raw, &areq); err != nil {
		p.writeAnthropicError(w, lg, http.StatusBadRequest, "invalid_request_error", "Request body is not valid JSON: "+err.Error())
		return
	}
	n := estimateTokens(approxRequestChars(&areq))
	body, _ := json.Marshal(map[string]int{"input_tokens": n})
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write(body)
	lg.debugf(dbgOut, "count_tokens -> %d (chars/4 estimate; provider has no counting API)", n)
	log.Printf("POST %s -> 200 (estimated %d input tokens)", r.URL.Path, n)
}

// approxRequestChars totals the characters we are about to send upstream, used
// for token estimates when the provider reports no usage.
func approxRequestChars(areq *anthropicRequest) int {
	n := len(rawToText(areq.System))
	for _, m := range areq.Messages {
		n += len(m.Content) // raw JSON length: close enough for an estimate
	}
	for _, t := range areq.Tools {
		n += len(t.Name) + len(t.Description) + len(t.InputSchema)
	}
	return n
}
