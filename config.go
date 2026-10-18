package main

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"
)

// Debug log tags. Each line is printed as [DEBUG][<tag>][<reqID>] so a single
// request's full round trip can be grepped by its request ID.
const (
	dbgIn    = "IN"           // inbound Anthropic request from Claude Code
	dbgUpOut = "UPSTREAM-OUT" // translated request we send to Z.ai
	dbgUpIn  = "UPSTREAM-IN"  // raw response/SSE lines from Z.ai
	dbgOut   = "OUT"          // translated response/SSE events to Claude Code
	dbgXlate = "XLATE"        // a translation step that fell back or guessed
)

type config struct {
	port            string
	upstreamBaseURL string
	upstreamAPIKey  string
	upstreamModel   string
	debug           bool
}

func envOr(key, def string) string {
	if v := strings.TrimSpace(os.Getenv(key)); v != "" {
		return v
	}
	return def
}

func loadConfig() (*config, error) {
	cfg := &config{
		port:            envOr("PROXY_PORT", "9090"),
		upstreamBaseURL: strings.TrimRight(envOr("UPSTREAM_BASE_URL", "https://api.z.ai/api/paas/v4"), "/"),
		upstreamAPIKey:  strings.TrimSpace(os.Getenv("UPSTREAM_API_KEY")),
		upstreamModel:   envOr("UPSTREAM_MODEL", "glm-4.7-flash"),
	}

	switch strings.ToLower(envOr("DEBUG", "false")) {
	case "1", "true", "yes", "on":
		cfg.debug = true
	}

	// Fail fast: without a key every request would 401 at the provider.
	if cfg.upstreamAPIKey == "" {
		return nil, fmt.Errorf("UPSTREAM_API_KEY is required but unset")
	}
	if _, err := strconv.Atoi(cfg.port); err != nil {
		return nil, fmt.Errorf("PROXY_PORT %q is not a valid port number", cfg.port)
	}
	return cfg, nil
}

// ---------------------------------------------------------------------------
// Per-request logger
// ---------------------------------------------------------------------------

type reqLogger struct {
	id    string
	debug bool
}

func newReqLogger(debug bool) *reqLogger {
	return &reqLogger{id: "req_" + randHex(4), debug: debug}
}

func (l *reqLogger) debugf(tag, format string, args ...any) {
	if !l.debug {
		return
	}
	log.Printf("[DEBUG][%s][%s] %s", tag, l.id, fmt.Sprintf(format, args...))
}

// debugBlob prints a multi-line payload on its own lines so pretty-printed
// JSON stays readable instead of being crammed onto one line.
func (l *reqLogger) debugBlob(tag, label string, body []byte) {
	if !l.debug {
		return
	}
	log.Printf("[DEBUG][%s][%s] %s:\n%s", tag, l.id, label, string(prettyJSON(body)))
}

// warnf always logs — fallbacks and guesses are rare and worth surfacing even
// when DEBUG is off.
func (l *reqLogger) warnf(format string, args ...any) {
	log.Printf("[WARN][%s] %s", l.id, fmt.Sprintf(format, args...))
}

func (l *reqLogger) errorf(format string, args ...any) {
	log.Printf("[ERROR][%s] %s", l.id, fmt.Sprintf(format, args...))
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

func randHex(n int) string {
	b := make([]byte, n)
	if _, err := rand.Read(b); err != nil {
		// crypto/rand failing is fatal-ish; a constant keeps us alive and the
		// ID is cosmetic (Claude Code does not interpret it).
		return strings.Repeat("0", n*2)
	}
	return hex.EncodeToString(b)
}

func newMessageID() string { return "msg_" + randHex(12) }
func newToolUseID() string { return "toolu_" + randHex(12) }

func prettyJSON(b []byte) []byte {
	var out strings.Builder
	var v any
	if err := json.Unmarshal(b, &v); err != nil {
		return b // not JSON (or truncated) — show as-is
	}
	enc := json.NewEncoder(&out)
	enc.SetIndent("", "  ")
	if err := enc.Encode(v); err != nil {
		return b
	}
	return []byte(strings.TrimRight(out.String(), "\n"))
}

func isJSONNull(raw json.RawMessage) bool {
	return len(raw) == 0 || string(raw) == "null"
}

// rawToText leniently extracts text from a polymorphic JSON value: a plain
// string, or an array of Anthropic-style blocks. Separate text blocks are
// joined with a blank line — Claude Code splits its system prompt into several
// blocks (for cache_control), and concatenating them bare would run the last
// sentence of one block into the first word of the next.
func rawToText(raw json.RawMessage) string {
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
			if b.Text != "" {
				parts = append(parts, b.Text)
			}
		}
		return strings.Join(parts, "\n\n")
	}
	return ""
}

// estimateTokens is a deliberately rough chars/4 heuristic, used only when the
// provider gives us no usage numbers. Claude Code does not depend on exactness.
func estimateTokens(chars int) int {
	if chars <= 0 {
		return 0
	}
	t := chars / 4
	if t < 1 {
		return 1
	}
	return t
}

func strPtr(s string) *string { return &s }
