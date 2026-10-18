#!/usr/bin/env python3
"""Mock OpenAI-compatible upstream for exercising translation paths that the
live GLM endpoint does not produce (fragmented tool arguments, parallel tool
calls, late names, odd finish_reasons, missing usage, mid-stream errors,
truncated streams). Scenario is selected by the `model` field in the request."""
import json
from http.server import BaseHTTPRequestHandler, HTTPServer


def chunk(**delta_kw):
    """One OpenAI-style stream chunk."""
    d = {"choices": [{"index": 0, "delta": delta_kw}]}
    return "data: " + json.dumps(d) + "\n\n"


def finish(reason, usage=True):
    c = {"choices": [{"index": 0, "finish_reason": reason, "delta": {"content": ""}}]}
    if usage:
        c["usage"] = {"prompt_tokens": 42, "completion_tokens": 7, "total_tokens": 49}
    return "data: " + json.dumps(c) + "\n\n"


def tc(index, *, id=None, name=None, args=None):
    e = {"index": index}
    if id is not None:
        e["id"] = id
    if name is not None:
        e["type"] = "function"
    fn = {}
    if name is not None:
        fn["name"] = name
    if args is not None:
        fn["arguments"] = args
    if fn:
        e["function"] = fn
    return e


SCENARIOS = {}


# Standard OpenAI contract: arguments arrive as fragments of a JSON string.
SCENARIOS["frag"] = [
    chunk(role="assistant", content=""),
    chunk(content="Let me check."),
    chunk(tool_calls=[tc(0, id="call_frag1", name="get_weather", args="")]),
    chunk(tool_calls=[tc(0, args='{"loc')]),
    chunk(tool_calls=[tc(0, args='ation": "San ')]),
    chunk(tool_calls=[tc(0, args='Francisco, CA"')]),
    chunk(tool_calls=[tc(0, args="}")]),
    finish("tool_calls"),
    "data: [DONE]\n\n",
]

# Two parallel tool calls, interleaved fragments, no leading text.
SCENARIOS["parallel"] = [
    chunk(role="assistant"),
    chunk(tool_calls=[tc(0, id="call_a", name="get_weather", args='{"location"')]),
    chunk(tool_calls=[tc(1, id="call_b", name="get_time", args='{"tz"')]),
    chunk(tool_calls=[tc(0, args=': "SF"}')]),
    chunk(tool_calls=[tc(1, args=': "UTC"}')]),
    finish("tool_calls"),
    "data: [DONE]\n\n",
]

# Name arrives in a later delta than the first fragment for that index.
SCENARIOS["name-late"] = [
    chunk(role="assistant"),
    chunk(tool_calls=[tc(0, id="call_late", args='{"x":')]),
    chunk(tool_calls=[tc(0, name="do_thing", args=" 1}")]),
    finish("tool_calls"),
    "data: [DONE]\n\n",
]

# Three tool calls, heavily interleaved, one with a late name, plus leading
# text: the worst case for the one-open-block rule.
SCENARIOS["parallel3"] = [
    chunk(role="assistant", content="working"),
    chunk(tool_calls=[tc(0, id="c0", name="alpha", args='{"a"')]),
    chunk(tool_calls=[tc(1, id="c1", args='{"b"')]),  # name not yet known
    chunk(tool_calls=[tc(2, id="c2", name="gamma", args='{"c"')]),
    chunk(tool_calls=[tc(1, name="beta")]),
    chunk(tool_calls=[tc(2, args=": 3}")]),
    chunk(tool_calls=[tc(0, args=": 1}")]),
    chunk(tool_calls=[tc(1, args=": 2}")]),
    finish("tool_calls"),
    "data: [DONE]\n\n",
]

# A finish_reason outside the documented set.
SCENARIOS["weird-finish"] = [
    chunk(role="assistant", content="filtered"),
    finish("content_filter"),
    "data: [DONE]\n\n",
]

# Provider reports no usage at all -> proxy must estimate.
SCENARIOS["no-usage"] = [
    chunk(role="assistant", content="no usage here"),
    finish("stop", usage=False),
    "data: [DONE]\n\n",
]

# Error object arrives mid-stream, after a 200 + partial content.
SCENARIOS["mid-error"] = [
    chunk(role="assistant", content="partial..."),
    'data: {"error":{"code":"1302","message":"API call frequency exceeded"}}\n\n',
]

# Stream just stops: no finish_reason, no [DONE].
SCENARIOS["no-done"] = [
    chunk(role="assistant", content="truncated"),
]

# Model returns nothing at all.
SCENARIOS["empty"] = [
    chunk(role="assistant", content=""),
    finish("stop"),
    "data: [DONE]\n\n",
]

# Text after a tool call -> must open a second text block at a new index.
SCENARIOS["text-after-tool"] = [
    chunk(role="assistant", content="before "),
    chunk(tool_calls=[tc(0, id="call_x", name="f", args="{}")]),
    chunk(content="after"),
    finish("stop"),
    "data: [DONE]\n\n",
]

# Non-streaming: tool_calls present but finish_reason says "stop".
NONSTREAM = {
    "stop-with-tools": {
        "id": "x", "model": "mock-model", "object": "chat.completion",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {
            "role": "assistant", "content": None,
            "tool_calls": [{"id": "call_z", "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"location":"SF"}'}}]}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 6, "total_tokens": 11},
    },
    "bad-args": {
        "id": "x", "model": "mock-model",
        "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": "",
            "tool_calls": [{"id": "call_bad", "type": "function",
                            "function": {"name": "f", "arguments": "{not valid json"}}]}}],
    },
    "weird-finish-ns": {
        "id": "x", "model": "mock-model",
        "choices": [{"index": 0, "finish_reason": "banana",
                     "message": {"role": "assistant", "content": "hi"}}],
    },
    "empty-ns": {
        "id": "x", "model": "mock-model",
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": None}}],
    },
}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        model = body.get("model", "")

        if model in NONSTREAM:
            payload = json.dumps(NONSTREAM[model]).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        events = SCENARIOS.get(model)
        if events is None:
            payload = json.dumps({"error": {"code": "404", "message": f"unknown scenario {model!r}"}}).encode()
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        for e in events:
            self.wfile.write(e.encode())
            self.wfile.flush()
        self.close_connection = True


if __name__ == "__main__":
    HTTPServer(("127.0.0.1", 9400), Handler).serve_forever()
