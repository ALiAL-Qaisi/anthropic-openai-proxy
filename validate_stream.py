#!/usr/bin/env python3
"""Validate proxy SSE output against the Anthropic streaming contract.

Reads an SSE stream on stdin, reconstructs the message, and asserts the
structural invariants Claude Code depends on:
  - every line is `event:`/`data:`/blank, data is valid JSON, name == data.type
  - message_start first, message_stop last, exactly one message_delta
  - blocks open/close in order, indices sequential from 0, none left open
  - accumulated input_json_delta fragments parse as JSON
"""
import json
import sys

raw = sys.stdin.read()
lines = raw.split("\n")
evs = []
problems = []

i = 0
while i < len(lines):
    line = lines[i]
    if line == "":
        i += 1
        continue
    if line.startswith("event: "):
        name = line[7:]
        if i + 1 >= len(lines) or not lines[i + 1].startswith("data: "):
            problems.append(f"event {name!r} not followed by a data: line")
            i += 1
            continue
        try:
            payload = json.loads(lines[i + 1][6:])
        except Exception as e:
            problems.append(f"invalid JSON in data for {name!r}: {e}")
            i += 2
            continue
        if payload.get("type") != name:
            problems.append(f"event name {name!r} != data.type {payload.get('type')!r}")
        evs.append((name, payload))
        i += 2
    else:
        problems.append(f"unexpected line: {line!r}")
        i += 1

names = [n for n, _ in evs]
if not names:
    print("FAIL: no events at all")
    sys.exit(1)

# Order invariants
if names[0] != "message_start":
    problems.append(f"first event is {names[0]!r}, expected message_start")
if names[-1] not in ("message_stop", "error"):
    problems.append(f"last event is {names[-1]!r}, expected message_stop (or error)")
if "error" not in names:
    if names.count("message_delta") != 1:
        problems.append(f"expected exactly 1 message_delta, got {names.count('message_delta')}")
    if names.count("message_start") != 1:
        problems.append(f"expected exactly 1 message_start, got {names.count('message_start')}")
    # every block must be closed before message_delta
    if "message_delta" in names:
        md = names.index("message_delta")
        for n, p in evs[md + 1:]:
            if n.startswith("content_block"):
                problems.append(f"{n} at index {p.get('index')} came after message_delta")

# Block bookkeeping
blocks, order, open_now = {}, [], None
for n, p in evs:
    if n == "content_block_start":
        idx = p["index"]
        if idx in blocks:
            problems.append(f"content_block_start repeated for index {idx}")
        if open_now is not None:
            problems.append(f"content_block_start index {idx} while index {open_now} still open")
        blocks[idx] = {"type": p["content_block"]["type"], "cb": p["content_block"], "text": "", "json": "", "stopped": False}
        order.append(idx)
        open_now = idx
    elif n == "content_block_delta":
        idx = p["index"]
        if idx not in blocks:
            problems.append(f"content_block_delta for unopened index {idx}")
            continue
        if blocks[idx]["stopped"]:
            problems.append(f"content_block_delta for already-stopped index {idx}")
        d = p["delta"]
        if d["type"] == "text_delta":
            if blocks[idx]["type"] != "text":
                problems.append(f"text_delta on a {blocks[idx]['type']} block (index {idx})")
            blocks[idx]["text"] += d["text"]
        elif d["type"] == "input_json_delta":
            if blocks[idx]["type"] != "tool_use":
                problems.append(f"input_json_delta on a {blocks[idx]['type']} block (index {idx})")
            blocks[idx]["json"] += d["partial_json"]
        else:
            problems.append(f"unknown delta type {d['type']!r}")
    elif n == "content_block_stop":
        idx = p["index"]
        if idx not in blocks:
            problems.append(f"content_block_stop for unopened index {idx}")
            continue
        blocks[idx]["stopped"] = True
        open_now = None

if open_now is not None:
    problems.append(f"block index {open_now} never stopped")
if order != sorted(order) or (order and order != list(range(len(order)))):
    problems.append(f"block indices not sequential from 0: {order}")

# Tool inputs must parse
tools = []
for idx in order:
    b = blocks[idx]
    if b["type"] != "tool_use":
        continue
    cb = b["cb"]
    if not cb.get("name"):
        problems.append(f"tool_use block {idx} has no name")
    if not cb.get("id"):
        problems.append(f"tool_use block {idx} has no id")
    if cb.get("input") != {}:
        problems.append(f"tool_use block {idx} content_block_start input should be {{}}, got {cb.get('input')}")
    parsed = None
    if b["json"] == "":
        parsed = {}
    else:
        try:
            parsed = json.loads(b["json"])
        except Exception as e:
            problems.append(f"tool_use block {idx} accumulated partial_json does not parse: {e} raw={b['json']!r}")
    tools.append((idx, cb.get("name"), cb.get("id"), parsed))

# Report
print(f"events: {len(evs)}  sequence: {' '.join(n for k, n in enumerate(names) if k == 0 or names[k-1] != n)}")
for idx in order:
    b = blocks[idx]
    if b["type"] == "text":
        print(f"  block {idx} text     : {b['text']!r}")
    else:
        print(f"  block {idx} tool_use : name={b['cb'].get('name')!r} id={b['cb'].get('id')!r}")
        print(f"            fragments  : {b['json']!r}")
for idx, name, tid, parsed in tools:
    print(f"  -> tool {name} input parsed: {json.dumps(parsed)}")
md = [p for n, p in evs if n == "message_delta"]
if md:
    print(f"  message_delta: stop_reason={md[0]['delta']['stop_reason']!r} usage={json.dumps(md[0].get('usage'))}")
err = [p for n, p in evs if n == "error"]
if err:
    print(f"  error event: {json.dumps(err[0])}")

if problems:
    print("FAIL (%d problem(s)):" % len(problems))
    for p in problems:
        print("   -", p)
    sys.exit(1)
print("PASS: stream satisfies the Anthropic streaming contract")
