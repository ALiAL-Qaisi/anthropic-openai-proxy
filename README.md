# anthropic-openai-proxy

A small translator between **Anthropic agents** (such as Claude Code) and **models that speak the OpenAI format**.

```text
Claude Code ──▶ anthropic-openai-proxy ──▶ OpenAI-compatible model
 (Anthropic)       (translator)            (your provider)
```

The agent thinks it is talking to Anthropic. The model thinks it is talking to an OpenAI client. The proxy translates messages, tool calls and streaming in both directions, so Claude Code can run on Z.ai's GLM models (the default) or on any other OpenAI-compatible provider.

## Build

You only need [Go](https://go.dev/dl/) (the minimum version is in `go.mod`). There are no other dependencies.

```bash
git clone https://github.com/ALiAL-Qaisi/anthropic-openai-proxy.git
cd anthropic-openai-proxy
go build -o anthropic-openai-proxy .
```

On Windows, use `-o anthropic-openai-proxy.exe` instead.

Prefer not to build? `go run .` compiles and starts it in one step.

## Configure

The proxy is configured **only with environment variables**.

| Variable | Required | Default | What it does |
|---|---|---|---|
| `UPSTREAM_API_KEY` | Yes | none | API key of your model provider, sent as a Bearer token. The proxy will not start without it. |
| `UPSTREAM_BASE_URL` | No | `https://api.z.ai/api/paas/v4` | Base URL of the OpenAI-compatible API, up to and including the version part (for example `.../v1`). The proxy adds `/chat/completions` itself. |
| `UPSTREAM_MODEL` | No | `glm-4.7-flash` | The model to use. It is sent on every request, whatever model name the agent asks for. |
| `PROXY_PORT` | No | `9090` | Port the proxy listens on. |
| `DEBUG` | No | `false` | Set to `1` or `true` to print every request and response. These logs contain your prompts and code, so keep them private. |

### How to set them

**Linux / macOS**

For one run only:

```bash
UPSTREAM_API_KEY="your-key" UPSTREAM_MODEL="your-model" ./anthropic-openai-proxy
```

For the whole terminal session:

```bash
export UPSTREAM_API_KEY="your-key"
export UPSTREAM_BASE_URL="https://api.example.com/v1"
export UPSTREAM_MODEL="your-model"
```

To keep them across sessions, add the `export` lines to `~/.bashrc` or `~/.zshrc`.

**Windows (PowerShell)**

```powershell
$env:UPSTREAM_API_KEY  = "your-key"
$env:UPSTREAM_BASE_URL = "https://api.example.com/v1"
$env:UPSTREAM_MODEL    = "your-model"
```

Variables set this way last until you close the window.

### Examples

**Z.ai (default)**: only the key is needed.

```bash
export UPSTREAM_API_KEY="your-zai-key"
```

**OpenRouter**

```bash
export UPSTREAM_BASE_URL="https://openrouter.ai/api/v1"
export UPSTREAM_API_KEY="your-openrouter-key"
export UPSTREAM_MODEL="model-id-from-openrouter"
```

**Ollama (local)**

```bash
export UPSTREAM_BASE_URL="http://localhost:11434/v1"
export UPSTREAM_API_KEY="ollama"   # any non-empty value; local servers usually ignore it
export UPSTREAM_MODEL="your-local-model"
```

Check your provider's documentation for its exact base URL and model names.

## Run

```bash
./anthropic-openai-proxy
```

On Windows: `.\anthropic-openai-proxy.exe`. Leave it running in its own terminal.

On startup it prints the settings it picked up, so you can confirm your variables were applied:

```text
anthropic-zai-proxy listening on http://localhost:9090
  upstream : https://api.z.ai/api/paas/v4/chat/completions
  model    : glm-4.7-flash (forced on every request)
  api key  : present (N chars, never logged)
  debug    : false
Point Claude Code at it with ANTHROPIC_BASE_URL=http://localhost:9090
```

## Connect Claude Code

Open a second terminal, point Claude Code at the proxy, and start it:

```bash
export ANTHROPIC_BASE_URL="http://localhost:9090"
export ANTHROPIC_AUTH_TOKEN="dummy"
claude
```

PowerShell:

```powershell
$env:ANTHROPIC_BASE_URL   = "http://localhost:9090"
$env:ANTHROPIC_AUTH_TOKEN = "dummy"
claude
```

| Variable | Value |
|---|---|
| `ANTHROPIC_BASE_URL` | The proxy address: `http://localhost:` plus your `PROXY_PORT`. Do not add `/v1`. |
| `ANTHROPIC_AUTH_TOKEN` | Any non-empty text. Claude Code needs a token to start; the proxy ignores it. Your real key stays in `UPSTREAM_API_KEY`. |

You do not need to set `ANTHROPIC_MODEL`: the proxy always uses `UPSTREAM_MODEL`.

## Good to know

- **The model must support tool calling.** Claude Code uses tools to read and edit files and run commands, so models without tool support will not be much use.
- **Some newer OpenAI models reject `max_tokens`** (they ask for `max_completion_tokens` instead), so they need a small code change before they work.
- **Built and tried against Z.ai (GLM).** Other OpenAI-compatible providers should work, but may need small tweaks.
- **Keep it private.** The proxy has no login of its own and listens on all network interfaces, so anyone who can reach the port can use your API key. Run it on your own machine or behind a firewall, and never commit your key.

