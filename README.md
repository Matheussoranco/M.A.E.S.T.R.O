# M.A.E.S.T.R.O.

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC_BY--NC--SA_4.0-lightgrey.svg)](LICENSE)

**Multi-Agent Ensemble Swarm Task-Routing Orchestrator** — a provider-agnostic
orchestrator that *instantiates and conducts agent swarms*, in the spirit of the
multi-agent systems from MiniMax and Kimi, and is source-available for you to
configure. It is licensed CC BY-NC-SA 4.0, including a noncommercial restriction.

Describe a swarm as plain data; MAESTRO brings it to life. Any agent may be
backed by **any** LLM — Anthropic, OpenAI, any OpenAI-compatible gateway (Groq,
Together, OpenRouter, vLLM, LM Studio, llama.cpp), or a fully **local** Ollama
model — and a swarm member can be a native LLM agent, an arbitrary command-line
program, an MCP server, or a sibling project such as **I.S.A.A.C.** and
**O.L.I.V.I.A.** enlisted as first-class members.

> The entire core runtime is **pure Python standard library** — no required
> third-party dependencies (streaming included: it rides on `urllib`). It
> imports, runs, and passes its whole test suite **offline, with no API keys**,
> thanks to a deterministic `echo` fallback.

---

## Why

Two commitments make MAESTRO different from a hard-coded agent app:

| Commitment | What it buys you |
|---|---|
| **Provider-agnostic** | Mix a strong cloud "lead" with cheap local workers in the same swarm. Swap `anthropic` → `ollama` by editing one line of a spec. |
| **Agent-agnostic** | A member is anything that takes a task and returns text: an LLM, a CLI, an MCP tool, or a whole other agent project. |
| **Declarative** | Swarms are *data* (YAML/JSON/dict). The same runtime instantiates any swarm you can describe — that's the "instantiator". |
| **Graceful degradation** | No key? Unreachable host? The swarm still runs on the deterministic `echo` backend so you can see its control-flow. |

## Install

```bash
# core is dependency-free; extras add YAML specs and the dev toolchain
pip install -e ".[dev]"          # from this directory
```

## 60-second tour

```bash
maestro demo                     # runs a full supervisor→workers→synthesis swarm, offline
maestro demo --stream --usage    # …streaming its tokens, and costing the run
maestro topologies               # sequential | parallel | supervisor | debate | router
maestro providers                # which backends are configured right now
maestro prices                   # the token price table used for cost totals
maestro validate examples/research_swarm.yaml
maestro describe  examples/isaac_olivia_swarm.yaml
maestro run examples/local_ollama_swarm.json "Should we build or buy an agent stack?"
```

## Concepts

```
Orchestrator ──reads──▶ Spec (YAML/JSON/dict)
     │  instantiates
     ▼
   Swarm ── agents[] + Topology + shared Blackboard
     │  conducts
     ▼
 Topology  decides who runs, in what order, who sees whose output
```

- **Provider** — a backend implementing `complete()` (plus optional streaming and
  native tool calling). Built in: `anthropic`, `openai`, `openai_compat`,
  `ollama`, `llamacpp`, `echo`, `null`.
- **Agent** — a swarm member. Types: `llm`, `cli`, `mcp`, `isaac`, `olivia`.
- **Topology** — the composition pattern:
  - `sequential` — a pipeline (draft → critique → polish).
  - `parallel` — fan-out ensemble, optionally reduced by an aggregator.
  - `supervisor` — a lead decomposes → workers execute → lead synthesizes (the
    canonical swarm).
  - `debate` — debaters argue across rounds; an optional judge decides.
  - `router` — pick the single best specialist and run only it.

## Writing a swarm spec

```yaml
name: research-swarm
topology: supervisor
topology_params: { supervisor: lead }

providers:
  claude: { provider: anthropic, model: claude-sonnet-5 }
  local:  { provider: ollama,    model: llama3.1 }

agents:
  - { name: lead,     type: llm,   provider: claude, role: supervisor }
  - { name: research, type: llm,   provider: local,  role: researcher }
  - { name: analyst,  type: llm,   provider: local,  role: analyst, tools: [calc, now] }
  - { name: isaac,    type: isaac }        # I.S.A.A.C. joins the swarm
  - { name: olivia,   type: olivia }       # O.L.I.V.I.A. joins the swarm
```

```bash
maestro run research-swarm.yaml "Compare markerless vs marker-based gait analysis."
```

Or from Python:

```python
from maestro import Orchestrator
orch = Orchestrator.from_file("research-swarm.yaml")
result = orch.run("Compare markerless vs marker-based gait analysis.")
print(result.final)
print(result.summary())          # per-agent status
print(result.usage.render())     # tokens & cost, per agent and per provider
print(result.tracer.render())    # timeline of the run
```

## Streaming

Streaming is **opt-in and observational**. Pass a sink and you receive fragments
as they are produced; the `SwarmResult` is byte-identical either way, so every
topology aggregates exactly what it would have without it.

```python
result = orch.run(task, on_token=lambda agent, text: print(f"[{agent}] {text}", end=""))
```

```bash
maestro run swarm.yaml "…" --stream
```

The contract lives on the provider interface, once:

```python
for event in client.complete_stream(messages, system="…"):
    if event.text:            # zero or more fragments; concatenating them
        ...                   # reproduces the completion
    if event.done:            # then exactly one terminal event
        response = event.response      # the same LLMResponse complete() returns
```

**Every backend satisfies it.** `anthropic`, `openai`(-compatible) and `ollama`
stream natively; anything else — including `echo` and any third-party client
written against the 0.1 interface — falls back to delivering the completion in
one chunk rather than raising. So callers never have to ask first;
`client.supports_streaming` only tells you whether it is genuinely incremental.
`collect_stream(events)` drains a stream down to its final `LLMResponse`.

## Tool calling: native, with a ReAct fallback

A tool is declared **once** — name, description, JSON Schema — and MAESTRO adapts
that declaration to whatever the agent's backend speaks:

| Backend | Protocol |
|---|---|
| `anthropic` | native `tool_use` / `tool_result` blocks |
| `openai`, `openai_compat`, `ollama` | native `tools` functions |
| anything else (incl. `echo`) | the plain-text `ACTION:` / `OBSERVATION:` convention |

Nothing in your spec changes between them — an agent uses the best protocol its
backend has. Pin one explicitly with `native_tools: true|false` when you want to
compare them, or force the portable path:

```yaml
agents:
  - { name: analyst, type: llm, provider: local, tools: [calc, now] }
  - { name: portable, type: llm, provider: local, tools: [calc], native_tools: false }
```

Custom tools subclass `Tool`. The default schema is a single string argument, so
simple tools work down both paths with no extra declaration; give `parameters` a
real JSON Schema (and override `call()`) when a tool needs structured arguments.

## Token & cost totals

`SwarmResult.usage` rolls every backend call up into one report — total input and
output tokens and cost, broken down **per agent** and **per provider**:

```python
u = result.usage
u.input_tokens, u.output_tokens, u.cost      # cost is None when it can't be known
u.per_agent["lead"].total_tokens
u.per_provider["anthropic"].cost
print(u.render())
```

```bash
maestro run swarm.yaml "…" --usage
```

Two things it is careful about:

- **Unknown is not zero.** A backend that reports no usage (many gateways don't)
  is counted as *unmeasured*, never as having used nothing — `Usage` counts are
  `int | None`, and `tokens_complete` tells you whether the total is a real
  measurement or a floor. A provider that reports `0` is a measurement.
- **Unpriced is not free.** Costs come from a price table matched against the
  model id (`maestro prices`). A model that isn't in it has an *unknown* cost, and
  a total that could only be partly priced says so (`cost_complete`). Fill the
  gaps per spec — including the honest `0` for models you host yourself:

```yaml
prices:                              # US$ per million tokens
  llama3.1: { input: 0, output: 0 }
```

Or globally: `maestro.telemetry.usage.register_price("my-model", 0.5, 1.5)`.

The rollup counts what you were *billed*, not what survived: a supervisor's
planning turn and a router's routing turn never reach `per_agent`, but their
tokens are in the totals.

## Enlisting I.S.A.A.C. and O.L.I.V.I.A.

Both ship non-interactive entry points, so they drop straight into a swarm:

- **I.S.A.A.C.** → `isaac agent "<task>"` (neuro-symbolic reasoning / synthesis)
- **O.L.I.V.I.A.** → `olivia ask "<question>"` (research, study, discovery)

If they live in their own virtualenvs, point MAESTRO at the right interpreter:

```bash
export MAESTRO_ISAAC_CMD="C:/Users/mathe/Documents/Code/I.S.A.A.C/.venv/Scripts/python.exe -m isaac agent"
export MAESTRO_OLIVIA_CMD="C:/Users/mathe/Documents/Code/O.L.I.V.I.A/.venv/Scripts/python.exe -m olivia ask"
maestro run examples/isaac_olivia_swarm.yaml "Solve a novel ARC task, then teach it."
```

MCP mode (`type: isaac, mode: mcp`) drives their MCP servers instead of the CLI.

## Configuration (environment)

| Variable | Purpose |
|---|---|
| `MAESTRO_PROVIDER` | default provider when a spec omits one (default `echo`) |
| `MAESTRO_MODEL` | default model |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | credentials (also `MAESTRO_*` prefixed) |
| `MAESTRO_OLLAMA_BASE_URL` | local Ollama endpoint (default `http://localhost:11434`) |
| `MAESTRO_OPENAI_BASE_URL` | point OpenAI-compatible clients at any gateway |
| `MAESTRO_ALLOW_STUB` | fall back to `echo` when a backend is unavailable (default on) |
| `MAESTRO_ISAAC_CMD` / `MAESTRO_OLIVIA_CMD` | how to launch the sibling agents |
| `MAESTRO_MCP_SPEC_ROOTS` | `os.pathsep`-separated directories allowed to MCP `run_spec`/`validate` (defaults to the server working directory) |

## MAESTRO as an MCP server

Expose the orchestrator itself so Claude (or a parent swarm) can spin up swarms:

```jsonc
// .mcp.json
{ "mcpServers": { "maestro": { "command": "maestro", "args": ["mcp-serve"] } } }
```

Tools: `maestro_run_demo`, `maestro_run_spec`, `maestro_validate`,
`maestro_list_topologies`. The MCP server is a local, trusted-process interface;
spec execution can launch external commands, so keep its allowlist narrow and do
not expose its stdio process to untrusted callers.

## Examples

Eight bundled specs, covering every topology (`maestro examples` lists them):

| Spec | Shows |
|---|---|
| `research_swarm.yaml` | supervisor; mixed cloud lead + local workers; tools |
| `debate_swarm.yaml` | debate across rounds with a judge |
| `local_ollama_swarm.json` | fully-local parallel ensemble (JSON — no PyYAML needed) |
| `isaac_olivia_swarm.yaml` | I.S.A.A.C. and O.L.I.V.I.A. as swarm members |
| `streaming_swarm.yaml` | **streaming** every stage of a pipeline |
| `cost_report_swarm.yaml` | **cost aggregation**, incl. unpriced/unmeasured handling |
| `tool_calling_swarm.yaml` | native tool calling *and* the ReAct fallback in one run |
| `support_router_swarm.yaml` | router picking a single specialist |

## Testing

```bash
pytest            # fully offline, deterministic, no keys required
ruff check .
```

CI runs the suite on **Python 3.10–3.13 across Linux, Windows and macOS**, with
no secrets configured. A separate `zero-deps` job installs the package with no
extras, asserts that pip pulled in nothing third-party, and then runs the real
entry points — so the dependency-free promise is enforced, not just asserted.

## Layout

```
src/maestro/
  providers/     anthropic · openai(-compat) · ollama · echo · null  (stdlib transport)
                 complete() · complete_stream() · native tool calling
  agents/        llm · cli · mcp · isaac/olivia presets · registry
  topologies/    sequential · parallel · supervisor · debate · router
  swarm/         Swarm · Blackboard · Message · RunContext
  orchestrator/  Spec (parse/validate) · Orchestrator
  tools/         built-in tools · one declaration, adapted per provider
  mcp/           MAESTRO's own MCP server
  telemetry/     run tracer · token & cost accounting
```

## License

[**CC BY-NC-SA 4.0**](https://creativecommons.org/licenses/by-nc-sa/4.0/) — see [LICENSE](LICENSE).

Copyright © 2026 Matheus Soranzo <matheussoranco@gmail.com>
`SPDX-License-Identifier: CC-BY-NC-SA-4.0`
Part of the I.S.A.A.C. / O.L.I.V.I.A. agent ecosystem.

You may use, study, modify and share M.A.E.S.T.R.O. for any noncommercial
purpose, on three conditions:

- **NonCommercial** — not primarily for or directed towards commercial advantage
  or monetary compensation (§1(k)).
- **ShareAlike** — anything you share onward, including modified versions, must
  carry these same terms. This is a copyleft licence (§3(b)).
- **Attribution** — keep the copyright notice, the licence reference and the
  warranty disclaimer, say if you changed it, and link back where practicable (§3(a)).

Commercial use requires separate written permission from the copyright holder.
Note the licence grants no patent or trademark rights (§2(b)(2)).

This project was previously MIT-licensed; anyone who received it under those
terms keeps their MIT rights to that version.
