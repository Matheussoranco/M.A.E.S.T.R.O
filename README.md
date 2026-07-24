# M.A.E.S.T.R.O.

**Multi-Agent Ensemble Swarm Task-Routing Orchestrator** — a provider-agnostic
orchestrator that *instantiates and conducts agent swarms*, in the spirit of the
multi-agent systems from MiniMax and Kimi, but open and yours to configure.

Describe a swarm as plain data; MAESTRO brings it to life. Any agent may be
backed by **any** LLM — Anthropic, OpenAI, any OpenAI-compatible gateway (Groq,
Together, OpenRouter, vLLM, LM Studio, llama.cpp), or a fully **local** Ollama
model — and a swarm member can be a native LLM agent, an arbitrary command-line
program, an MCP server, or a sibling project such as **I.S.A.A.C.** and
**O.L.I.V.I.A.** enlisted as first-class members.

> The entire core runtime is **pure Python standard library** — no required
> third-party dependencies. It imports, runs, and passes its whole test suite
> **offline, with no API keys**, thanks to a deterministic `echo` fallback.

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
maestro topologies               # sequential | parallel | supervisor | debate | router
maestro providers                # which backends are configured right now
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

- **Provider** — a backend implementing one method, `complete()`. Built in:
  `anthropic`, `openai`, `openai_compat`, `ollama`, `llamacpp`, `echo`, `null`.
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
print(result.tracer.render())    # timeline of the run
```

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

## MAESTRO as an MCP server

Expose the orchestrator itself so Claude (or a parent swarm) can spin up swarms:

```jsonc
// .mcp.json
{ "mcpServers": { "maestro": { "command": "maestro", "args": ["mcp-serve"] } } }
```

Tools: `maestro_run_demo`, `maestro_run_spec`, `maestro_validate`,
`maestro_list_topologies`.

## Testing

```bash
pytest            # fully offline, deterministic, no keys required
ruff check .
```

## Layout

```
src/maestro/
  providers/     anthropic · openai(-compat) · ollama · echo · null  (stdlib transport)
  agents/        llm · cli · mcp · isaac/olivia presets · registry
  topologies/    sequential · parallel · supervisor · debate · router
  swarm/         Swarm · Blackboard · Message · RunContext
  orchestrator/  Spec (parse/validate) · Orchestrator
  tools/         safe built-in tools for the ReAct loop
  mcp/           MAESTRO's own MCP server
  telemetry/     run tracer
```

## License

[PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0)
© 2026 Matheus Soranzo. Part of the I.S.A.A.C. / O.L.I.V.I.A. agent ecosystem.

You may use, modify, and share this software for any noncommercial purpose; commercial use is not permitted.
