## Problem Statement

M.A.E.S.T.R.O. has 207 passing unit tests, 73% coverage, clean lint, and comprehensive CI — but all tests run against the deterministic `echo` backend. We don't know if it works with real LLMs (Anthropic, Ollama, OpenAI-compatible) on non-trivial tasks. The goal is to validate three target swarms end-to-end with real providers.

## Solution

Validate M.A.E.S.T.R.O. works with real LLMs by:
1. Fixing mypy core errors and bandit findings (clean baseline)
2. Setting up Ollama with `llama3.1:8b` (general) + `qwen2.5-coder:7b` (code-gen)
3. Writing integration tests + a standalone validation script
4. Running validation against real providers, producing dated reports
5. Adding example specs for the validated swarms
6. Updating docs (README, validation.md, examples/)
7. Adding CI integration (Ollama in GitHub Actions, gated by env var)

## Stories

### US-001 — Validate supervisor swarm decomposes and synthesizes with real LLM

**Type:** User Story

As a developer using M.A.E.S.T.R.O., I want the supervisor topology to actually decompose tasks into sensible subtasks (not fan-out) and synthesize worker outputs into a coherent answer, so that I can trust it for real work.

### US-002 — Validate parallel ensemble produces diverse outputs aggregated meaningfully

**Type:** User Story

As a developer using M.A.E.S.T.R.O., I want parallel workers to produce diverse outputs that an aggregator meaningfully synthesizes, so that ensemble swarms are useful.

### US-003 — Validate tool calling works end-to-end across providers

**Type:** User Story

As a developer using M.A.E.S.T.R.O., I want native function calling and ReAct fallback to work correctly on real backends, so that agents can use tools reliably.

### US-004 — Validate streaming doesn't break aggregation or token accounting

**Type:** User Story

As a developer using M.A.E.S.T.R.O., I want streaming tokens to be forwarded correctly while SwarmResult remains byte-identical, so that observability doesn't change behavior.

### US-005 — Validate code generation swarm (supervisor → coder → reviewer → tester)

**Type:** User Story

As a developer, I want a code-gen swarm that implements a binary search tree with tests, so that I can use M.A.E.S.T.R.O. for programming tasks.

### US-006 — Validate decision pipeline (router topology with LLM agents)

**Type:** User Story

As a developer, I want a router that routes 10 varied prompts to correct specialists, so that I can use M.A.E.S.T.R.O. for task routing.

### US-007 — Validate debate swarm (3 debaters + LLM judge)

**Type:** User Story

As a developer, I want a debate swarm that argues a technical question across rounds with a judge, so that I can explore contentious decisions.

### DS-001 — Validation reports are dated, comparable, and machine-readable

**Type:** Data Story

As a developer, I want each validation run to produce a Markdown report (human) and JSONL log (machine), so that I can track regressions over time.

### AS-001 — CI runs integration tests with Ollama when enabled

**Type:** Agent Story

As a maintainer, I want GitHub Actions to install Ollama, pull `llama3.1:8b`, and run integration tests when `OLLAMA_HOST` is set, so that regressions are caught automatically.

## Implementation Decisions

1. **Provider for validation**: Ollama (local, zero-cost, no API keys). Models: `llama3.1:8b` (general swarm roles) + `qwen2.5-coder:7b` (code-gen swarm specifically). CI uses only `llama3.1:8b` for speed.

2. **Mypy strategy**: Suppress optional-dep stub errors (`laya`, `onnxruntime`, `typesafe_sdk`) with `# type: ignore[import-not-found]`; fix ~22 core errors in `decision_agent.py`, `system1_serve.py`, `cli.py`, `parallel.py`, `builtin.py`, `registry.py`.

3. **Bandit SHA1 fix**: Use `hashlib.sha1(..., usedforsecurity=False).hexdigest()[:8]` in `EchoClient._render` (Python 3.9+, clean, signals intent).

4. **Test structure**: 
   - `tests/integration/` with `@pytest.mark.integration` marker (pytest for CI regression)
   - `scripts/validate_real_llms.py` standalone script (human-readable reports)
   - Both produce Markdown + JSONL output

5. **CI integration**: GitHub Actions `ubuntu-latest`, install Ollama via script, pull `llama3.1:8b` (~4.7GB, ~5 min), run integration tests only when `OLLAMA_HOST` env var is set.

6. **Decision-model agents (Laya/Jev)**: Deferred. Validate router topology with LLM agents first; decision agents are a separate subsystem.

7. **Debate judge**: LLM judge (same providers as other swarms), not decision model.

8. **Documentation updates**: 
   - README: Ollama quickstart, real-provider examples
   - `docs/validation.md`: dated results, model-specific notes, known issues
   - `examples/`: three new validated specs (`codegen_swarm.yaml`, `decision_pipeline.yaml`, `debate_swarm.yaml`)

9. **Validation tasks** (acceptance criteria):
   - Code-gen: "Implement a binary search tree with insert/delete/search in Python, with tests"
   - Decision pipeline: Route 10 varied prompts (code, research, chat, analysis) → correct specialist
   - Debate: "Best architecture for a local-first agent framework?" (3 debaters + LLM judge)

10. **Execution order**: Fix mypy/bandit → Pull Ollama models → Write integration tests + validation script → Run validation → Add example specs → Update docs → CI integration → Local qwen validation.

## Testing Decisions

- **External behavior only**: Tests verify SwarmResult outputs (final answer, per-agent outputs, usage, trace), not internal implementation.
- **Modules tested**: Orchestrator, all topologies (supervisor, parallel, debate, router), provider resolution, tool calling (native + ReAct), streaming, usage accounting.
- **Seam mapping**:
  - BS for US-001/US-002/US-005 → Verify through `Orchestrator.run()` with real Ollama backend
  - BS for US-003 → Verify through `LLMAgent._run_native()` and `_run_react()` with real providers
  - BS for US-004 → Verify through `on_token` callback + `SwarmResult` byte-identity check
  - BS for US-006 → Verify through `RouterTopology.run()` with real LLM agents
  - BS for US-007 → Verify through `DebateTopology.run()` with real LLM agents
  - BS for DS-001 → Verify `scripts/validate_real_llms.py` produces both Markdown and JSONL
  - BS for AS-001 → Verify CI workflow runs and passes when `OLLAMA_HOST` is set
- **Prior art**: Existing `test_streaming.py`, `test_tool_calling.py`, `test_topologies.py` patterns — extend with real-provider fixtures.

## Out of Scope

- Commercial API keys (Anthropic, OpenAI) — not required for validation
- Jev API key / TypeSafe SDK — decision agents deferred
- Laya / Laya ONNX local installation — decision agents deferred
- MCP server validation for I.S.A.A.C. / O.L.I.V.I.A. — separate effort
- Performance benchmarking / latency optimization
- Multi-machine / distributed swarm execution

## Further Notes

- This spec was synthesized from a grilling session (`/grill-with-docs`) and SAIL skills setup (`/setup-sail-skills`).
- The `ready-for-agent` label means the spec is complete and an agent can start implementation.
- First implementation step: mypy/bandit fixes (no external deps).
- Validation script output format: Markdown report with task, swarm config, per-agent outputs, synthesis, tokens, cost, pass/fail + JSONL log (one line per agent turn).