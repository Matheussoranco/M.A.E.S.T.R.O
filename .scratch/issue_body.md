## Problem Statement

M.A.E.S.T.R.O. has 207 passing tests, 73% coverage, and clean lint — but all tests run against the deterministic `echo` backend. We don't know if it works end-to-end with real LLMs on non-trivial tasks. Need to validate three target swarms with real providers (Ollama locally, CI-gated) so we can confidently use it for research and development.

## Solution

Run a phased validation:
1. Fix mypy/bandit issues (clean baseline)
2. Pull Ollama models (`llama3.1:8b`, `qwen2.5-coder:7b`)
3. Write integration tests + standalone validation script for three swarms
4. Execute validation locally → dated report (Markdown + JSONL)
5. Add example specs for validated swarms
6. Update docs (README, `docs/validation.md`)
7. CI integration (Ollama in GitHub Actions, gated by env var)

## Stories

### AS-001 — Validate supervisor code-gen swarm end-to-end

**Type:** Agent Story

- **Agent goal:** Produce correct, tested code from a natural-language task via a supervisor→coder→reviewer→tester pipeline
- **Possible tasks:**
  - Implement a binary search tree with insert/delete/search in Python, with tests
  - Refactor a 200-line function into smaller typed functions
  - Write a FastAPI endpoint with Pydantic models, auth dependency, and unit tests
- **Available artifacts:**
  - **Tools:** built-in tools (calc, now, etc.), native function calling + ReAct fallback
  - **Skills:** supervisor topology planning, parallel worker execution, synthesis
  - **Context / knowledge:** swarm spec defines roles and provider routing
  - **Memory:** blackboard shares context across agents
- **Autonomy:** Swarm runs autonomously once invoked; human reviews final output and validation report
- **Expected performance:** All 4 agents complete without error; synthesis produces compilable, test-passing code; token/cost accounting accurate

### AS-002 — Validate router decision pipeline end-to-end

**Type:** Agent Story

- **Agent goal:** Route varied prompts to the correct specialist agent and synthesize a coherent answer
- **Possible tasks:**
  - Route 10 varied prompts (code, research, chat, analysis) → correct specialist
  - Verify routing accuracy and synthesis quality
- **Available artifacts:**
  - **Tools:** router topology, LLM agents with role-specific system prompts
  - **Skills:** router topology selection, single-specialist execution
  - **Context / knowledge:** agent roles defined in spec
- **Autonomy:** Fully autonomous routing and execution
- **Expected performance:** Router picks correct specialist for ≥8/10 prompts; synthesis integrates specialist output; no hallucinated routing

### AS-003 — Validate debate swarm with LLM judge end-to-end

**Type:** Agent Story

- **Agent goal:** Multiple debaters argue across rounds; LLM judge selects winning position
- **Possible tasks:**
  - "Best architecture for a local-first agent framework?" (technical debate)
  - 3 debaters + 1 LLM judge, 3 rounds
- **Available artifacts:**
  - **Tools:** debate topology, judge agent
  - **Skills:** multi-round debate, structured argument, judge decision
  - **Context / knowledge:** debate topic and rounds in topology params
- **Autonomy:** Fully autonomous debate rounds and judging
- **Expected performance:** All debaters produce distinct arguments; judge produces reasoned decision; trace shows round-by-round progression

### US-001 — Run validation script and get dated report

**Type:** User Story

As a developer, I want to run a single command that executes all three validation swarms against real LLMs and produces a dated Markdown report with per-agent outputs, token usage, cost, and pass/fail status, so that I can verify M.A.E.S.T.R.O. works and share results.

### US-002 — Read validation docs with known issues and model-specific notes

**Type:** User Story

As a developer, I want a `docs/validation.md` file with dated results, known issues, and model-specific notes (Ollama vs cloud providers), so that I can understand current validation state and limitations.

### US-003 — Run validated example specs with one command

**Type:** User Story

As a developer, I want three new example specs in `examples/` (`codegen_swarm.yaml`, `decision_pipeline.yaml`, `debate_swarm.yaml`) that I can run with `maestro run` and they work without modification, so that I have working templates for my own swarms.

## Implementation Decisions

- Fix mypy core errors (~22) first: suppress optional-dep stub errors (`laya`, `onnxruntime`, `typesafe_sdk`) with `# type: ignore[import-not-found]`; fix real type issues in `decision_agent.py`, `system1_serve.py`, `cli.py`, `parallel.py`, `builtin.py`, `registry.py`
- Fix bandit SHA1 in `EchoClient._render`: use `hashlib.sha1(..., usedforsecurity=False)` (Python 3.9+)
- Ollama models: `llama3.1:8b` (general) + `qwen2.5-coder:7b` (code-gen only, local validation)
- Integration tests in `tests/integration/` with `@pytest.mark.integration` marker; skipped by default, run when `OLLAMA_HOST` env var set
- Standalone validation script: `scripts/validate_real_llms.py` → produces Markdown report + JSONL log
- CI: GitHub Actions `ubuntu-latest`, install Ollama, pull `llama3.1:8b` only (CI speed), run integration tests when `OLLAMA_HOST` set
- Example specs: `examples/codegen_swarm.yaml` (supervisor), `examples/decision_pipeline.yaml` (router), `examples/debate_swarm.yaml` (debate with LLM judge)
- Docs: Update README with Ollama quickstart; add `docs/validation.md` with dated results

## Testing Decisions

- Integration tests verify: swarm completes without error, agents produce expected output types, token accounting non-null, trace captures all agents
- For each BS entry (below), verify through the public `maestro run` CLI boundary
- Existing test suite (207 tests) must pass before and after changes
- Validation script output compared across runs via JSONL diff

## Behavioral Specifications

### BS-001 — Code-gen swarm produces working code

**Stories:** AS-001, US-001, US-003

**Audience:** developer-facing

```gherkin
Feature: Code generation swarm validation

  Scenario: Supervisor decomposes task and workers produce compilable code
    Given a codegen_swarm.yaml spec with supervisor topology
    And Ollama is running with llama3.1:8b and qwen2.5-coder:7b
    When the swarm runs "Implement a binary search tree with insert/delete/search in Python, with tests"
    Then all 4 agents (lead, coder, reviewer, tester) complete without error
    And the synthesis output contains valid Python code
    And the code passes the generated tests
    And token usage is reported per agent and per provider
```

### BS-002 — Decision pipeline routes correctly

**Stories:** AS-002, US-001, US-003

**Audience:** developer-facing

```gherkin
Feature: Decision pipeline validation

  Scenario: Router picks correct specialist for varied prompts
    Given a decision_pipeline.yaml spec with router topology
    And Ollama is running with llama3.1:8b
    When the swarm runs each of 10 varied prompts (code, research, chat, analysis)
    Then the router selects a specialist agent for each prompt
    And the selected specialist matches the prompt domain for at least 8 of 10
    And the final synthesis integrates the specialist output
    And token usage is reported
```

### BS-003 — Debate swarm produces judged decision

**Stories:** AS-003, US-001, US-003

**Audience:** developer-facing

```gherkin
Feature: Debate swarm validation

  Scenario: Three debaters argue and judge decides
    Given a debate_swarm.yaml spec with debate topology and judge
    And Ollama is running with llama3.1:8b
    When the swarm runs "Best architecture for a local-first agent framework?"
    Then 3 debaters produce distinct arguments across 3 rounds
    And the judge agent produces a reasoned decision
    And the trace shows round-by-round progression
    And token usage is reported per agent
```

### BS-004 — Validation script produces dated report

**Stories:** US-001, US-002

**Audience:** developer-facing

```gherkin
Feature: Validation script output

  Scenario: Single command produces Markdown report and JSONL log
    Given Ollama is running with required models
    When running `python scripts/validate_real_llms.py`
    Then a dated Markdown report is written with per-agent outputs, tokens, cost, pass/fail
    And a JSONL log is written with one line per agent turn
    And both files are timestamped for comparison across runs
```

## Out of Scope

- Commercial API keys (Anthropic, OpenAI) — not required for validation
- Decision-model agents (Laya/Jev) — validated later if needed
- Jev API key setup — only if hosted decision models wanted
- Multi-provider swarms in CI — CI uses single Ollama model for speed
- Performance benchmarking — only functional correctness validated

## Further Notes

- This spec was synthesized from a grilling session (7 rounds) that established shared understanding
- The `ready-for-agent` triage label should be applied
- Implementation order: mypy/bandit → Ollama setup → integration tests → validation script → example specs → docs → CI
- Validation reports are dated and comparable; first run establishes baseline