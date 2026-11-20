#!/usr/bin/env python3
"""
Live validation script for MAESTRO debate swarm with real Ollama LLMs.

Runs the debate_swarm.yaml spec with Ollama (llama3.1:8b) and produces:
- Dated Markdown report with outcomes, outputs, traces, and token usage
- JSONL report with structured data for programmatic analysis

This is an opt-in test — it requires a running Ollama daemon with llama3.1:8b pulled.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# CRITICAL: Allow localhost/private URLs for Ollama BEFORE importing maestro
os.environ["MAESTRO_ALLOW_PRIVATE_URLS"] = "true"

# Ensure we can import maestro from the local source tree
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from maestro.orchestrator import Orchestrator


def run_debate(spec_path: str, task: str) -> tuple[bool, str, str]:
    """
    Run the debate swarm and return (success, markdown_report, jsonl_line).
    """
    print(f"[validate] Loading spec from {spec_path}...")
    orch = Orchestrator.from_file(spec_path)
    print(f"[validate] Swarm loaded:\n{orch.describe()}")

    # Check availability
    avail = orch.availability()
    unavailable = [name for name, ok in avail.items() if not ok]
    if unavailable:
        msg = f"Agents unavailable: {unavailable}. Is Ollama running with llama3.1:8b?"
        print(f"[validate] ERROR: {msg}")
        return False, "", ""

    print(f"[validate] All {len(avail)} agents available. Starting debate...")
    print(f"[validate] Task: {task}")

    # Run with trace enabled
    result = orch.run(task, trace=True)

    # Build markdown report
    timestamp = datetime.now().isoformat(timespec="seconds")
    date_str = datetime.now().strftime("%Y-%m-%d")

    md_lines = [
        f"# Debate Swarm Validation Report — {date_str}",
        "",
        f"**Timestamp:** {timestamp}",
        f"**Spec:** {spec_path}",
        f"**Task:** {task}",
        f"**Topology:** {result.topology}",
        f"**Overall Status:** {'SUCCESS' if result.ok() else 'FAILED'}",
        f"**Error:** {result.error if result.error else 'None'}",
        "",
        "## Final Verdict",
        "",
        result.final or "(no output)",
        "",
        "## Per-Agent Results",
        "",
    ]

    for r in result.per_agent:
        status = "✅" if r.ok() else "❌"
        md_lines.append(f"### {status} {r.name} ({r.role})")
        md_lines.append(f"**OK:** {r.ok()}")
        if r.error:
            md_lines.append(f"**Error:** {r.error}")
        md_lines.append(f"**Output:**")
        md_lines.append("")
        md_lines.append(r.output or "(no output)")
        md_lines.append("")
        if r.meta:
            md_lines.append(f"**Meta:** `{r.meta}`")
            md_lines.append("")

    # Trace
    if result.tracer:
        md_lines.append("## Execution Trace")
        md_lines.append("")
        md_lines.append("```")
        md_lines.append(result.tracer.render())
        md_lines.append("```")
        md_lines.append("")

    # Usage / Cost
    md_lines.append("## Token Usage & Cost")
    md_lines.append("")
    md_lines.append("```")
    md_lines.append(result.usage.render())
    md_lines.append("```")
    md_lines.append("")

    markdown_report = "\n".join(md_lines)

    # Build JSONL record
    jsonl_record = {
        "timestamp": timestamp,
        "date": date_str,
        "spec": spec_path,
        "task": task,
        "topology": result.topology,
        "success": result.ok(),
        "error": result.error,
        "final_output": result.final,
        "agents": [
            {
                "name": r.name,
                "role": r.role,
                "ok": r.ok(),
                "output": r.output,
                "error": r.error,
                "meta": r.meta,
            }
            for r in result.per_agent
        ],
        "usage": result.usage.as_dict(),
        "trace": result.tracer.to_list() if result.tracer else [],
    }

    jsonl_line = json.dumps(jsonl_record, ensure_ascii=False)

    return True, markdown_report, jsonl_line


def main():
    parser = argparse.ArgumentParser(
        description="Run live debate swarm validation with Ollama and produce reports"
    )
    parser.add_argument(
        "--spec",
        default="examples/debate_swarm.yaml",
        help="Path to the debate swarm spec (default: examples/debate_swarm.yaml)",
    )
    parser.add_argument(
        "--task",
        default="Best architecture for a local-first agent framework?",
        help="Task for the debate (default: 'Best architecture for a local-first agent framework?')",
    )
    parser.add_argument(
        "--output-dir",
        default="reports",
        help="Directory to write reports (default: reports/)",
    )
    parser.add_argument(
        "--no-md",
        action="store_true",
        help="Skip Markdown report generation",
    )
    parser.add_argument(
        "--no-jsonl",
        action="store_true",
        help="Skip JSONL report generation",
    )
    args = parser.parse_args()

    spec_path = Path(args.spec).resolve()
    if not spec_path.exists():
        print(f"ERROR: Spec not found: {spec_path}")
        sys.exit(2)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%H-%M-%S")
    base_name = f"debate_validation_{date_str}_{time_str}"

    success, markdown, jsonl_line = run_debate(str(spec_path), args.task)

    if not success:
        print("[validate] Debate run failed.")
        sys.exit(1)

    # Write Markdown report
    if not args.no_md:
        md_path = output_dir / f"{base_name}.md"
        md_path.write_text(markdown, encoding="utf-8")
        print(f"[validate] Markdown report written to {md_path}")

    # Write JSONL report (append mode for multiple runs)
    if not args.no_jsonl:
        jsonl_path = output_dir / f"{date_str}_debate_validations.jsonl"
        with jsonl_path.open("a", encoding="utf-8") as f:
            f.write(jsonl_line + "\n")
        print(f"[validate] JSONL record appended to {jsonl_path}")

    print("[validate] Validation completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())