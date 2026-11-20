#!/usr/bin/env python3
"""
Run 10 varied prompts through the decision pipeline router and verify routing accuracy.

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


# 10 varied prompts with expected specialists
TEST_CASES = [
    {
        "prompt": "Write a Python function to parse JSON and handle missing keys gracefully",
        "expected": "coder",
        "category": "code_generation"
    },
    {
        "prompt": "Debug this Python code: it raises KeyError when the API returns null",
        "expected": "coder",
        "category": "debugging"
    },
    {
        "prompt": "Research the latest developments in transformer architectures for LLMs",
        "expected": "researcher",
        "category": "research"
    },
    {
        "prompt": "Find and summarize academic papers on few-shot learning published in 2024",
        "expected": "researcher",
        "category": "literature_review"
    },
    {
        "prompt": "Analyze this sales data CSV and compute monthly growth rates",
        "expected": "analyst",
        "category": "data_analysis"
    },
    {
        "prompt": "Create a dashboard showing user retention metrics with SQL queries",
        "expected": "analyst",
        "category": "reporting"
    },
    {
        "prompt": "Explain how photosynthesis works in simple terms",
        "expected": "general",
        "category": "explanation"
    },
    {
        "prompt": "What are the best practices for writing effective prompts for LLMs?",
        "expected": "general",
        "category": "qa"
    },
    {
        "prompt": "Refactor this legacy JavaScript code to use modern ES6+ patterns",
        "expected": "coder",
        "category": "refactoring"
    },
    {
        "prompt": "Compare the trade-offs between SQL and NoSQL databases for a logging system",
        "expected": "researcher",
        "category": "analysis"
    },
]


def run_routing_test(spec_path: str, test_cases: list[dict]) -> dict:
    """Run all test cases through the orchestrator and return results."""
    print(f"[validate] Loading spec from {spec_path}...")
    orch = Orchestrator.from_file(spec_path)
    print(f"[validate] Swarm loaded:\n{orch.describe()}")

    # Check availability
    avail = orch.availability()
    unavailable = [name for name, ok in avail.items() if not ok]
    if unavailable:
        msg = f"Agents unavailable: {unavailable}. Is Ollama running with llama3.1:8b?"
        print(f"[validate] ERROR: {msg}")
        return {"success": False, "error": msg, "results": []}

    print(f"[validate] All {len(avail)} agents available. Starting routing tests...\n")

    results = []
    correct = 0
    fallback_count = 0

    for i, case in enumerate(test_cases, 1):
        prompt = case["prompt"]
        expected = case["expected"]
        category = case["category"]

        print(f"[validate] Test {i}/10 [{category}]: {prompt[:60]}...")
        print(f"[validate]   Expected: {expected}")

        # Run with trace enabled
        result = orch.run(prompt, trace=True)

        # Extract routed agent from trace
        routed_agent = None
        if result.tracer:
            for event in result.tracer.to_list():
                if event.get("event") == "route":
                    routed_agent = event.get("detail", "").replace("→ ", "").strip()
                    break

        # Fallback: check which agent actually ran (only one in router topology)
        if not routed_agent and result.per_agent:
            routed_agent = result.per_agent[0].name

        is_correct = routed_agent == expected
        if is_correct:
            correct += 1
            status = "✅"
        else:
            status = "❌"

        print(f"[validate]   Routed to: {routed_agent} {status}")

        # Check if fallback was used (router agent failed/unavailable)
        used_fallback = False
        if result.tracer:
            for event in result.tracer.to_list():
                if event.get("event") == "agent_start" and event.get("name") == "router":
                    # Check if router produced output
                    pass
                # We'll infer fallback from router not being in per_agent
        if result.per_agent and result.per_agent[0].name != "router":
            # In router topology, only the chosen agent runs, not the router
            # But the router runs first internally. Let's check trace for router execution.
            pass

        results.append({
            "test_num": i,
            "category": category,
            "prompt": prompt,
            "expected": expected,
            "routed_to": routed_agent,
            "correct": is_correct,
            "result_ok": result.ok(),
            "final_output": result.final[:200] if result.final else "",
            "trace_events": result.tracer.to_list() if result.tracer else [],
        })

        print()

    accuracy = correct / len(test_cases) * 100
    print(f"[validate] Routing accuracy: {correct}/{len(test_cases)} = {accuracy:.1f}%")

    return {
        "success": True,
        "accuracy": accuracy,
        "correct": correct,
        "total": len(test_cases),
        "results": results,
    }


def generate_report(validation_result: dict, output_dir: Path, date_str: str, time_str: str) -> tuple[Path, Path]:
    """Generate markdown and JSONL reports."""
    base_name = f"decision_pipeline_validation_{date_str}_{time_str}"

    # Markdown report
    md_lines = [
        f"# Decision Pipeline Validation Report — {date_str}",
        "",
        f"**Timestamp:** {datetime.now().isoformat(timespec='seconds')}",
        f"**Accuracy:** {validation_result['accuracy']:.1f}% ({validation_result['correct']}/{validation_result['total']})",
        "",
        "## Test Results",
        "",
    ]

    for r in validation_result["results"]:
        status = "✅" if r["correct"] else "❌"
        md_lines.append(f"### {status} Test {r['test_num']} [{r['category']}]")
        md_lines.append(f"**Prompt:** {r['prompt']}")
        md_lines.append(f"**Expected:** {r['expected']}")
        md_lines.append(f"**Routed to:** {r['routed_to']}")
        md_lines.append(f"**Correct:** {r['correct']}")
        md_lines.append(f"**Swarm OK:** {r['result_ok']}")
        if r["final_output"]:
            md_lines.append(f"**Output preview:** {r['final_output']}")
        md_lines.append("")

    # Trace summary
    md_lines.append("## Trace Summary")
    md_lines.append("")
    for r in validation_result["results"]:
        md_lines.append(f"**Test {r['test_num']} [{r['category']}]**")
        for event in r["trace_events"]:
            md_lines.append(f"- {event.get('event', '?')}: {event.get('detail', '')}")
        md_lines.append("")

    markdown_report = "\n".join(md_lines)

    # JSONL record
    jsonl_record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "date": date_str,
        "spec": "examples/decision_pipeline.yaml",
        "topology": "router",
        "accuracy": validation_result["accuracy"],
        "correct": validation_result["correct"],
        "total": validation_result["total"],
        "tests": [
            {
                "test_num": r["test_num"],
                "category": r["category"],
                "prompt": r["prompt"],
                "expected": r["expected"],
                "routed_to": r["routed_to"],
                "correct": r["correct"],
                "result_ok": r["result_ok"],
            }
            for r in validation_result["results"]
        ],
    }

    jsonl_line = json.dumps(jsonl_record, ensure_ascii=False)

    md_path = output_dir / f"{base_name}.md"
    md_path.write_text(markdown_report, encoding="utf-8")

    jsonl_path = output_dir / f"{date_str}_decision_pipeline_validations.jsonl"
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(jsonl_line + "\n")

    return md_path, jsonl_path


def main():
    parser = argparse.ArgumentParser(
        description="Run decision pipeline routing validation with Ollama and produce reports"
    )
    parser.add_argument(
        "--spec",
        default="examples/decision_pipeline.yaml",
        help="Path to the decision pipeline spec (default: examples/decision_pipeline.yaml)",
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

    validation_result = run_routing_test(str(spec_path), TEST_CASES)

    if not validation_result["success"]:
        print("[validate] Validation failed.")
        sys.exit(1)

    # Write reports
    if not args.no_md or not args.no_jsonl:
        md_path, jsonl_path = generate_report(validation_result, output_dir, date_str, time_str)
        if not args.no_md:
            print(f"[validate] Markdown report written to {md_path}")
        if not args.no_jsonl:
            print(f"[validate] JSONL record appended to {jsonl_path}")

    # Exit with error code if accuracy is below threshold (e.g., 70%)
    if validation_result["accuracy"] < 70:
        print(f"[validate] WARNING: Routing accuracy {validation_result['accuracy']:.1f}% below 70% threshold")
        sys.exit(1)

    print("[validate] Validation completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())