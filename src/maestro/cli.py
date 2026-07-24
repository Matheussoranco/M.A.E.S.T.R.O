"""MAESTRO command-line interface (standard-library ``argparse`` only).

maestro demo                       run the offline demo swarm
maestro run <spec> "<task>"        instantiate a swarm from a spec and run it
maestro validate <spec>            check a spec without running it
maestro describe <spec>            show the swarm & agent availability
maestro providers                  list providers and which are configured
maestro topologies                 list available topologies
maestro agents                     list agent types and built-in tools
maestro mcp-serve                  expose MAESTRO itself over MCP (stdio)
maestro version
"""

from __future__ import annotations

import argparse
import sys

from maestro import __version__


def _print(*args) -> None:
    # Force UTF-8-safe printing on Windows terminals.
    text = " ".join(str(a) for a in args)
    try:
        print(text)
    except UnicodeEncodeError:  # pragma: no cover - console-dependent
        sys.stdout.buffer.write(text.encode("utf-8", "replace") + b"\n")


def _emit_result(result, as_json: bool) -> None:
    if as_json:
        import json

        payload = {
            "task": result.task,
            "topology": result.topology,
            "final": result.final,
            "error": result.error,
            "agents": [
                {"name": r.name, "role": r.role, "ok": r.ok(), "output": r.output, "error": r.error}
                for r in result.per_agent
            ],
            "trace": result.tracer.to_list() if result.tracer else [],
        }
        _print(json.dumps(payload, indent=2, ensure_ascii=False))
        return
    _print("\n=== FINAL ===")
    _print(result.final or f"(no output — {result.error})")
    _print("\n=== SWARM ===")
    _print(result.summary())
    if result.tracer:
        _print("\n=== TRACE ===")
        _print(result.tracer.render())


def cmd_run(args) -> int:
    from maestro.orchestrator import Orchestrator

    try:
        orch = Orchestrator.from_file(args.spec)
    except Exception as exc:
        _print(f"error: {exc}")
        return 2
    result = orch.run(args.task, trace=not args.no_trace)
    _emit_result(result, args.json)
    return 0 if result.ok() else 1


def cmd_validate(args) -> int:
    from maestro.orchestrator.spec import SwarmSpec

    try:
        spec = SwarmSpec.from_file(args.spec)
    except Exception as exc:
        _print(f"error: {exc}")
        return 2
    problems = spec.validate()
    if problems:
        _print(f"INVALID — {len(problems)} problem(s):")
        for p in problems:
            _print(f"  - {p}")
        return 1
    _print(f"OK — '{spec.name}' ({spec.topology}, {len(spec.agents)} agents) is valid.")
    return 0


def cmd_describe(args) -> int:
    from maestro.orchestrator import Orchestrator

    try:
        orch = Orchestrator.from_file(args.spec)
    except Exception as exc:
        _print(f"error: {exc}")
        return 2
    _print(orch.describe())
    return 0


def cmd_demo(args) -> int:
    from maestro.demo import run_demo

    result = run_demo(args.task) if args.task else run_demo()
    _emit_result(result, args.json)
    return 0 if result.ok() else 1


def cmd_providers(_args) -> int:
    from maestro.config.settings import settings
    from maestro.providers import PROVIDERS

    _print("Providers:")
    for p in PROVIDERS:
        _print(f"  - {p}")
    _print("\nEnvironment (redacted):")
    for k, v in settings.redacted().items():
        _print(f"  {k}: {v}")
    return 0


def cmd_topologies(_args) -> int:
    from maestro.topologies import topology_names

    _print("Topologies:")
    for t in topology_names():
        _print(f"  - {t}")
    return 0


def cmd_agents(_args) -> int:
    from maestro.agents.registry import AGENT_TYPES
    from maestro.tools import available_tools

    _print("Agent types:")
    for a in AGENT_TYPES:
        _print(f"  - {a}")
    _print("\nBuilt-in tools:")
    for t in available_tools():
        _print(f"  - {t}")
    return 0


def cmd_mcp_serve(_args) -> int:
    from maestro.mcp.server import serve

    serve()
    return 0


def cmd_version(_args) -> int:
    _print(f"maestro {__version__}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="maestro",
        description="M.A.E.S.T.R.O. — orchestrate & instantiate provider-agnostic agent swarms.",
    )
    sub = p.add_subparsers(dest="command")

    pr = sub.add_parser("run", help="instantiate a swarm from a spec and run a task")
    pr.add_argument("spec", help="path to a swarm spec (.yaml/.yml/.json)")
    pr.add_argument("task", help="the task for the swarm")
    pr.add_argument("--no-trace", action="store_true", help="disable the run trace")
    pr.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    pr.set_defaults(func=cmd_run)

    pv = sub.add_parser("validate", help="validate a spec without running it")
    pv.add_argument("spec")
    pv.set_defaults(func=cmd_validate)

    pd = sub.add_parser("describe", help="show a swarm and its agent availability")
    pd.add_argument("spec")
    pd.set_defaults(func=cmd_describe)

    pm = sub.add_parser("demo", help="run the built-in offline demo swarm")
    pm.add_argument("task", nargs="?", default="", help="optional custom task")
    pm.add_argument("--json", action="store_true")
    pm.set_defaults(func=cmd_demo)

    sub.add_parser("providers", help="list providers and configuration").set_defaults(
        func=cmd_providers
    )
    sub.add_parser("topologies", help="list topologies").set_defaults(func=cmd_topologies)
    sub.add_parser("agents", help="list agent types and tools").set_defaults(func=cmd_agents)
    sub.add_parser("mcp-serve", help="expose MAESTRO over MCP (stdio)").set_defaults(
        func=cmd_mcp_serve
    )
    sub.add_parser("version", help="print version").set_defaults(func=cmd_version)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
