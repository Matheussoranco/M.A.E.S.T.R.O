"""MAESTRO command-line interface (standard-library ``argparse`` only).

maestro demo                       run the offline demo swarm
maestro run <spec> "<task>"        instantiate a swarm from a spec and run it
maestro validate <spec>            check a spec without running it
maestro describe <spec>            show the swarm & agent availability
maestro providers                  list providers and which are configured
maestro topologies                 list available topologies
maestro agents                     list agent types and built-in tools
maestro tools [name] [arg]         list built-in tools, or run one
maestro prices                     show the token price table used for costs
maestro config [key]               show effective settings (redacted)
maestro examples                   list the bundled example swarm specs
maestro init [out.yaml]            scaffold a new swarm spec
maestro mcp-serve                  expose MAESTRO itself over MCP (stdio)
maestro version

``run`` and ``demo`` also take ``--stream`` (print tokens as they arrive) and
``--usage`` (print the token/cost report for the run).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from maestro import __version__

logger = logging.getLogger("maestro.cli")

STUB_BANNER = (
    "[MAESTRO] STUB fallback ENABLED — unavailable providers will use the "
    "deterministic 'echo' backend. Pass --allow-stub explicitly to acknowledge "
    "(or set MAESTRO_ALLOW_STUB=1/0)."
)


def _cli_settings(args):
    """Effective settings for CLI runs.

    The library default is ``allow_stub_fallback=False`` outside demo/test
    (``MAESTRO_ALLOW_STUB`` opts in, pytest/demo opt out via
    ``_default_stub_fallback``); the CLI is equally strict: silent stub use
    requires explicit opt-in via ``--allow-stub`` or ``MAESTRO_ALLOW_STUB``
    in the environment.  Otherwise fallback is disabled for this invocation.
    When fallback stays enabled a STUB banner (stderr, so ``--json`` stdout stays clean) plus
    ``logger.warning`` are emitted.
    """
    from maestro.config.settings import Settings

    s = Settings()
    explicit_flag = bool(getattr(args, "allow_stub", False))
    explicit_env = "MAESTRO_ALLOW_STUB" in os.environ
    if explicit_flag:
        s.allow_stub_fallback = True
    elif not explicit_env:
        s.allow_stub_fallback = False
    if s.allow_stub_fallback:
        logger.warning("STUB fallback enabled — providers may degrade to 'echo' backend")
        print(STUB_BANNER, file=sys.stderr)
    return s


def _print(*args) -> None:
    # Force UTF-8-safe printing on Windows terminals.
    text = " ".join(str(a) for a in args)
    try:
        print(text)
    except UnicodeEncodeError:  # pragma: no cover - console-dependent
        sys.stdout.buffer.write(text.encode("utf-8", "replace") + b"\n")


def _emit_result(result, as_json: bool, usage: bool = False) -> None:
    if as_json:
        import json

        payload = {
            "task": result.task,
            "topology": result.topology,
            "final": result.final,
            "error": result.error,
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
            # Always included: consumers of the JSON should not have to ask for
            # the bill, and its "unknown" fields are meaningful either way.
            "usage": result.usage.as_dict(),
            "trace": result.tracer.to_list() if result.tracer else [],
        }
        _print(json.dumps(payload, indent=2, ensure_ascii=False))
        return
    _print("\n=== FINAL ===")
    _print(result.final or f"(no output — {result.error})")
    _print("\n=== SWARM ===")
    _print(result.summary())
    if usage:
        _print("\n=== USAGE ===")
        _print(result.usage.render())
    if result.tracer:
        _print("\n=== TRACE ===")
        _print(result.tracer.render())


def _token_printer(output=None):
    """A sink that prints streamed fragments, tagging each change of speaker."""
    output = output or sys.stdout
    state = {"who": ""}

    def on_token(agent: str, text: str) -> None:
        if agent != state["who"]:
            state["who"] = agent
            output.write(f"\n[{agent}] ")
        output.write(text)
        output.flush()

    return on_token


def _run_swarm(orch, task: str, args):
    """Run a swarm, honouring --stream, and return the result."""
    stream = getattr(args, "stream", False)
    if not stream:
        return orch.run(task, trace=not getattr(args, "no_trace", False))
    # stdout must remain valid JSON when --json and --stream are combined.
    output = sys.stderr if getattr(args, "json", False) else sys.stdout
    print("=== STREAM ===", file=output)
    result = orch.run(
        task,
        trace=not getattr(args, "no_trace", False),
        on_token=_token_printer(output),
    )
    print("", file=output)
    return result


def cmd_run(args) -> int:
    from maestro.orchestrator import Orchestrator

    try:
        orch = Orchestrator.from_file(args.spec, settings=_cli_settings(args))
    except Exception as exc:
        _print(f"error: {exc}")
        return 2
    result = _run_swarm(orch, args.task, args)
    _emit_result(result, args.json, args.usage)
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
    from maestro.demo import demo_spec
    from maestro.orchestrator import Orchestrator

    orch = Orchestrator.from_dict(demo_spec(), settings=_cli_settings(args))
    task = args.task or "Design a plan to evaluate a new AI agent on ARC-AGI-2."
    result = _run_swarm(orch, task, args)
    _emit_result(result, args.json, args.usage)
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


def cmd_prices(_args) -> int:
    from maestro.telemetry.usage import known_prices

    prices = known_prices()
    _print("Token prices (US$ per million tokens, matched as a substring of the model id):")
    for model in sorted(prices):
        inp, out = prices[model]
        _print(f"  {model:<24} in ${inp:>7.2f}   out ${out:>7.2f}")
    _print(
        "\nModels not listed here are *unpriced*, not free: their cost is reported as\n"
        "unknown. Add one with a spec 'prices:' block, or with\n"
        "maestro.telemetry.usage.register_price(model, input, output)."
    )
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
    from maestro.config import settings as cfg
    from maestro.mcp.server import serve

    effective = _cli_settings(_args)
    # The MCP server builds swarms internally from the module-global settings;
    # propagate the explicit opt-in so `mcp-serve` honors the same policy.
    cfg.settings.allow_stub_fallback = effective.allow_stub_fallback
    serve()
    return 0


def cmd_version(_args) -> int:
    _print(f"maestro {__version__}")
    return 0


def cmd_config(args) -> int:
    from maestro.config.settings import settings

    data = settings.redacted()

    if args.key:
        if args.key not in data:
            _print(f"error: no such setting {args.key!r}")
            _print(f"known keys: {', '.join(sorted(data))}")
            return 2
        _print(str(data[args.key]))
        return 0

    if args.json:
        import json

        _print(json.dumps(data, indent=2, ensure_ascii=False, default=str))
        return 0

    _print("Settings (redacted; every value is overridable by env var):")
    for key in sorted(data):
        _print(f"  {key:24s} {data[key]}")
    return 0


def cmd_tools(args) -> int:
    from maestro.tools.builtin import builtin_tools

    registry = builtin_tools()

    if not args.name:
        _print("Built-in tools:")
        for name in sorted(registry):
            _print(f"  {name:12s} {registry[name].description}")
        return 0

    tool = registry.get(args.name)
    if tool is None:
        _print(f"error: unknown tool {args.name!r}; available: {', '.join(sorted(registry))}")
        return 2

    if args.arg is None:
        _print(f"{tool.name}: {tool.description}")
        return 0

    try:
        _print(tool.run(args.arg))
    except Exception as exc:
        _print(f"error: {exc}")
        return 1
    return 0


def cmd_examples(_args) -> int:
    from pathlib import Path

    # Examples ship in the repo, not the wheel; look next to the package root.
    root = Path(__file__).resolve().parents[2]
    examples = root / "examples"
    if not examples.is_dir():
        _print(f"No examples directory found at {examples}.")
        return 1

    specs = sorted(p for p in examples.iterdir() if p.suffix in {".yaml", ".yml", ".json"})
    if not specs:
        _print("No example specs found.")
        return 1

    _print(f"Example swarm specs in {examples}:")
    for path in specs:
        summary = ""
        try:
            from maestro.orchestrator.spec import SwarmSpec

            spec = SwarmSpec.from_file(str(path))
            summary = f"{spec.topology}, {len(spec.agents)} agents — {spec.description}"
        except Exception as exc:  # a broken example should not hide the others
            summary = f"(could not parse: {exc})"
        _print(f"  {path.name:28s} {summary}")
    _print(f'\nRun one with:  maestro run {specs[0]} "your task"')
    return 0


_INIT_TEMPLATE = """\
# {name} — generated by `maestro init`.
#
# Run:  maestro run {filename} "your task here"
# Check without running:  maestro validate {filename}

name: {name}
description: {description}
topology: {topology}
{topology_params}
providers:
  default:
    provider: {provider}
    model: {model}

agents:
{agents}
"""

_AGENT_TEMPLATE = """\
  - name: {name}
    type: llm
    provider: default
    role: {role}
    system_prompt: {prompt}
"""


def cmd_init(args) -> int:
    from pathlib import Path

    from maestro.topologies import topology_names

    known = topology_names()
    if args.topology not in known:
        _print(f"error: unknown topology {args.topology!r}; choose one of {', '.join(known)}")
        return 2

    roles = [r.strip() for r in args.agents.split(",") if r.strip()]
    if not roles:
        _print("error: --agents needs at least one name")
        return 2

    out = Path(args.output)
    if out.exists() and not args.force:
        _print(f"error: {out} already exists (use --force to overwrite)")
        return 2

    description = f"A {args.topology} swarm of {len(roles)} agents."

    if out.suffix.lower() == ".json":
        # A .json target must contain valid JSON — the YAML template below is
        # not parseable JSON, so build the equivalent structure directly.
        import json

        spec: dict = {
            "name": args.name,
            "description": description,
            "topology": args.topology,
            "providers": {"default": {"provider": args.provider, "model": args.model}},
            "agents": [
                {
                    "name": role,
                    "type": "llm",
                    "provider": "default",
                    "role": role.replace("_", " "),
                    "system_prompt": (
                        f"You are the {role.replace('_', ' ')}. Do your part of the task."
                    ),
                }
                for role in roles
            ],
        }
        # The supervisor topology needs to know which agent leads; default to the first.
        if args.topology == "supervisor":
            spec["topology_params"] = {"supervisor": roles[0]}
        text = json.dumps(spec, indent=2, ensure_ascii=False) + "\n"
    else:
        # The supervisor topology needs to know which agent leads; default to the first.
        topology_params = ""
        if args.topology == "supervisor":
            topology_params = f"topology_params:\n  supervisor: {roles[0]}\n"

        agents_block = "".join(
            _AGENT_TEMPLATE.format(
                name=role,
                role=role.replace("_", " "),
                prompt=f"You are the {role.replace('_', ' ')}. Do your part of the task.",
            )
            for role in roles
        )

        text = _INIT_TEMPLATE.format(
            name=args.name,
            filename=out.name,
            description=description,
            topology=args.topology,
            topology_params=topology_params,
            provider=args.provider,
            model=args.model,
            agents=agents_block,
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    _print(f"Wrote {out} — {args.topology} swarm with {len(roles)} agents.")

    # Validate what we just generated so a bad template fails here, not later.
    from maestro.orchestrator.spec import SwarmSpec

    try:
        problems = SwarmSpec.from_file(str(out)).validate()
    except Exception as exc:
        _print(f"warning: generated spec did not parse: {exc}")
        return 1
    if problems:
        _print(f"warning: generated spec has {len(problems)} problem(s):")
        for p in problems:
            _print(f"  - {p}")
        return 1

    _print(f'Next:  maestro run {out} "your task here"')
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
    pr.add_argument("--stream", action="store_true", help="print tokens as they arrive")
    pr.add_argument("--usage", action="store_true", help="print the token/cost report")
    pr.add_argument(
        "--allow-stub",
        action="store_true",
        help="explicitly allow fallback to the deterministic 'echo' stub backend",
    )
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
    pm.add_argument("--stream", action="store_true", help="print tokens as they arrive")
    pm.add_argument("--usage", action="store_true", help="print the token/cost report")
    pm.add_argument("--no-trace", action="store_true", help="disable the run trace")
    pm.add_argument(
        "--allow-stub",
        action="store_true",
        help="explicitly allow fallback to the deterministic 'echo' stub backend",
    )
    pm.set_defaults(func=cmd_demo)

    sub.add_parser("providers", help="list providers and configuration").set_defaults(
        func=cmd_providers
    )
    sub.add_parser("topologies", help="list topologies").set_defaults(func=cmd_topologies)
    sub.add_parser("prices", help="show the token price table used for cost totals").set_defaults(
        func=cmd_prices
    )
    sub.add_parser("agents", help="list agent types and tools").set_defaults(func=cmd_agents)
    ms = sub.add_parser("mcp-serve", help="expose MAESTRO over MCP (stdio)")
    ms.add_argument(
        "--allow-stub",
        action="store_true",
        help="explicitly allow fallback to the deterministic 'echo' stub backend",
    )
    ms.set_defaults(func=cmd_mcp_serve)
    sub.add_parser("version", help="print version").set_defaults(func=cmd_version)

    pc = sub.add_parser("config", help="show effective settings (redacted)")
    pc.add_argument("key", nargs="?", default="", help="print just this setting")
    pc.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    pc.set_defaults(func=cmd_config)

    pt = sub.add_parser("tools", help="list built-in tools, or run one")
    pt.add_argument("name", nargs="?", default="", help="tool to describe or run")
    pt.add_argument("arg", nargs="?", default=None, help="argument to run the tool with")
    pt.set_defaults(func=cmd_tools)

    sub.add_parser("examples", help="list the bundled example swarm specs").set_defaults(
        func=cmd_examples
    )

    pi = sub.add_parser("init", help="scaffold a new swarm spec")
    pi.add_argument("output", nargs="?", default="swarm.yaml", help="path to write")
    pi.add_argument("--name", default="my-swarm", help="swarm name")
    pi.add_argument("--topology", default="sequential", help="topology to use")
    pi.add_argument(
        "--agents",
        default="researcher,analyst,critic",
        help="comma-separated agent names",
    )
    pi.add_argument("--provider", default="ollama", help="provider for the default block")
    pi.add_argument("--model", default="llama3.1", help="model for the default block")
    pi.add_argument("--force", action="store_true", help="overwrite an existing file")
    pi.set_defaults(func=cmd_init)

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
