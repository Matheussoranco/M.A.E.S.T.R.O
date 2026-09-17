"""Declarative swarm specification — parsing & validation.

A *spec* is plain data (a dict, or a YAML/JSON file) describing a swarm: its
topology, a table of named providers, and a list of agents.  Keeping it data
(not code) is what makes MAESTRO an *instantiator*: the same runtime spins up any
swarm you can describe.

Example (YAML)::

    name: research-swarm
    topology: supervisor
    topology_params: {supervisor: lead}
    providers:
      claude: {provider: anthropic, model: claude-sonnet-5}
      local:  {provider: ollama,    model: llama3.1}
    agents:
      - {name: lead,   type: llm, provider: claude, role: supervisor}
      - {name: coder,  type: llm, provider: local,  role: engineer}
      - {name: isaac,  type: isaac}
      - {name: olivia, type: olivia}
    prices:                       # optional; $ per million tokens
      llama3.1: {input: 0, output: 0}
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

from maestro.agents.registry import AGENT_TYPES
from maestro.providers import PROVIDERS, ProviderSpec
from maestro.topologies import TOPOLOGIES

# Keys that reference an agent by name inside topology_params.
_AGENT_REF_PARAMS = ("supervisor", "aggregator", "judge", "router")


@dataclass
class SwarmSpec:
    name: str = "swarm"
    description: str = ""
    topology: str = "sequential"
    topology_params: dict = field(default_factory=dict)
    providers: dict[str, ProviderSpec] = field(default_factory=dict)
    agents: list[dict] = field(default_factory=list)
    #: ``{model_substring: (input_per_mtok, output_per_mtok)}`` in US dollars.
    #: Lets a spec price models MAESTRO does not know — local models included,
    #: where the honest number is usually ``0``.
    prices: dict[str, tuple[float, float]] = field(default_factory=dict)
    #: Unknown provider fields seen at load time (strict mode).  ``from_dict``
    #: still builds the spec (unknown keys ignored for the live object) but
    #: records them here so :meth:`validate` can report typos like ``modell``.
    provider_unknown_keys: dict[str, list[str]] = field(
        default_factory=dict, repr=False, compare=False
    )

    # -- construction ---------------------------------------------------------
    @classmethod
    def from_dict(cls, data: dict) -> SwarmSpec:
        if not isinstance(data, dict):
            raise ValueError("swarm spec must be a mapping/object")
        raw_providers = data.get("providers") or {}
        if not isinstance(raw_providers, dict):
            raise ValueError("'providers' must be a mapping")
        raw_params = data.get("topology_params") or {}
        if not isinstance(raw_params, dict):
            raise ValueError("'topology_params' must be a mapping")
        raw_agents = data.get("agents") or []
        if not isinstance(raw_agents, list):
            raise ValueError("'agents' must be a list")
        providers: dict[str, ProviderSpec] = {}
        provider_unknown_keys: dict[str, list[str]] = {}
        for key, pv in raw_providers.items():
            if isinstance(pv, str):
                providers[key] = ProviderSpec(provider=pv)
            elif isinstance(pv, dict):
                fields = ProviderSpec.__dataclass_fields__
                unknown = sorted(k for k in pv if k not in fields)
                if unknown:
                    provider_unknown_keys[str(key)] = unknown
                providers[key] = ProviderSpec(**{k: v for k, v in pv.items() if k in fields})
            else:
                raise ValueError(f"provider {key!r} must be a string or mapping")
        _raw_topology = data.get("topology", "sequential")
        # isinstance FIRST, .lower() second — calling .lower() on a non-str
        # (e.g. null/int from a hand-edited YAML) must not raise AttributeError.
        _topology = (
            _raw_topology.lower()
            if isinstance(_raw_topology, str) and _raw_topology
            else "sequential"
            if not isinstance(_raw_topology, str)
            else _raw_topology.lower()
        )
        return cls(
            name=data.get("name", "swarm"),
            description=data.get("description", ""),
            topology=_topology,
            topology_params=dict(raw_params),
            providers=providers,
            agents=list(raw_agents),
            prices=_parse_prices(data.get("prices")),
            provider_unknown_keys=provider_unknown_keys,
        )

    @classmethod
    def from_file(cls, path: str) -> SwarmSpec:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        ext = os.path.splitext(path)[1].lower()
        if ext in (".yaml", ".yml"):
            data = _load_yaml(text, path)
        elif ext == ".json":
            data = json.loads(text)
        else:
            # Try JSON first, then YAML, so extension-less specs still work.
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                data = _load_yaml(text, path)
        return cls.from_dict(data)

    # -- validation -----------------------------------------------------------
    def validate(self) -> list[str]:
        """Return a list of human-readable problems (empty ⇒ valid)."""
        problems: list[str] = []
        if not self.name:
            problems.append("spec is missing a 'name'")
        if not isinstance(self.topology, str) or self.topology not in TOPOLOGIES:
            problems.append(
                f"unknown topology {self.topology!r} (choose: {', '.join(sorted(TOPOLOGIES))})"
            )
        if not self.agents:
            problems.append("spec defines no agents")

        seen: set[str] = set()
        for i, ag in enumerate(self.agents):
            if not isinstance(ag, dict):
                problems.append(f"agent #{i} is not a mapping")
                continue
            name = ag.get("name")
            if not isinstance(name, str) or not name.strip():
                problems.append(f"agent #{i} is missing 'name'")
            elif name in seen:
                problems.append(f"duplicate agent name {name!r}")
            else:
                seen.add(name)
            raw_type = ag.get("type") or "llm"
            atype = raw_type.lower() if isinstance(raw_type, str) else ""
            if atype not in AGENT_TYPES:
                problems.append(f"agent {name!r} has unknown type {raw_type!r}")
            if atype == "cli" and not ag.get("command"):
                problems.append(f"cli agent {name!r} needs a 'command'")
            if atype == "mcp" and not (ag.get("server_cmd") and ag.get("tool")):
                problems.append(f"mcp agent {name!r} needs 'server_cmd' and 'tool'")
            ref = ag.get("provider")
            if isinstance(ref, str) and ref and ref not in self.providers and ref not in PROVIDERS:
                problems.append(f"agent {name!r} references unknown provider {ref!r}")
            if atype == "llm":
                _positive_int(problems, ag, "max_tokens", f"agent {name!r}")
                _positive_int(problems, ag, "max_tool_iters", f"agent {name!r}")
                _finite_number(problems, ag, "temperature", f"agent {name!r}")
                if "tools" in ag and not isinstance(ag["tools"], list):
                    problems.append(f"agent {name!r} field 'tools' must be a list")
            if atype in {"cli", "mcp", "isaac", "olivia"}:
                _positive_number(problems, ag, "timeout", f"agent {name!r}")
                if atype in {"cli", "mcp"}:
                    command_key = "command" if atype == "cli" else "server_cmd"
                    if command_key in ag and not isinstance(ag[command_key], (str, list)):
                        problems.append(
                            f"agent {name!r} field '{command_key}' must be a string or list"
                        )

        if self.topology in {"parallel", "supervisor"}:
            _positive_int(problems, self.topology_params, "max_workers", "topology_params")
        if self.topology == "debate":
            _positive_int(problems, self.topology_params, "rounds", "topology_params")

        for key, provider in self.providers.items():
            if not isinstance(provider.provider, str) or not provider.provider:
                problems.append(f"provider {key!r} needs a non-empty 'provider'")
            if provider.timeout is not None and (
                not isinstance(provider.timeout, (int, float))
                or isinstance(provider.timeout, bool)
                or not math.isfinite(provider.timeout)
                or provider.timeout <= 0
            ):
                problems.append(f"provider {key!r} field 'timeout' must be a positive number")
            if not isinstance(provider.options, dict):
                problems.append(f"provider {key!r} field 'options' must be a mapping")

        # Strict mode: unknown provider keys are typos (e.g. 'modell'), not
        # silently ignored.  They were recorded at load time in
        # ``provider_unknown_keys`` so validation can surface them.
        for key, unknown in (self.provider_unknown_keys or {}).items():
            problems.append(
                f"provider {key!r} has unknown field(s): {', '.join(unknown)} "
                f"(allowed: {', '.join(sorted(ProviderSpec.__dataclass_fields__))})"
            )

        # Prices schema: {model: (input, output)} with finite, non-negative
        # dollars-per-MTok.  _parse_prices enforces this at load (raising),
        # but specs built programmatically must still be caught here.
        for model, pair in (self.prices or {}).items():
            if (
                not isinstance(pair, (list, tuple))
                or len(pair) != 2
                or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in pair)
            ):
                problems.append(f"price for {model!r} must be [input, output] numbers")
                continue
            if not all(math.isfinite(v) and v >= 0 for v in pair):
                problems.append(f"price for {model!r} must be finite and non-negative")

        for key in _AGENT_REF_PARAMS:
            ref = self.topology_params.get(key)
            if ref and not isinstance(ref, str):
                problems.append(f"topology_params.{key} must be an agent name string")
            elif ref and ref not in seen:
                problems.append(f"topology_params.{key} = {ref!r} does not match any agent name")
        return problems


def _parse_prices(raw: object) -> dict[str, tuple[float, float]]:
    """Parse a ``prices:`` block into ``{model: (input, output)}``.

    Accepts either mapping form (``{input: 3, output: 15}``) or a two-item
    sequence (``[3, 15]``).  Both are dollars per **million** tokens.
    """
    if not raw:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("'prices' must be a mapping of model -> {input, output}")
    out: dict[str, tuple[float, float]] = {}
    for model, entry in raw.items():
        if isinstance(entry, dict):
            values = (entry.get("input", 0), entry.get("output", 0))
        elif isinstance(entry, (list, tuple)) and len(entry) == 2:
            values = (entry[0], entry[1])
        else:
            raise ValueError(
                f"price for {model!r} must be {{input: x, output: y}} or [x, y], got {entry!r}"
            )
        try:
            input_price, output_price = float(values[0]), float(values[1])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"price for {model!r} must be numeric") from exc
        if not all(math.isfinite(value) and value >= 0 for value in (input_price, output_price)):
            raise ValueError(f"price for {model!r} must be finite and non-negative")
        out[str(model)] = (input_price, output_price)
    return out


def _positive_int(problems: list[str], mapping: dict, key: str, where: str) -> None:
    if key not in mapping:
        return
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        problems.append(f"{where} field '{key}' must be a positive integer")


def _positive_number(problems: list[str], mapping: dict, key: str, where: str) -> None:
    if key not in mapping:
        return
    value = mapping[key]
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        problems.append(f"{where} field '{key}' must be a positive number")


def _finite_number(problems: list[str], mapping: dict, key: str, where: str) -> None:
    if key not in mapping or mapping[key] is None:
        return
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        problems.append(f"{where} field '{key}' must be a finite number")


def _load_yaml(text: str, path: str) -> dict:
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on env
        raise RuntimeError(
            f"reading {path} needs PyYAML — install it (`pip install maestro-swarm[yaml]`) "
            "or use a .json spec"
        ) from exc
    return yaml.safe_load(text) or {}
