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

    # -- construction ---------------------------------------------------------
    @classmethod
    def from_dict(cls, data: dict) -> SwarmSpec:
        if not isinstance(data, dict):
            raise ValueError("swarm spec must be a mapping/object")
        providers: dict[str, ProviderSpec] = {}
        for key, pv in (data.get("providers") or {}).items():
            if isinstance(pv, str):
                providers[key] = ProviderSpec(provider=pv)
            elif isinstance(pv, dict):
                fields = ProviderSpec.__dataclass_fields__
                providers[key] = ProviderSpec(**{k: v for k, v in pv.items() if k in fields})
            else:
                raise ValueError(f"provider {key!r} must be a string or mapping")
        return cls(
            name=data.get("name", "swarm"),
            description=data.get("description", ""),
            topology=data.get("topology", "sequential"),
            topology_params=dict(data.get("topology_params") or {}),
            providers=providers,
            agents=list(data.get("agents") or []),
            prices=_parse_prices(data.get("prices")),
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
        if self.topology not in TOPOLOGIES:
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
            if not name:
                problems.append(f"agent #{i} is missing 'name'")
            elif name in seen:
                problems.append(f"duplicate agent name {name!r}")
            else:
                seen.add(name)
            atype = (ag.get("type") or "llm").lower()
            if atype not in AGENT_TYPES:
                problems.append(f"agent {name!r} has unknown type {atype!r}")
            if atype == "cli" and not ag.get("command"):
                problems.append(f"cli agent {name!r} needs a 'command'")
            if atype == "mcp" and not (ag.get("server_cmd") and ag.get("tool")):
                problems.append(f"mcp agent {name!r} needs 'server_cmd' and 'tool'")
            ref = ag.get("provider")
            if isinstance(ref, str) and ref and ref not in self.providers and ref not in PROVIDERS:
                problems.append(f"agent {name!r} references unknown provider {ref!r}")

        for key in _AGENT_REF_PARAMS:
            ref = self.topology_params.get(key)
            if ref and ref not in seen:
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
            out[str(model)] = (float(values[0]), float(values[1]))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"price for {model!r} must be numeric") from exc
    return out


def _load_yaml(text: str, path: str) -> dict:
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on env
        raise RuntimeError(
            f"reading {path} needs PyYAML — install it (`pip install maestro-swarm[yaml]`) "
            "or use a .json spec"
        ) from exc
    return yaml.safe_load(text) or {}
