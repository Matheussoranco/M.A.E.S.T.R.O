"""Topology registry — build a topology from its name + params."""

from __future__ import annotations

from maestro.topologies.base import SwarmResult, Topology
from maestro.topologies.debate import DebateTopology
from maestro.topologies.parallel import ParallelTopology
from maestro.topologies.router import RouterTopology
from maestro.topologies.sequential import SequentialTopology
from maestro.topologies.supervisor import SupervisorTopology

TOPOLOGIES: dict[str, type[Topology]] = {
    "sequential": SequentialTopology,
    "parallel": ParallelTopology,
    "supervisor": SupervisorTopology,
    "debate": DebateTopology,
    "router": RouterTopology,
}


def build_topology(name: str, **params) -> Topology:
    key = name.lower() if isinstance(name, str) and name else "sequential"
    if key not in TOPOLOGIES:
        raise ValueError(
            f"unknown topology {name!r}; choose one of {', '.join(sorted(TOPOLOGIES))}"
        )
    return TOPOLOGIES[key](**params)


def topology_names() -> list[str]:
    return sorted(TOPOLOGIES)


__all__ = [
    "TOPOLOGIES",
    "SwarmResult",
    "Topology",
    "build_topology",
    "topology_names",
]
