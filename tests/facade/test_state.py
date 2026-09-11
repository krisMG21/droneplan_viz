"""Tests de droneplan_viz.facade.state.

Estado interno: dataclasses pasivas. Tests mínimos de contrato (defaults
independientes por instancia, campos mutables, _QueuedAction guarda lo que
recibe). La lógica que las llena se prueba en world_builder/agent_builder/
facade.
"""
from __future__ import annotations

from droneplan_viz.commands import Move
from droneplan_viz.facade.state import _FacadeState, _QueuedAction


class TestFacadeState:
    def test_arranca_vacio(self):
        s = _FacadeState()
        assert s.locations == {}
        assert s.contents == {}
        assert s.persons == {}
        assert s.packages == {}
        assert s.drones == {}
        assert s.transporters == {}
        assert s.costs == {}
        assert s.queued == []

    def test_defaults_independientes_por_instancia(self):
        """Cierra el antipatrón del mutable default compartido."""
        a = _FacadeState()
        b = _FacadeState()
        a.locations["casa1"] = object()  # type: ignore[assignment]
        assert b.locations == {}
        assert a.queued is not b.queued

    def test_campos_mutables(self):
        s = _FacadeState()
        s.costs[("a", "b")] = 5.0
        assert s.costs == {("a", "b"): 5.0}


class TestQueuedAction:
    def test_guarda_campos(self):
        cmd = Move(drone_id="d1", destination_id="casa1")
        qa = _QueuedAction(
            command=cmd, start_time=3.0, duration=8.0, declared_id="mover_pddl"
        )
        assert qa.command is cmd
        assert qa.start_time == 3.0
        assert qa.duration == 8.0
        assert qa.declared_id == "mover_pddl"

    def test_start_time_none_modo_secuencial(self):
        cmd = Move(drone_id="d1", destination_id="casa1")
        qa = _QueuedAction(
            command=cmd, start_time=None, duration=0.0, declared_id=None
        )
        assert qa.start_time is None
        assert qa.declared_id is None
