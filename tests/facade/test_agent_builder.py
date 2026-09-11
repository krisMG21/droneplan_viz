"""Tests de droneplan_viz.facade.agent_builder.

Verifican drone() y transporter(): creación correcta, los tres casos de
`arms` (default / monobrazo / explorador), capacidad obligatoria y validada,
y las validaciones de construcción (id duplicado, posición no declarada,
brazos vacíos o repetidos).
"""
from __future__ import annotations

import pytest

from droneplan_viz.domain import Arm, Drone, DroneState, Transporter
from droneplan_viz.facade.agent_builder import AgentBuilder
from droneplan_viz.facade.errors import (
    DuplicateIdError,
    FacadeError,
    UnknownLocationError,
)
from droneplan_viz.facade.state import _FacadeState
from droneplan_viz.facade.world_builder import WorldBuilder


@pytest.fixture
def state() -> _FacadeState:
    s = _FacadeState()
    # Infraestructura mínima: una localización para anclar agentes.
    WorldBuilder(s).location("deposito")
    return s


@pytest.fixture
def ab(state: _FacadeState) -> AgentBuilder:
    return AgentBuilder(state)


# ===========================================================================
# drone: casos de arms
# ===========================================================================


class TestDroneArms:
    def test_default_dos_brazos_izq_der(self, ab):
        d = ab.drone("dron1", at="deposito")
        assert isinstance(d, Drone)
        assert d.arms == (Arm(id="izq"), Arm(id="der"))

    def test_monobrazo(self, ab):
        d = ab.drone("dron2", at="deposito", arms=["unico"])
        assert d.arms == (Arm(id="unico"),)

    def test_explorador_sin_brazos(self, ab):
        d = ab.drone("explorador", at="deposito", arms=[])
        assert d.arms == ()

    def test_arranca_en_idle(self, ab):
        d = ab.drone("dron1", at="deposito")
        assert d.state is DroneState.IDLE

    def test_default_no_se_comparte_entre_llamadas(self, ab):
        """El antipatrón del mutable default: dos drones default no deben
        compartir ni contaminar la lista de brazos."""
        d1 = ab.drone("dron1", at="deposito")
        d2 = ab.drone("dron2", at="deposito")
        assert d1.arms == (Arm(id="izq"), Arm(id="der"))
        assert d2.arms == (Arm(id="izq"), Arm(id="der"))

    def test_brazo_vacio_lanza(self, ab):
        with pytest.raises(FacadeError):
            ab.drone("dron1", at="deposito", arms=["izq", ""])

    def test_brazos_repetidos_lanzan(self, ab):
        with pytest.raises(FacadeError) as exc:
            ab.drone("dron1", at="deposito", arms=["izq", "izq"])
        assert "izq" in str(exc.value)


# ===========================================================================
# drone: guardado, referencia, validaciones de id/posición
# ===========================================================================


class TestDroneConstruccion:
    def test_se_guarda_en_state(self, ab, state):
        d = ab.drone("dron1", at="deposito")
        assert state.drones["dron1"] is d

    def test_acepta_referencia_de_location(self, state):
        casa = WorldBuilder(state).location("casa1")
        d = AgentBuilder(state).drone("dron1", at=casa)
        assert d.position == "casa1"

    def test_posicion_no_declarada_lanza(self, ab):
        with pytest.raises(UnknownLocationError) as exc:
            ab.drone("dron1", at="casa9")
        assert "al declarar el dron 'dron1'" in str(exc.value)

    def test_duplicado_lanza(self, ab):
        ab.drone("dron1", at="deposito")
        with pytest.raises(DuplicateIdError):
            ab.drone("dron1", at="deposito")

    def test_id_vacio_lanza(self, ab):
        with pytest.raises(FacadeError):
            ab.drone("", at="deposito")


# ===========================================================================
# transporter
# ===========================================================================


class TestTransporter:
    def test_crea_y_guarda(self, ab, state):
        t = ab.transporter("t1", capacidad=4, at="deposito")
        assert isinstance(t, Transporter)
        assert t.capacity == 4
        assert t.position == "deposito"
        assert state.transporters["t1"] is t

    def test_capacidad_obligatoria(self, ab):
        """capacidad es keyword sin default: omitirla es TypeError de Python."""
        with pytest.raises(TypeError):
            ab.transporter("t1", at="deposito")  # type: ignore[call-arg]

    def test_capacidad_no_entera_lanza(self, ab):
        with pytest.raises(FacadeError):
            ab.transporter("t1", capacidad=2.5, at="deposito")  # type: ignore[arg-type]

    def test_capacidad_bool_lanza(self, ab):
        """bool es subclase de int; no debe colarse como capacidad 1."""
        with pytest.raises(FacadeError):
            ab.transporter("t1", capacidad=True, at="deposito")  # type: ignore[arg-type]

    def test_capacidad_cero_lanza(self, ab):
        with pytest.raises(FacadeError):
            ab.transporter("t1", capacidad=0, at="deposito")

    def test_capacidad_negativa_lanza(self, ab):
        with pytest.raises(FacadeError):
            ab.transporter("t1", capacidad=-3, at="deposito")

    def test_posicion_no_declarada_lanza(self, ab):
        with pytest.raises(UnknownLocationError):
            ab.transporter("t1", capacidad=4, at="casa9")

    def test_duplicado_lanza(self, ab):
        ab.transporter("t1", capacidad=4, at="deposito")
        with pytest.raises(DuplicateIdError):
            ab.transporter("t1", capacidad=2, at="deposito")


# ===========================================================================
# Estado compartido world+agents
# ===========================================================================


class TestEstadoCompartido:
    def test_world_y_agent_builder_comparten_state(self):
        state = _FacadeState()
        WorldBuilder(state).location("base")
        d = AgentBuilder(state).drone("dron1", at="base")
        assert d.position == "base"
        assert "dron1" in state.drones
