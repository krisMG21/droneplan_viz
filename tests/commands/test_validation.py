"""Tests de commands/validation.py: dispatch validate(world, cmd).

Estos tests verifican EL DISPATCH, no las reglas del Validador del dominio
(que tienen sus propios tests en Sesión A). Por cada tipo de Command:
- un caso feliz que confirma que la validación pasa cuando todo es legal,
- al menos un caso de fallo que confirma que la razón llega del Validador
  subyacente con identificadores concretos.

También se verifica el branch defensivo del `case _:` que rechaza objetos
no-Command.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from droneplan_viz.commands.manipulation import Deliver, PickUp
from droneplan_viz.commands.movement import Move, MoveWithTransporter
from droneplan_viz.commands.transport import LoadIntoTransporter, UnloadFromTransporter
from droneplan_viz.commands.validation import validate


# ---------------------------------------------------------------------------
# Move
# ---------------------------------------------------------------------------
class TestValidateMove:
    def test_caso_feliz(self, base_world):
        cmd = Move(drone_id="d1", destination_id="casa1")
        result = validate(base_world, cmd)
        assert result.ok is True
        assert result.reason is None

    def test_falla_si_drone_no_existe(self, base_world):
        cmd = Move(drone_id="d_inexistente", destination_id="casa1")
        result = validate(base_world, cmd)
        assert result.ok is False
        assert "d_inexistente" in result.reason

    def test_falla_si_destino_no_existe(self, base_world):
        cmd = Move(drone_id="d1", destination_id="loc_fantasma")
        result = validate(base_world, cmd)
        assert result.ok is False
        assert "loc_fantasma" in result.reason


# ---------------------------------------------------------------------------
# MoveWithTransporter
# ---------------------------------------------------------------------------
class TestValidateMoveWithTransporter:
    def test_caso_feliz(self, base_world):
        # base_world tiene un brazo (der) libre y otro (izq) sostiene paquete
        # -> al menos un brazo libre, ejecutable.
        cmd = MoveWithTransporter(
            drone_id="d1", transporter_id="t1", destination_id="casa1"
        )
        result = validate(base_world, cmd)
        assert result.ok is True

    def test_falla_si_transportador_no_existe(self, base_world):
        cmd = MoveWithTransporter(
            drone_id="d1", transporter_id="t_inexistente", destination_id="casa1"
        )
        result = validate(base_world, cmd)
        assert result.ok is False
        assert "t_inexistente" in result.reason


# ---------------------------------------------------------------------------
# PickUp
# ---------------------------------------------------------------------------
class TestValidatePickUp:
    def test_caso_feliz(self, base_world):
        # brazo 'der' está libre, paquete 'libre1' está en deposito = drone
        cmd = PickUp(drone_id="d1", arm_id="der", package_id="libre1")
        result = validate(base_world, cmd)
        assert result.ok is True

    def test_falla_si_brazo_no_libre(self, base_world):
        # brazo 'izq' sostiene 'sostenida' -> no puede recoger 'libre1'
        cmd = PickUp(drone_id="d1", arm_id="izq", package_id="libre1")
        result = validate(base_world, cmd)
        assert result.ok is False
        assert "izq" in result.reason

    def test_falla_si_paquete_no_existe(self, base_world):
        cmd = PickUp(drone_id="d1", arm_id="der", package_id="caja_fantasma")
        result = validate(base_world, cmd)
        assert result.ok is False
        assert "caja_fantasma" in result.reason


# ---------------------------------------------------------------------------
# Deliver
# ---------------------------------------------------------------------------
class TestValidateDeliver:
    def test_caso_feliz(self, base_world):
        # Necesito un setup donde el drone esté en casa1 con 'sostenida'
        # que contenga MEDICINA (lo que ana necesita).
        # En base_world, 'sostenida' contiene COMIDA, no medicina.
        # Construyo un World ad-hoc.
        from droneplan_viz.domain import (
            Arm, Content, Drone, DroneState, HeldByArm, Location, Package,
            Person, World,
        )
        med = Content(id="medicina")
        w = World(
            locations={"casa1": Location(id="casa1")},
            drones={
                "d1": Drone(
                    id="d1", position="casa1",
                    arms=(Arm(id="izq"),), state=DroneState.IDLE,
                )
            },
            packages={
                "p_med": Package(
                    id="p_med", contains=med,
                    at=HeldByArm(drone_id="d1", arm_id="izq"),
                )
            },
            persons={
                "ana": Person(
                    id="ana", position="casa1", needs=(med,), has_received=(),
                )
            },
            contents={"medicina": med},
        )
        # Deliver no lleva arm_id; el Validator deriva qué brazo del drone
        # sostiene el paquete.
        cmd = Deliver(drone_id="d1", package_id="p_med", person_id="ana")
        result = validate(w, cmd)
        assert result.ok is True

    def test_falla_si_persona_no_co_localizada(self, base_world):
        # En base_world, ana está en casa1 y d1 en deposito -> falla.
        # 'sostenida' está siendo agarrada por izq, así que pasa el primer
        # chequeo (paquete sostenido por d1) y falla en co-localización.
        cmd = Deliver(drone_id="d1", package_id="sostenida", person_id="ana")
        result = validate(base_world, cmd)
        assert result.ok is False


# ---------------------------------------------------------------------------
# LoadIntoTransporter
# ---------------------------------------------------------------------------
class TestValidateLoad:
    def test_caso_feliz(self, base_world):
        # paquete 'sostenida' sostenido por d1 (en algún brazo: izq),
        # transportador t1 en mismo sitio, capacidad 4, ya hay 1 dentro
        # -> hay hueco. Load no lleva arm_id.
        cmd = LoadIntoTransporter(
            drone_id="d1", package_id="sostenida", transporter_id="t1"
        )
        result = validate(base_world, cmd)
        assert result.ok is True

    def test_falla_si_paquete_no_sostenido_por_el_drone(self, base_world):
        # 'libre1' está libre en deposito, no lo sostiene nadie
        cmd = LoadIntoTransporter(
            drone_id="d1", package_id="libre1", transporter_id="t1"
        )
        result = validate(base_world, cmd)
        assert result.ok is False


# ---------------------------------------------------------------------------
# UnloadFromTransporter
# ---------------------------------------------------------------------------
class TestValidateUnload:
    def test_caso_feliz(self, base_world):
        # brazo der libre, paquete 'en_trans' en t1, t1 co-localizado.
        cmd = UnloadFromTransporter(
            drone_id="d1", arm_id="der", package_id="en_trans",
            transporter_id="t1",
        )
        result = validate(base_world, cmd)
        assert result.ok is True

    def test_falla_si_paquete_no_esta_en_ese_transportador(self, base_world):
        # 'libre1' está en suelo, no en t1
        cmd = UnloadFromTransporter(
            drone_id="d1", arm_id="der", package_id="libre1",
            transporter_id="t1",
        )
        result = validate(base_world, cmd)
        assert result.ok is False


# ---------------------------------------------------------------------------
# Drone en ERROR: regla transversal aplicada por todos
# ---------------------------------------------------------------------------
class TestValidateRechazaDroneEnError:
    """La regla transversal 'no en ERROR' del Validador aplica a las seis
    acciones. Confirmamos el dispatch propagando este caso por las dos
    primeras (Move y PickUp); el resto está cubierto por los tests del
    Validador de Sesión A.
    """

    def test_move_rechaza_drone_en_error(self, base_world):
        from tests.commands.conftest import world_with_drone_in_error
        broken = world_with_drone_in_error(base_world)
        cmd = Move(drone_id="d1", destination_id="casa1")
        result = validate(broken, cmd)
        assert result.ok is False
        assert "ERROR" in result.reason

    def test_pick_up_rechaza_drone_en_error(self, base_world):
        from tests.commands.conftest import world_with_drone_in_error
        broken = world_with_drone_in_error(base_world)
        cmd = PickUp(drone_id="d1", arm_id="der", package_id="libre1")
        result = validate(broken, cmd)
        assert result.ok is False
        assert "ERROR" in result.reason


# ---------------------------------------------------------------------------
# Branch defensivo: objeto no-Command
# ---------------------------------------------------------------------------
class TestValidateRechazaNoCommand:
    """El case _: del match lanza TypeError. Aunque el sistema de tipos
    debería impedir que se llame validate() con algo que no es Command,
    verificamos que el branch defensivo funciona.
    """

    def test_lanza_typeerror_con_objeto_arbitrario(self, base_world):
        with pytest.raises(TypeError, match="no es un Command"):
            validate(base_world, "esto_no_es_un_command")  # type: ignore[arg-type]

    def test_lanza_typeerror_con_dataclass_no_command(self, base_world):
        @dataclass
        class FakeCommand:
            drone_id: str
            duration: float
            command_id: str
        fake = FakeCommand(drone_id="d1", duration=0.0, command_id="x")
        with pytest.raises(TypeError, match="no es un Command"):
            validate(base_world, fake)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Dispatch a Load y Unload con la misma intención semántica
# ---------------------------------------------------------------------------
class TestValidateDispatchLoadYUnload:
    """Load y Unload tienen FORMAS distintas (Load sin arm_id, Unload con),
    pero queremos confirmar que el dispatch los enruta correctamente: si
    construimos un Load y un Unload sobre el mismo paquete y transportador,
    ambos tienen sentido distinto y producen resultados de validación
    coherentes con sus reglas.
    """

    def test_load_para_paquete_sostenido_pasa(self, base_world):
        # 'sostenida' lo sostiene d1 (en izq) -> Load OK
        load = LoadIntoTransporter(
            drone_id="d1", package_id="sostenida", transporter_id="t1"
        )
        result = validate(base_world, load)
        assert result.ok is True

    def test_unload_para_paquete_en_transportador_pasa(self, base_world):
        # 'en_trans' está en t1 -> Unload OK con brazo libre der
        unload = UnloadFromTransporter(
            drone_id="d1", arm_id="der", package_id="en_trans",
            transporter_id="t1",
        )
        result = validate(base_world, unload)
        assert result.ok is True
