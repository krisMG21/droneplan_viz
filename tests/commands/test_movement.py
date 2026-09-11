"""Tests de commands/movement.py: Move y MoveWithTransporter."""
from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from droneplan_viz.commands.base import Command
from droneplan_viz.commands.movement import Move, MoveWithTransporter


# ---------------------------------------------------------------------------
# Move: construcción
# ---------------------------------------------------------------------------
class TestMoveConstruccion:
    def test_construccion_minima(self):
        cmd = Move(drone_id="d1", destination_id="casa")
        assert cmd.drone_id == "d1"
        assert cmd.destination_id == "casa"

    def test_duration_default_es_cero(self):
        cmd = Move(drone_id="d1", destination_id="casa")
        assert cmd.duration == 0.0

    def test_duration_explicito_se_respeta(self):
        cmd = Move(drone_id="d1", destination_id="casa", duration=12.5)
        assert cmd.duration == 12.5

    def test_command_id_se_autogenera(self):
        cmd = Move(drone_id="d1", destination_id="casa")
        assert isinstance(cmd.command_id, str)
        assert len(cmd.command_id) == 32

    def test_command_id_explicito_se_respeta(self):
        cmd = Move(drone_id="d1", destination_id="casa", command_id="my_id")
        assert cmd.command_id == "my_id"

    def test_construccion_con_todos_los_campos(self):
        cmd = Move(
            drone_id="d1",
            destination_id="casa",
            duration=5.0,
            command_id="step_1",
        )
        assert cmd.drone_id == "d1"
        assert cmd.destination_id == "casa"
        assert cmd.duration == 5.0
        assert cmd.command_id == "step_1"


# ---------------------------------------------------------------------------
# Move: inmutabilidad
# ---------------------------------------------------------------------------
class TestMoveInmutabilidad:
    def test_no_se_pueden_reasignar_campos(self):
        cmd = Move(drone_id="d1", destination_id="casa")
        with pytest.raises(FrozenInstanceError):
            cmd.drone_id = "d2"  # type: ignore[misc]

    def test_no_se_pueden_anadir_atributos(self):
        cmd = Move(drone_id="d1", destination_id="casa")
        with pytest.raises((AttributeError, TypeError)):
            cmd.nuevo_campo = "x"  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Move: igualdad estructural (incluye command_id)
# ---------------------------------------------------------------------------
class TestMoveIgualdad:
    def test_dos_moves_con_mismo_id_explicito_son_iguales(self):
        a = Move(drone_id="d1", destination_id="casa", command_id="x")
        b = Move(drone_id="d1", destination_id="casa", command_id="x")
        assert a == b

    def test_dos_moves_con_mismos_campos_pero_ids_distintos_son_distintos(self):
        a = Move(drone_id="d1", destination_id="casa")
        b = Move(drone_id="d1", destination_id="casa")
        # command_id se auto-genera distinto en cada uno
        assert a != b

    def test_un_move_no_es_igual_a_otro_con_destino_distinto(self):
        a = Move(drone_id="d1", destination_id="casa", command_id="x")
        b = Move(drone_id="d1", destination_id="otra", command_id="x")
        assert a != b


# ---------------------------------------------------------------------------
# Move: conforma el Protocol Command
# ---------------------------------------------------------------------------
class TestMoveProtocolo:
    def test_move_cumple_protocol_command(self):
        cmd = Move(drone_id="d1", destination_id="casa")
        assert isinstance(cmd, Command)


# ---------------------------------------------------------------------------
# MoveWithTransporter: análogo
# ---------------------------------------------------------------------------
class TestMoveWithTransporterConstruccion:
    def test_construccion_minima(self):
        cmd = MoveWithTransporter(
            drone_id="d1", transporter_id="t1", destination_id="casa"
        )
        assert cmd.drone_id == "d1"
        assert cmd.transporter_id == "t1"
        assert cmd.destination_id == "casa"

    def test_duration_default_cero(self):
        cmd = MoveWithTransporter(
            drone_id="d1", transporter_id="t1", destination_id="casa"
        )
        assert cmd.duration == 0.0

    def test_no_lleva_arm_id(self):
        """Decisión explícita: MoveWithTransporter no especifica brazo.

        Coherente con la firma de validate_move_with_transporter del
        Validador (sesión A), que verifica 'al menos un brazo libre'
        sin especificar cuál.
        """
        cmd = MoveWithTransporter(
            drone_id="d1", transporter_id="t1", destination_id="casa"
        )
        # No debe haber un atributo arm_id
        assert not hasattr(cmd, "arm_id")

    def test_command_id_se_autogenera(self):
        cmd = MoveWithTransporter(
            drone_id="d1", transporter_id="t1", destination_id="casa"
        )
        assert len(cmd.command_id) == 32


class TestMoveWithTransporterInmutabilidad:
    def test_no_se_pueden_reasignar_campos(self):
        cmd = MoveWithTransporter(
            drone_id="d1", transporter_id="t1", destination_id="casa"
        )
        with pytest.raises(FrozenInstanceError):
            cmd.destination_id = "otra"  # type: ignore[misc]

    def test_no_se_pueden_anadir_atributos(self):
        cmd = MoveWithTransporter(
            drone_id="d1", transporter_id="t1", destination_id="casa"
        )
        with pytest.raises((AttributeError, TypeError)):
            cmd.extra = "x"  # type: ignore[attr-defined]


class TestMoveWithTransporterIgualdad:
    def test_iguales_con_mismo_id(self):
        a = MoveWithTransporter(
            drone_id="d1", transporter_id="t1", destination_id="casa",
            command_id="x",
        )
        b = MoveWithTransporter(
            drone_id="d1", transporter_id="t1", destination_id="casa",
            command_id="x",
        )
        assert a == b

    def test_distintos_con_ids_auto(self):
        a = MoveWithTransporter(
            drone_id="d1", transporter_id="t1", destination_id="casa"
        )
        b = MoveWithTransporter(
            drone_id="d1", transporter_id="t1", destination_id="casa"
        )
        assert a != b


class TestMoveWithTransporterProtocolo:
    def test_cumple_protocol_command(self):
        cmd = MoveWithTransporter(
            drone_id="d1", transporter_id="t1", destination_id="casa"
        )
        assert isinstance(cmd, Command)


# ---------------------------------------------------------------------------
# Move y MoveWithTransporter son tipos distintos
# ---------------------------------------------------------------------------
class TestMoveYMoveWithTransporterSonDistintos:
    def test_son_clases_distintas(self):
        assert Move is not MoveWithTransporter

    def test_un_move_no_es_un_movewithtransporter(self):
        m = Move(drone_id="d1", destination_id="casa")
        assert not isinstance(m, MoveWithTransporter)

    def test_un_movewithtransporter_no_es_un_move(self):
        m = MoveWithTransporter(
            drone_id="d1", transporter_id="t1", destination_id="casa"
        )
        assert not isinstance(m, Move)
