"""Tests de commands/manipulation.py: PickUp y Deliver."""
from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from droneplan_viz.commands.base import Command
from droneplan_viz.commands.manipulation import Deliver, PickUp


# ---------------------------------------------------------------------------
# PickUp (lleva arm_id obligatorio)
# ---------------------------------------------------------------------------
class TestPickUpConstruccion:
    def test_construccion_minima(self):
        cmd = PickUp(drone_id="d1", arm_id="izq", package_id="caja3")
        assert cmd.drone_id == "d1"
        assert cmd.arm_id == "izq"
        assert cmd.package_id == "caja3"

    def test_duration_default_cero(self):
        cmd = PickUp(drone_id="d1", arm_id="izq", package_id="caja3")
        assert cmd.duration == 0.0

    def test_duration_explicito(self):
        cmd = PickUp(drone_id="d1", arm_id="izq", package_id="caja3", duration=5.0)
        assert cmd.duration == 5.0

    def test_command_id_se_autogenera(self):
        cmd = PickUp(drone_id="d1", arm_id="izq", package_id="caja3")
        assert isinstance(cmd.command_id, str)
        assert len(cmd.command_id) == 32

    def test_command_id_explicito(self):
        cmd = PickUp(
            drone_id="d1", arm_id="izq", package_id="caja3", command_id="pick_1"
        )
        assert cmd.command_id == "pick_1"

    def test_no_lleva_person_id(self):
        """PickUp NO entrega a nadie; el destinatario es brazo del drone."""
        cmd = PickUp(drone_id="d1", arm_id="izq", package_id="caja3")
        assert not hasattr(cmd, "person_id")


class TestPickUpInmutabilidad:
    def test_no_reasignar(self):
        cmd = PickUp(drone_id="d1", arm_id="izq", package_id="caja3")
        with pytest.raises(FrozenInstanceError):
            cmd.arm_id = "der"  # type: ignore[misc]

    def test_no_anadir(self):
        cmd = PickUp(drone_id="d1", arm_id="izq", package_id="caja3")
        with pytest.raises((AttributeError, TypeError)):
            cmd.x = 1  # type: ignore[attr-defined]


class TestPickUpIgualdad:
    def test_iguales_con_mismo_id(self):
        a = PickUp(drone_id="d1", arm_id="izq", package_id="c", command_id="x")
        b = PickUp(drone_id="d1", arm_id="izq", package_id="c", command_id="x")
        assert a == b

    def test_distintos_con_ids_auto(self):
        a = PickUp(drone_id="d1", arm_id="izq", package_id="c")
        b = PickUp(drone_id="d1", arm_id="izq", package_id="c")
        assert a != b

    def test_distintos_si_arm_distinto(self):
        a = PickUp(drone_id="d1", arm_id="izq", package_id="c", command_id="x")
        b = PickUp(drone_id="d1", arm_id="der", package_id="c", command_id="x")
        assert a != b


class TestPickUpProtocolo:
    def test_cumple_protocol(self):
        cmd = PickUp(drone_id="d1", arm_id="izq", package_id="caja3")
        assert isinstance(cmd, Command)


# ---------------------------------------------------------------------------
# Deliver (NO lleva arm_id; decisión Sesión A)
# ---------------------------------------------------------------------------
class TestDeliverConstruccion:
    def test_construccion_minima(self):
        cmd = Deliver(drone_id="d1", package_id="c", person_id="ana")
        assert cmd.drone_id == "d1"
        assert cmd.package_id == "c"
        assert cmd.person_id == "ana"

    def test_no_lleva_arm_id(self):
        """Decisión Sesión A: el Validator no exige arm_id en entregar,
        porque el efecto libera 'el brazo que sea'. Esta es una de las
        decisiones estructurales heredadas; mantenerla aquí refleja
        simetría perfecta entre Command y validate_X.
        """
        cmd = Deliver(drone_id="d1", package_id="c", person_id="ana")
        assert not hasattr(cmd, "arm_id")

    def test_duration_default_cero(self):
        cmd = Deliver(drone_id="d1", package_id="c", person_id="ana")
        assert cmd.duration == 0.0

    def test_command_id_autogenerado(self):
        cmd = Deliver(drone_id="d1", package_id="c", person_id="ana")
        assert len(cmd.command_id) == 32

    def test_lleva_person_id(self):
        """Deliver SÍ lleva destinatario; es la diferencia clave con PickUp."""
        cmd = Deliver(drone_id="d1", package_id="c", person_id="ana")
        assert cmd.person_id == "ana"


class TestDeliverInmutabilidad:
    def test_no_reasignar(self):
        cmd = Deliver(drone_id="d1", package_id="c", person_id="ana")
        with pytest.raises(FrozenInstanceError):
            cmd.person_id = "luis"  # type: ignore[misc]

    def test_no_anadir(self):
        cmd = Deliver(drone_id="d1", package_id="c", person_id="ana")
        with pytest.raises((AttributeError, TypeError)):
            cmd.x = 1  # type: ignore[attr-defined]


class TestDeliverIgualdad:
    def test_iguales_con_mismo_id(self):
        a = Deliver(drone_id="d1", package_id="c", person_id="ana", command_id="x")
        b = Deliver(drone_id="d1", package_id="c", person_id="ana", command_id="x")
        assert a == b

    def test_distintos_si_persona_distinta(self):
        a = Deliver(drone_id="d1", package_id="c", person_id="ana", command_id="x")
        b = Deliver(drone_id="d1", package_id="c", person_id="luis", command_id="x")
        assert a != b


class TestDeliverProtocolo:
    def test_cumple_protocol(self):
        cmd = Deliver(drone_id="d1", package_id="c", person_id="ana")
        assert isinstance(cmd, Command)


# ---------------------------------------------------------------------------
# PickUp y Deliver son tipos distintos
# ---------------------------------------------------------------------------
class TestPickUpYDeliverSonDistintos:
    def test_son_clases_distintas(self):
        assert PickUp is not Deliver

    def test_un_pickup_no_es_deliver(self):
        p = PickUp(drone_id="d1", arm_id="izq", package_id="c")
        assert not isinstance(p, Deliver)

    def test_un_deliver_no_es_pickup(self):
        d = Deliver(drone_id="d1", package_id="c", person_id="ana")
        assert not isinstance(d, PickUp)
