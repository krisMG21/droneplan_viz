"""Tests de commands/transport.py: LoadIntoTransporter y UnloadFromTransporter."""
from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from droneplan_viz.commands.base import Command
from droneplan_viz.commands.transport import LoadIntoTransporter, UnloadFromTransporter


# ---------------------------------------------------------------------------
# LoadIntoTransporter (NO lleva arm_id; decisión Sesión A)
# ---------------------------------------------------------------------------
class TestLoadConstruccion:
    def test_construccion_minima(self):
        cmd = LoadIntoTransporter(
            drone_id="d1", package_id="c", transporter_id="t1"
        )
        assert cmd.drone_id == "d1"
        assert cmd.package_id == "c"
        assert cmd.transporter_id == "t1"

    def test_no_lleva_arm_id(self):
        """Decisión Sesión A: el Validator no exige arm_id en cargar,
        porque el efecto libera 'el brazo que sea'."""
        cmd = LoadIntoTransporter(
            drone_id="d1", package_id="c", transporter_id="t1"
        )
        assert not hasattr(cmd, "arm_id")

    def test_duration_default_cero(self):
        cmd = LoadIntoTransporter(
            drone_id="d1", package_id="c", transporter_id="t1"
        )
        assert cmd.duration == 0.0

    def test_command_id_autogenerado(self):
        cmd = LoadIntoTransporter(
            drone_id="d1", package_id="c", transporter_id="t1"
        )
        assert len(cmd.command_id) == 32


class TestLoadInmutabilidad:
    def test_no_reasignar(self):
        cmd = LoadIntoTransporter(
            drone_id="d1", package_id="c", transporter_id="t1"
        )
        with pytest.raises(FrozenInstanceError):
            cmd.transporter_id = "t2"  # type: ignore[misc]

    def test_no_anadir(self):
        cmd = LoadIntoTransporter(
            drone_id="d1", package_id="c", transporter_id="t1"
        )
        with pytest.raises((AttributeError, TypeError)):
            cmd.x = 1  # type: ignore[attr-defined]


class TestLoadIgualdad:
    def test_iguales_con_mismo_id(self):
        a = LoadIntoTransporter(
            drone_id="d1", package_id="c", transporter_id="t1", command_id="x"
        )
        b = LoadIntoTransporter(
            drone_id="d1", package_id="c", transporter_id="t1", command_id="x"
        )
        assert a == b

    def test_distintos_con_ids_auto(self):
        a = LoadIntoTransporter(
            drone_id="d1", package_id="c", transporter_id="t1"
        )
        b = LoadIntoTransporter(
            drone_id="d1", package_id="c", transporter_id="t1"
        )
        assert a != b


class TestLoadProtocolo:
    def test_cumple_protocol(self):
        cmd = LoadIntoTransporter(
            drone_id="d1", package_id="c", transporter_id="t1"
        )
        assert isinstance(cmd, Command)


# ---------------------------------------------------------------------------
# UnloadFromTransporter (lleva arm_id; decisión Sesión A)
# ---------------------------------------------------------------------------
class TestUnloadConstruccion:
    def test_construccion_minima(self):
        cmd = UnloadFromTransporter(
            drone_id="d1", arm_id="izq", package_id="c", transporter_id="t1"
        )
        assert cmd.drone_id == "d1"
        assert cmd.arm_id == "izq"
        assert cmd.package_id == "c"
        assert cmd.transporter_id == "t1"

    def test_lleva_arm_id(self):
        """Decisión Sesión A: el Validator exige arm_id en descargar,
        porque el efecto OCUPA un brazo concreto."""
        cmd = UnloadFromTransporter(
            drone_id="d1", arm_id="izq", package_id="c", transporter_id="t1"
        )
        assert cmd.arm_id == "izq"

    def test_duration_default_cero(self):
        cmd = UnloadFromTransporter(
            drone_id="d1", arm_id="izq", package_id="c", transporter_id="t1"
        )
        assert cmd.duration == 0.0

    def test_command_id_autogenerado(self):
        cmd = UnloadFromTransporter(
            drone_id="d1", arm_id="izq", package_id="c", transporter_id="t1"
        )
        assert len(cmd.command_id) == 32


class TestUnloadInmutabilidad:
    def test_no_reasignar(self):
        cmd = UnloadFromTransporter(
            drone_id="d1", arm_id="izq", package_id="c", transporter_id="t1"
        )
        with pytest.raises(FrozenInstanceError):
            cmd.package_id = "otra"  # type: ignore[misc]

    def test_no_anadir(self):
        cmd = UnloadFromTransporter(
            drone_id="d1", arm_id="izq", package_id="c", transporter_id="t1"
        )
        with pytest.raises((AttributeError, TypeError)):
            cmd.x = 1  # type: ignore[attr-defined]


class TestUnloadIgualdad:
    def test_iguales_con_mismo_id(self):
        a = UnloadFromTransporter(
            drone_id="d1", arm_id="izq", package_id="c",
            transporter_id="t1", command_id="x",
        )
        b = UnloadFromTransporter(
            drone_id="d1", arm_id="izq", package_id="c",
            transporter_id="t1", command_id="x",
        )
        assert a == b


class TestUnloadProtocolo:
    def test_cumple_protocol(self):
        cmd = UnloadFromTransporter(
            drone_id="d1", arm_id="izq", package_id="c", transporter_id="t1"
        )
        assert isinstance(cmd, Command)


# ---------------------------------------------------------------------------
# Load y Unload son tipos distintos con FORMAS distintas
# ---------------------------------------------------------------------------
class TestLoadYUnloadSonDistintos:
    """Tras la reconciliación con Sesión A, Load y Unload tienen formas
    diferentes: Load no lleva arm_id, Unload sí. La asimetría refleja
    los efectos del PDDL: Load libera un brazo, Unload ocupa uno.
    """

    def test_son_clases_distintas(self):
        assert LoadIntoTransporter is not UnloadFromTransporter

    def test_un_load_no_es_unload(self):
        load = LoadIntoTransporter(
            drone_id="d1", package_id="c", transporter_id="t1"
        )
        assert not isinstance(load, UnloadFromTransporter)

    def test_un_unload_no_es_load(self):
        unload = UnloadFromTransporter(
            drone_id="d1", arm_id="izq", package_id="c", transporter_id="t1"
        )
        assert not isinstance(unload, LoadIntoTransporter)

    def test_unload_lleva_arm_id_load_no(self):
        load = LoadIntoTransporter(
            drone_id="d1", package_id="c", transporter_id="t1"
        )
        unload = UnloadFromTransporter(
            drone_id="d1", arm_id="izq", package_id="c", transporter_id="t1"
        )
        assert not hasattr(load, "arm_id")
        assert hasattr(unload, "arm_id")
