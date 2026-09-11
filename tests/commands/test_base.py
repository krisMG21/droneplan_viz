"""Tests de commands/base.py: Protocol Command y new_command_id()."""
from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from droneplan_viz.commands.base import Command, new_command_id


# ---------------------------------------------------------------------------
# new_command_id()
# ---------------------------------------------------------------------------
class TestNewCommandId:
    def test_devuelve_string(self):
        cid = new_command_id()
        assert isinstance(cid, str)

    def test_no_vacio(self):
        cid = new_command_id()
        assert len(cid) > 0

    def test_es_uuid4_hex_32_caracteres(self):
        cid = new_command_id()
        assert len(cid) == 32
        # uuid4().hex es hexadecimal puro, sin guiones
        assert all(c in "0123456789abcdef" for c in cid)

    def test_dos_invocaciones_producen_ids_distintos(self):
        a = new_command_id()
        b = new_command_id()
        assert a != b

    def test_muchas_invocaciones_unicas(self):
        ids = {new_command_id() for _ in range(1000)}
        assert len(ids) == 1000


# ---------------------------------------------------------------------------
# Protocol Command
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class _DummyCommand:
    """Dataclass mínima que cumple el contrato Command. Usada solo en tests."""
    drone_id: str
    duration: float = 0.0
    command_id: str = field(default_factory=new_command_id)


@dataclass(frozen=True, slots=True)
class _PartialCommand:
    """Dataclass que NO cumple Command (falta command_id). Solo en tests."""
    drone_id: str
    duration: float


class TestCommandProtocol:
    def test_dataclass_minima_cumple_protocol(self):
        cmd = _DummyCommand(drone_id="d1")
        assert isinstance(cmd, Command)

    def test_un_objeto_sin_los_campos_no_cumple_protocol(self):
        class NotACommand:
            pass
        assert not isinstance(NotACommand(), Command)

    def test_un_objeto_con_solo_dos_de_los_tres_campos_no_cumple_protocol(self):
        assert not isinstance(_PartialCommand(drone_id="d1", duration=0.0), Command)


# ---------------------------------------------------------------------------
# Comportamiento de la factory como default_factory
# ---------------------------------------------------------------------------
class TestCommandIdEnDataclass:
    def test_command_id_se_autogenera_si_no_se_pasa(self):
        cmd = _DummyCommand(drone_id="d1")
        assert cmd.command_id  # no vacío
        assert len(cmd.command_id) == 32

    def test_command_id_explicito_se_respeta(self):
        cmd = _DummyCommand(drone_id="d1", command_id="my_custom_id")
        assert cmd.command_id == "my_custom_id"

    def test_dos_dataclasses_creadas_consecutivamente_tienen_ids_distintos(self):
        a = _DummyCommand(drone_id="d1")
        b = _DummyCommand(drone_id="d1")
        assert a.command_id != b.command_id

    def test_dataclass_es_inmutable(self):
        """frozen=True impide reasignar campos existentes."""
        from dataclasses import FrozenInstanceError
        cmd = _DummyCommand(drone_id="d1")
        with pytest.raises(FrozenInstanceError):
            cmd.command_id = "hack"  # type: ignore[misc]

    def test_dataclass_rechaza_atributos_nuevos(self):
        """frozen=True + slots=True impide añadir campos no declarados.

        En Python 3.12 la excepción concreta es TypeError (no AttributeError)
        debido al orden en que frozen intercepta __setattr__ antes que slots.
        Lo importante semánticamente es que NO se pueda hacer la asignación.
        """
        cmd = _DummyCommand(drone_id="d1")
        with pytest.raises((AttributeError, TypeError)):
            cmd.nuevo_campo = "x"  # type: ignore[attr-defined]
