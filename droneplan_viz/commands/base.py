"""
Paquete commands: patrón Command como objetos inmutables.

Este módulo define la superficie común a todos los Commands:
- El Protocol Command, contrato estructural que cumplen las seis dataclasses
  de acción (Move, MoveWithTransporter, PickUp, Deliver, LoadIntoTransporter,
  UnloadFromTransporter).
- La factory new_command_id() utilizada como default_factory en cada Command
  para auto-generar identidades únicas cuando el caller no las proporciona.

Decisiones de diseño documentadas:

- Protocol estructural en lugar de clase base abstracta. La el runtime consagró
  composición sobre herencia; las dataclasses Command son dato puro y no
  comparten comportamiento, solo estructura. Un Protocol expresa exactamente
  eso sin imponer una jerarquía artificial.

- El Protocol contiene únicamente lo común a TODAS las acciones:
  command_id, drone_id, duration. Campos específicos (arm_id, package_id,
  person_id, transporter_id, destination_id) viven en las dataclasses
  concretas y se acceden vía match/isinstance en los dispatchers.

- new_command_id() devuelve uuid.uuid4().hex (32 caracteres hex, sin guiones)
  para legibilidad en logs y trazas. La unicidad la garantiza UUID4.

- El default_factory permite override explícito por parte del facade o de
  los tests, que pueden inyectar ids legibles como "move_1", "plan_step_3".
"""
from __future__ import annotations

import uuid
from typing import Protocol, runtime_checkable


def new_command_id() -> str:
    """Genera un identificador único para un Command.

    Usado como default_factory en el campo command_id de cada dataclass
    Command. Devuelve el hex de un UUID4 (32 caracteres, sin guiones)
    para mantener los ids cortos y legibles en logs y mensajes de error.
    """
    return uuid.uuid4().hex


@runtime_checkable
class Command(Protocol):
    """Contrato estructural común a todas las acciones del dominio.

    Las seis dataclasses concretas (Move, MoveWithTransporter, PickUp,
    Deliver, LoadIntoTransporter, UnloadFromTransporter) cumplen este
    Protocol implícitamente al exponer los tres atributos siguientes.
    No hace falta heredar ni registrarse; basta con tener los campos.

    Atributos:
        command_id: identidad única, opaca, generada por new_command_id()
            salvo override explícito. Garantiza que dos invocaciones de
            "la misma acción" sean Commands distintos a efectos de
            igualdad, historial y trazabilidad.
        drone_id: identificador del drone que ejecuta la acción. Todas
            las acciones del dominio son protagonizadas por un único
            drone, así que el campo es universal.
        duration: duración semántica declarada por el plan (parte 3 del
            PDDL: durativas) o 0.0 cuando la acción es atemporal (partes
            1 y 2). El renderer puede aplicar una duración visual aparte;
            ese es problema de la capa render.

    Nota sobre runtime_checkable: marcado para permitir isinstance(x,
    Command) en tests y assertions defensivas. Verifica la presencia
    de los atributos por nombre, no su tipo. Los handlers concretos
    NO dependen de isinstance(Command); usan match exhaustivo sobre
    las dataclasses específicas.
    """

    @property
    def command_id(self) -> str: ...

    @property
    def drone_id(self) -> str: ...

    @property
    def duration(self) -> float: ...
