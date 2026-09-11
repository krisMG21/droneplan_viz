"""
Commands de carga/descarga de transportador: LoadIntoTransporter y
UnloadFromTransporter.

Estos Commands representan las dos acciones del dominio PDDL (introducidas
en parte 2) que mueven paquetes entre un brazo del drone y un transportador
co-localizado:

- LoadIntoTransporter: el drone mete una caja sostenida en el transportador
  (acción 'poner-caja-en-transportador' del PDDL).
- UnloadFromTransporter: el drone saca una caja del transportador y la
  agarra con uno de sus brazos (acción 'coger-caja-del-transportador').

Decisión clave de modelado (heredada, establecida en el
Validator): asimetría entre Load y Unload respecto al brazo, paralela a
la de Deliver/PickUp.

- LoadIntoTransporter NO lleva arm_id: el efecto del PDDL es LIBERAR
  el brazo que sostenía el paquete, sea cual sea. validate_load_into_
  transporter del Validator no recibe arm_id; comprueba que ALGÚN brazo
  del drone sostiene el paquete.

- UnloadFromTransporter SÍ lleva arm_id: el efecto es OCUPAR un brazo
  concreto con el paquete extraído. validate_unload_from_transporter
  exige arm_id como parámetro.

Mapeo con el Validator:
- LoadIntoTransporter -> validate_load_into_transporter
    (world, drone_id, package_id, transporter_id)
- UnloadFromTransporter -> validate_unload_from_transporter
    (world, drone_id, package_id, transporter_id, arm_id)
"""
from __future__ import annotations

from dataclasses import dataclass, field

from droneplan_viz.commands.base import new_command_id


@dataclass(frozen=True, slots=True)
class LoadIntoTransporter:
    """Acción 'poner-caja-en-transportador'.

    Atributos:
        drone_id: identificador del drone que carga.
        package_id: identificador del paquete a cargar. El Validador
            comprueba que algún brazo del drone lo sostiene; el brazo
            concreto se deriva del estado.
        transporter_id: identificador del transportador receptor. El
            Validador exige co-localización con el drone y capacidad
            disponible.
        duration: duración semántica de la acción.
        command_id: identidad única.

    Nota: este Command NO lleva arm_id, en consonancia con validate_load_
    into_transporter del Validator. El handler apply_load_into_transporter
    deriva qué brazo libera consultando world.packages.
    """
    drone_id: str
    package_id: str
    transporter_id: str
    duration: float = 0.0
    command_id: str = field(default_factory=new_command_id)


@dataclass(frozen=True, slots=True)
class UnloadFromTransporter:
    """Acción 'coger-caja-del-transportador'.

    Atributos:
        drone_id: identificador del drone que descarga.
        arm_id: identificador del brazo libre con el que recoge el
            paquete del transportador. Tras la acción, el brazo
            sostiene el paquete.
        package_id: identificador del paquete a descargar.
        transporter_id: identificador del transportador de origen.
        duration: duración semántica de la acción.
        command_id: identidad única.

    Lleva arm_id obligatorio porque el efecto ocupa un brazo concreto,
    paralelo a PickUp.
    """
    drone_id: str
    arm_id: str
    package_id: str
    transporter_id: str
    duration: float = 0.0
    command_id: str = field(default_factory=new_command_id)
