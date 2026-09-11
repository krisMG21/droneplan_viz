"""
Commands de manipulación directa: PickUp y Deliver.

Estos Commands representan las dos acciones del dominio PDDL en las que el
drone interactúa con un paquete usando un brazo:

- PickUp: recoger un paquete del suelo (acción 'recoger' del PDDL).
- Deliver: entregar un paquete sostenido a una persona (acción 'entregar').

Decisión clave de modelado (heredada, establecida en el
Validator): asimetría entre PickUp y Deliver respecto al brazo.

- PickUp lleva arm_id obligatorio: la acción OCUPA un brazo concreto.
  El efecto del PDDL es (sujetando ?d ?b ?c) con ?b específico, y por
  tanto hay que decir cuál. La función validate_pick_up del Validator
  exige arm_id como parámetro.

- Deliver NO lleva arm_id: el efecto del PDDL es LIBERAR el brazo que
  sostenía el paquete, sea cual sea. La función validate_deliver del
  Validator no recibe arm_id: comprueba que ALGÚN brazo del drone
  sostiene el paquete. Cita literal del validator.py:
  "no se comprueba qué brazo concreto: la API de la facade no lo exige
  porque el efecto de la entrega libera el brazo que sea."

Esta asimetría es la regla del proyecto: simetría perfecta entre cada
Command y su validate_X correspondiente. Mantener arm_id en Deliver pero
ignorarlo en validate sería datos zombies.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from droneplan_viz.commands.base import new_command_id


@dataclass(frozen=True, slots=True)
class PickUp:
    """Acción 'recoger': el drone toma un paquete con uno de sus brazos.

    Mapea uno-a-uno con validate_pick_up(world, drone_id, package_id,
    arm_id) del Validador.

    Atributos:
        drone_id: identificador del drone que recoge.
        arm_id: identificador del brazo (relativo al drone) con el que
            recoge. El Validador comprueba que el brazo existe en el
            drone y está libre.
        package_id: identificador del paquete a recoger. El Validador
            comprueba que el paquete está libre en la misma localización
            que el drone.
        duration: duración semántica. Parte 1-2: 0.0; parte 3: 5.0
            (constante para acciones no-vuelo, según enunciado).
        command_id: identidad única, auto-generada por defecto.
    """
    drone_id: str
    arm_id: str
    package_id: str
    duration: float = 0.0
    command_id: str = field(default_factory=new_command_id)


@dataclass(frozen=True, slots=True)
class Deliver:
    """Acción 'entregar': el drone da un paquete sostenido a una persona.

    Mapea uno-a-uno con validate_deliver(world, drone_id, package_id,
    person_id) del Validador.

    Atributos:
        drone_id: identificador del drone que entrega.
        package_id: identificador del paquete a entregar. El Validador
            verifica que algún brazo de este drone lo sostiene; el brazo
            concreto se deriva del estado, no se especifica en el Command.
        person_id: identificador de la persona destinataria. El Validador
            comprueba que está co-localizada con el drone y necesita
            el contenido del paquete.
        duration: ver PickUp.duration.
        command_id: identidad única, auto-generada por defecto.

    Nota explícita: este Command NO lleva arm_id. Decisión heredada de
    el runtime. El handler apply_deliver determina qué brazo libera
    consultando world.packages para encontrar dónde está package_id.
    """
    drone_id: str
    package_id: str
    person_id: str
    duration: float = 0.0
    command_id: str = field(default_factory=new_command_id)
