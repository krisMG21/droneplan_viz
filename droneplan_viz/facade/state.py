"""Estado interno de la fachada: el acumulador compartido.

`_FacadeState` es el objeto mutable que vive en DronePlanViz y que TANTO
WorldBuilder como AgentBuilder mutan por referencia (decisión "estado
compartido" de la propuesta). Cada llamada del alumno
(`viz.world.location(...)`, `viz.agents.drone(...)`, `viz.mover(...)`) deja
su huella aquí; `build()` se limita a ensamblar este estado en un World y
un Plan del dominio.

Construcción EAGER: las entidades del dominio se crean dentro de cada
llamada del builder y se guardan ya construidas en los dicts de abajo. No
se difiere nada a `build()`. Esto permite errores tempranos y localizados:
un `at="casa9"` no declarado falla en la línea del `person(...)`, no al
final. Como las entidades del dominio son frozen, guardarlas y luego
referenciarlas por id es seguro: una `Location` declarada se referencia
siempre vía `_state.locations["casa1"]`, sin clonar.

`_QueuedAction` es el registro de una acción encolada todavía SIN su
identidad final. Separa "lo que el usuario dijo" de "lo que la fachada
decidió":

    - command:     el Command con sus campos semánticos y su duración. Su
                   command_id de fábrica (uuid) es irrelevante aquí: build()
                   lo sustituye por el id determinista o por declared_id.
    - start_time:  None si la acción se encoló sin `inicio=` (modo
                   secuencial); float si se dio (modo temporal). Es el
                   discriminante de la regla "todo o nada".
    - duration:    duración de la acción (== command.duration). Se guarda
                   explícita para dejar clara su intención: alimenta tanto
                   la animación como, en el modo secuencial, el cálculo de
                   timestamps encadenados sin solape.
    - declared_id: el id que el alumno pasó por `id=` (la etiqueta del paso
                   PDDL), o None si dejó que la fachada lo generase.

Este módulo es interno (privado del paquete facade). No se reexporta en la
API pública: el alumno nunca construye un _FacadeState a mano.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from droneplan_viz.commands.base import Command
from droneplan_viz.domain.content import Content
from droneplan_viz.domain.drone import Drone
from droneplan_viz.domain.location import Location
from droneplan_viz.domain.package import Package
from droneplan_viz.domain.person import Person
from droneplan_viz.domain.transporter import Transporter


@dataclass(slots=True)
class _QueuedAction:
    """Una acción del plan encolada, antes de fijar su command_id final.

    Atributos:
        command: el Command tal y como lo construyó la acción de la
            fachada (campos semánticos + duración). Su command_id se
            reescribe en build().
        start_time: instante de inicio absoluto si se dio `inicio=`, o
            None si la acción va en modo secuencial.
        duration: duración de la acción. Coincide con command.duration.
        declared_id: id explícito pasado por el usuario vía `id=`, o None.
    """

    command: Command
    start_time: float | None
    duration: float
    declared_id: str | None


@dataclass(slots=True)
class _FacadeState:
    """Acumulador mutable del escenario en construcción.

    Los seis primeros campos son las colecciones de entidades del dominio
    indexadas por id, idénticas en forma a las que recibirá el World.
    `costs` es el grafo de costes ya aplanado (con la simetría expandida
    por el builder antes de llegar aquí). `queued` es la lista ordenada de
    acciones del plan, en orden de llamada.
    """

    locations: dict[str, Location] = field(default_factory=dict)
    contents: dict[str, Content] = field(default_factory=dict)
    persons: dict[str, Person] = field(default_factory=dict)
    packages: dict[str, Package] = field(default_factory=dict)
    drones: dict[str, Drone] = field(default_factory=dict)
    transporters: dict[str, Transporter] = field(default_factory=dict)
    costs: dict[tuple[str, str], float] = field(default_factory=dict)
    queued: list[_QueuedAction] = field(default_factory=list)
