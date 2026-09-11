"""Package: paquete transportable que contiene un Content.

Modela los objetos que los drones recogen, transportan y entregan. Cada
paquete tiene un contenido fijo (la categoría que lleva dentro) y una
ubicación que puede estar en uno de tres estados mutuamente excluyentes:

    - libre en una Location del grafo,
    - sujeto por un brazo concreto de un drone concreto,
    - dentro de un transporter.

En lugar de modelar esto con tres campos opcionales independientes y dejar
al Validator vigilar que "exactamente uno es no-None", usamos un tipo suma
(tagged union). El invariante de cardinalidad lo garantiza el tipo: por
construcción, un Package está en exactamente un sitio. Esto es una
aplicación directa del principio "make illegal states unrepresentable":
estados imposibles no se pueden expresar en el código fuente.

Beneficios concretos:
    - El Validator no comprueba la cardinalidad; solo las preconditions
      semánticas (que el loc_id referenciado exista, etc.).
    - Las consultas se hacen con isinstance/match y son exhaustivas: el
      type checker avisa si se olvida un caso.
    - La intención queda explícita en el tipo del campo, no escondida
      en una invariante documentada en otro sitio.

Los tres subtipos de PackageLocation reutilizan el vocabulario PDDL del
dominio: at, holding, in. Esto facilita la traducción manual del plan
generado por el planificador a llamadas de la API, que es justamente el
flujo pedagógico de la asignatura (decisión 8 del diseño).
"""

from dataclasses import dataclass

from droneplan_viz.domain.content import Content


# ---------------------------------------------------------------------------
# Tagged union: dónde está un Package
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AtLocation:
    """Paquete libre, depositado en una localización del grafo.

    Corresponde al predicado PDDL (at ?paquete ?loc).
    """

    loc_id: str


@dataclass(frozen=True, slots=True)
class HeldByArm:
    """Paquete sujeto por un brazo concreto de un drone concreto.

    Corresponde al predicado PDDL (holding ?dron ?brazo ?paquete) o
    equivalente según la variante del dominio. Identificamos el brazo por
    el par (drone_id, arm_id) porque los arm_id ("izq", "der") solo son
    únicos dentro de su drone.
    """

    drone_id: str
    arm_id: str


@dataclass(frozen=True, slots=True)
class InTransporter:
    """Paquete cargado dentro de un transporter.

    Corresponde al predicado PDDL (in ?paquete ?transportador).
    """

    transporter_id: str


# Alias del tipo suma. Cualquier ubicación posible de un Package es uno de
# estos tres. mypy/pyright tratarán esto como una unión discriminable y
# avisarán si un match no es exhaustivo.
PackageLocation = AtLocation | HeldByArm | InTransporter


# ---------------------------------------------------------------------------
# Package
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Package:
    """Paquete transportable.

    Atributos:
        id: identificador simbólico único en el mundo.
        contains: Content que transporta. Es inmutable durante toda la vida
            del paquete: un paquete no cambia de categoría.
        at: ubicación actual, expresada como uno de los tres subtipos de
            PackageLocation. Garantía estructural de exclusividad mutua.

    Las transiciones entre ubicaciones se hacen creando una nueva instancia
    con dataclasses.replace, no mutando: misma filosofía que Person.receive.
    """

    id: str
    contains: Content
    at: PackageLocation
