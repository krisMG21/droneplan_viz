"""Arm: brazo de un drone.

Un drone tiene una lista de Arms, identificados por un nombre semántico
("izq", "der", "central", etc.). Estos id son únicos dentro de su drone,
no a nivel global del mundo: dos drones distintos pueden tener cada uno
un brazo "izq".

Decisión clave de modelado: el Arm NO almacena qué paquete sostiene. La
relación "este brazo lleva este paquete" vive exclusivamente en
Package.at, mediante HeldByArm(drone_id, arm_id). Hay una sola fuente de
verdad. Consecuencias:

    - Recoger un paquete solo muta el Package, no el Drone ni su Arm.
    - No hay riesgo de desincronización entre Arm.holding y Package.at.
    - "¿Qué lleva el brazo izq de dron1?" se responde recorriendo los
      paquetes del mundo y filtrando por HeldByArm("dron1", "izq").
      Para los pocos paquetes de un escenario docente, irrelevante.

¿Por qué Arm existe como clase si solo tiene un id?

    1. Modela la realidad: un drone tiene "brazos", no "una lista de
       strings". El tipo lo refleja.
    2. Deja sitio para crecer: si un futuro brazo gana atributos
       (alcance, capacidad de carga, tipo de pinza), se añaden aquí
       sin tocar la firma de Drone.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Arm:
    """Brazo de un drone.

    Atributos:
        id: identificador simbólico del brazo, único dentro de su drone
            (típicamente "izq", "der"). No es globalmente único: el brazo
            "izq" de dron1 y el brazo "izq" de dron2 son distintos físicamente,
            y se identifican siempre por el par (drone_id, arm_id) cuando
            se necesita unicidad global, p. ej. en HeldByArm.
    """

    id: str
