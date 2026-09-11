"""Transporter: vehículo terrestre con capacidad para varios paquetes.

Un transporter es un componente del dominio que un drone puede arrastrar
entre localizaciones (decisión 1 del diseño: "transporter implícito por
co-localización, no hay dock/undock"). Sirve para cargar varios paquetes
a la vez y trasladarlos en un solo movimiento.

Decisión clave de modelado, coherente con Arm: el Transporter NO almacena
qué paquetes contiene. Esa relación vive exclusivamente en Package.at vía
InTransporter(transporter_id). Una sola fuente de verdad. Consecuencias:

    - Cargar o descargar un paquete solo muta el Package, no el
      Transporter.
    - No hay riesgo de desincronización entre Transporter.content y
      Package.at.
    - "¿Qué hay en t1?" se responde recorriendo los packages del mundo
      y filtrando por InTransporter("t1"). Para los pocos paquetes de
      un escenario docente, irrelevante.
    - La regla "no superar la capacidad" la aplica el Validator
      contando cuántos paquetes tienen InTransporter(este_id) y
      comparando con Transporter.capacity. La capacidad es propiedad
      estructural del vehículo y sí vive aquí.

position se modela como loc_id (str), igual que en Drone y Person.
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Transporter:
    """Vehículo de capacidad fija que un drone puede arrastrar.

    Atributos:
        id: identificador simbólico único en el mundo.
        position: id de la Location actual del transporter.
        capacity: número máximo de paquetes que puede contener
            simultáneamente. El Validator es quien comprueba que no se
            exceda al cargar; aquí solo se declara el límite.
    """

    id: str
    position: str
    capacity: int
