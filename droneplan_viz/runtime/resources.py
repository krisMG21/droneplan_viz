"""
ResourceTable: registro de reservas de recursos por intervalo temporal.

Esta es la pieza central de la concurrencia del PDDL parte 3 (Ejercicio
3 del PDF). Implementa la detección de colisiones entre Commands que
comparten algún recurso durante intervalos temporales solapados.

Reglas del PDF parte 3 modeladas aquí (las cinco):

1. "Cada dron solo puede realizar una acción al mismo tiempo."
   Recurso: ('drone', drone_id). Lo ocupan TODOS los seis Commands.

2. "Una misma caja [...] solo puede ser cogida por un dron."
   Recurso: ('package', package_id). Lo ocupan: PickUp, Deliver,
   LoadIntoTransporter, UnloadFromTransporter.

3. "Un mismo transportador solo puede ser cogido por un dron."
   "Mientras un dron mete o saca una caja de un transportador, ningún
    otro dron puede [...] coger dicho transportador."
   Recurso: ('transporter', transporter_id). Lo ocupan:
   MoveWithTransporter, LoadIntoTransporter, UnloadFromTransporter.

4. "Una persona solo puede recibir una entrega de un dron al mismo
   tiempo."
   Recurso: ('person', person_id). Lo ocupa Deliver.

5. "Todas las acciones que no sean de vuelo tendrán una duración de
   5 segundos. [...] La duración de las acciones de vuelo seguirá
   dependiendo de la función fly-cost."
   La duración la propone el caller en command.duration; el runtime
   solo la honra. El facade es responsable de poner 5.0 o fly-cost
   al construir los Commands desde un .pddl parte 3.

Lo que NO está aquí (YAGNI, NO inventar reglas):

- Brazos como recurso independiente. El PDF no los menciona. La regla
  1 ('un dron, una acción') excluye que el mismo dron use dos brazos
  a la vez. Brazos de drones DISTINTOS no chocan entre sí: cada uno
  está protegido por su propio ('drone', drone_id).

- Localizaciones como recurso. El PDF no las menciona. Dos drones
  pueden coexistir en la misma Location sin conflicto.

- Contenidos. No aplican como recurso temporal.

Decisiones de diseño:

1. La tabla es MUTABLE internamente. La concurrencia es naturalmente
   stateful: 'añadir una reserva' es una operación. Encapsulada dentro
   del runner; no se expone como API pública del paquete.

2. Las reservas se almacenan como tuplas (start, end, owner) por cada
   recurso. owner es opaco (command_id del Command que reservó), útil
   para mensajes de error legibles.

3. Detección de colisión: intervalos cerrado-abierto [start, end).
   Dos intervalos solapan si max(s1,s2) < min(e1,e2). Es la convención
   canónica de tiempo del proyecto, la misma que usa el tiebreak
   END-antes-que-START del runner.

4. resources_of(command) es función libre, no método. Coherente con
   el dominio (validate_X como funciones libres, decisión 2 de Sesión
   A). Permite tests aislados sin construir una tabla.

5. La tabla NO hace 'liberar' una reserva por su lado: las reservas
   expiran naturalmente porque la consulta de colisión compara
   intervalos, y un intervalo cuyo end <= start_nuevo simplemente no
   solapa. Sin embargo, ofrecemos release_before(now) para que el
   runner pueda podar reservas antiguas y mantener la tabla compacta
   en planes largos. release_before NO es necesario para corrección;
   es optimización.

6. CollisionInfo: dataclass devuelta cuando hay colisión, con
   suficiente información para construir un CommandFailure con
   reason informativo (qué recurso, contra qué owner, en qué
   intervalo).
"""
from __future__ import annotations

from dataclasses import dataclass

from droneplan_viz.commands import (
    Command,
    Deliver,
    LoadIntoTransporter,
    Move,
    MoveWithTransporter,
    PickUp,
    UnloadFromTransporter,
)


# ---------------------------------------------------------------------------
# Tipos
# ---------------------------------------------------------------------------
# Etiqueta de un recurso: (tipo, id). Tipo es uno de cuatro literales;
# id es el identificador de la entidad concreta. Lo guardamos como tupla
# en lugar de un dataclass porque las tuplas son hashables nativos,
# comparables por igualdad estructural, y no requieren ceremonial.
#
# Ejemplos: ('drone', 'd1'), ('package', 'libre1'), ('transporter', 't1'),
# ('person', 'ana').
Resource = tuple[str, str]


@dataclass(frozen=True, slots=True)
class Reservation:
    """Reserva inmutable de un recurso durante un intervalo.

    Atributos:
        start: instante de inicio (inclusive).
        end: instante de fin (exclusive). Convención [start, end).
        owner: command_id del Command que hizo la reserva. Opaco; el
            runtime lo usa para construir mensajes de error y para
            facilitar trazas.
    """

    start: float
    end: float
    owner: str


@dataclass(frozen=True, slots=True)
class CollisionInfo:
    """Información sobre una colisión detectada.

    Atributos:
        resource: el recurso compartido que causa la colisión.
        existing: la reserva preexistente con la que choca.
        attempted_start: start_time de la acción que se intentó programar.
        attempted_end: end_time de la acción que se intentó programar.

    Permite al runtime construir un CommandFailure con reason del tipo:
        "drone 'd1' ocupado en [0.0, 10.0) por command 'mov_3'; la
         acción se programó en [5.0, 15.0)"
    """

    resource: Resource
    existing: Reservation
    attempted_start: float
    attempted_end: float

    def describe(self) -> str:
        """Construye un mensaje legible de la colisión.

        Returns:
            Texto humano-legible adecuado para CommandFailure.reason.
        """
        kind, rid = self.resource
        return (
            f"recurso {kind} '{rid}' ocupado en "
            f"[{self.existing.start}, {self.existing.end}) por la acción "
            f"'{self.existing.owner}'; se intentó programar en "
            f"[{self.attempted_start}, {self.attempted_end})"
        )


# ---------------------------------------------------------------------------
# resources_of: qué recursos ocupa cada Command
# ---------------------------------------------------------------------------
def resources_of(command: Command) -> tuple[Resource, ...]:
    """Devuelve el conjunto de recursos que un Command ocupa durante su
    ejecución, según las cinco reglas del PDF parte 3.

    Mapping por tipo de Command:

        Move(d, dest):
            ('drone', d)
        MoveWithTransporter(d, t, dest):
            ('drone', d), ('transporter', t)
        PickUp(d, a, p):
            ('drone', d), ('package', p)
        Deliver(d, p, pe):
            ('drone', d), ('package', p), ('person', pe)
        LoadIntoTransporter(d, p, t):
            ('drone', d), ('package', p), ('transporter', t)
        UnloadFromTransporter(d, a, p, t):
            ('drone', d), ('package', p), ('transporter', t)

    Brazos NO aparecen. Localizaciones NO aparecen. Contenidos NO
    aparecen. Es deliberado: el PDF no los menciona como recursos
    exclusivos.

    Args:
        command: cualquiera de las seis dataclasses Command.

    Returns:
        Tupla de Resource en orden (drone primero, luego los demás
        en orden alfabético del tipo). El orden es estable para
        facilitar tests deterministas, pero no es semánticamente
        significativo: la tabla los procesa como un conjunto.

    Raises:
        TypeError: si el command no es uno de los seis tipos conocidos.
    """
    match command:
        case Move(drone_id=d):
            return (("drone", d),)
        case MoveWithTransporter(drone_id=d, transporter_id=t):
            return (("drone", d), ("transporter", t))
        case PickUp(drone_id=d, package_id=p):
            return (("drone", d), ("package", p))
        case Deliver(drone_id=d, package_id=p, person_id=pe):
            return (("drone", d), ("package", p), ("person", pe))
        case LoadIntoTransporter(
            drone_id=d, package_id=p, transporter_id=t
        ):
            return (("drone", d), ("package", p), ("transporter", t))
        case UnloadFromTransporter(
            drone_id=d, package_id=p, transporter_id=t
        ):
            return (("drone", d), ("package", p), ("transporter", t))
        case _:
            raise TypeError(
                f"resources_of() recibió un objeto que no es un Command "
                f"conocido: {type(command).__name__}"
            )


# ---------------------------------------------------------------------------
# Helper interno: ¿solapan dos intervalos cerrado-abiertos?
# ---------------------------------------------------------------------------
def _intervals_overlap(
    a_start: float, a_end: float, b_start: float, b_end: float
) -> bool:
    """Determina si dos intervalos [start, end) se solapan.

    Convención cerrado-abierto: el end NO está incluido. Dos intervalos
    back-to-back ([0, 5) y [5, 10)) NO solapan. Intervalos puntuales
    (start == end, duración 0) tampoco solapan con nada por convención:
    su 'intervalo' es vacío.

    Args:
        a_start, a_end: extremos del primer intervalo.
        b_start, b_end: extremos del segundo intervalo.

    Returns:
        True si y solo si la intersección [max(starts), min(ends)) es
        no vacía.
    """
    # Intervalos vacíos (start == end) no solapan con nada.
    if a_start >= a_end:
        return False
    if b_start >= b_end:
        return False
    return max(a_start, b_start) < min(a_end, b_end)


# ---------------------------------------------------------------------------
# ResourceTable
# ---------------------------------------------------------------------------
class ResourceTable:
    """Registro mutable de reservas de recursos por intervalo temporal.

    API:
        check(resources, start, end) -> CollisionInfo | None
            ¿Se puede reservar este conjunto de recursos en [start, end)
            sin chocar con reservas preexistentes?
        reserve(resources, start, end, owner) -> None
            Añade una reserva por cada recurso. NO comprueba colisiones;
            asume que el caller llamó a check() antes. El runner siempre
            sigue ese patrón: check, decidir, reservar si check pasó.
        release_before(now) -> None
            Optimización opcional: descarta reservas cuyo end <= now.
            No afecta a corrección; mantiene la tabla compacta en
            planes largos.

    Estado interno: dict mapeando Resource -> list[Reservation]. Mutable
    porque la naturaleza de la concurrencia es stateful. Encapsulado:
    nadie fuera del runner construye una ResourceTable.

    Complejidad: cada check es O(R * K) donde R = nº de recursos del
    Command y K = nº de reservas preexistentes en cada recurso. Para
    los escenarios del TFG (pocos drones, pocas decenas de Commands),
    esto es irrelevante. Si en un futuro se quisiera escalar, se puede
    sustituir cada list[Reservation] por un interval tree sin tocar
    la API.
    """

    __slots__ = ("_reservations",)

    def __init__(self) -> None:
        """Tabla vacía. Las reservas se añaden con reserve()."""
        self._reservations: dict[Resource, list[Reservation]] = {}

    def check(
        self,
        resources: tuple[Resource, ...],
        start: float,
        end: float,
    ) -> CollisionInfo | None:
        """Comprueba si una nueva reserva chocaría con las existentes.

        Devuelve la primera colisión encontrada (recorriendo los
        recursos en el orden recibido y las reservas en el orden de
        inserción). Si hay varias colisiones simultáneas, solo se
        reporta una; el usuario corrige la primera, vuelve a planificar
        y descubre la siguiente. Patrón coherente con el Validator de
        el runtime: 'devolver al primer fallo'.

        Args:
            resources: tupla de recursos que la nueva acción ocupa.
            start: inicio del intervalo (inclusive).
            end: fin del intervalo (exclusive).

        Returns:
            CollisionInfo si hay choque, None si la reserva es viable.
        """
        # Intervalos puntuales (duración 0) no pueden colisionar con
        # nada por la convención de _intervals_overlap.
        if start >= end:
            return None

        for resource in resources:
            existing_list = self._reservations.get(resource, ())
            for existing in existing_list:
                if _intervals_overlap(
                    start, end, existing.start, existing.end
                ):
                    return CollisionInfo(
                        resource=resource,
                        existing=existing,
                        attempted_start=start,
                        attempted_end=end,
                    )
        return None

    def reserve(
        self,
        resources: tuple[Resource, ...],
        start: float,
        end: float,
        owner: str,
    ) -> None:
        """Registra una reserva para cada recurso en el intervalo dado.

        NO comprueba colisiones. El runner llama primero a check();
        si no hay colisión, llama a reserve(). Separar las dos
        operaciones permite reportar la colisión sin haber escrito
        ya en la tabla, lo que sería un side effect indeseable.

        Reservas de duración 0 (start >= end) se ignoran: representan
        Commands con duration=0.0 que no tienen presencia temporal y
        por tanto no necesitan reserva. Coherente con check(), que
        las trata como no-colisionables.

        Args:
            resources: recursos a reservar.
            start: inicio del intervalo.
            end: fin del intervalo.
            owner: command_id de quien reserva.
        """
        if start >= end:
            # Reserva vacía: no se almacena.
            return

        reservation = Reservation(start=start, end=end, owner=owner)
        for resource in resources:
            if resource not in self._reservations:
                self._reservations[resource] = []
            self._reservations[resource].append(reservation)

    def release_before(self, now: float) -> None:
        """Descarta todas las reservas que terminan antes o en `now`.

        Optimización opcional: mantiene la tabla compacta. El bucle de
        ejecución por eventos puede llamar a esta función tras procesar
        cada evento END para evitar acumular reservas obsoletas en
        planes largos.

        Una reserva con end == now SE descarta: por convención
        cerrado-abierto, en el instante `now` la acción ya está
        completada (no es activa); su recurso ya está libre.

        Args:
            now: instante hasta el que se podan reservas (inclusive
                respecto al end).
        """
        for resource in list(self._reservations.keys()):
            self._reservations[resource] = [
                r for r in self._reservations[resource] if r.end > now
            ]
            if not self._reservations[resource]:
                del self._reservations[resource]

    def reservations_for(
        self, resource: Resource
    ) -> tuple[Reservation, ...]:
        """Devuelve las reservas activas para un recurso, como tupla
        inmutable.

        Útil para tests y para introspección. NO se ofrece como acceso
        a la lista mutable interna: el snapshot que devuelve esta
        función es seguro.
        """
        return tuple(self._reservations.get(resource, ()))

    def __len__(self) -> int:
        """Número TOTAL de reservas en la tabla (sumando todos los
        recursos). Útil para tests y assertions defensivas.
        """
        return sum(
            len(lst) for lst in self._reservations.values()
        )

    def is_empty(self) -> bool:
        """True si la tabla no contiene ninguna reserva."""
        return len(self) == 0
