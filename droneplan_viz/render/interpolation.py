"""
Interpolation: clasificador de transiciones entre dos WorldSnapshot.

Este módulo NO toca pygame y NO dibuja. Su única responsabilidad es,
dado un par (snap_a, snap_b), devolver un objeto Transition que
describa qué cambió entre ambos. El painter.py consumirá la Transition
para decidir qué interpolar visualmente.

Filosofía:

- El clasificador inspecciona `snap_b.produced_by` (el Command que generó
  el snapshot end), no diffea los Worlds.
- Razón: el Command nos da semántica explícita ("esto fue un Move").
  Diffear worlds sería frágil (¿cómo distinguir un Deliver de un
  movimiento de paquete del suelo a otra loc? Por orden de cambios,
  inviable). El Command lo dice directamente.
- Dispatch por isinstance/match sobre las seis dataclasses Command de
  el runtime. Si el runtime añade un Command nuevo en el futuro, el
  classifier lanza un error explícito en vez de devolver TransitionStatic
  silenciosamente: fail-fast.

Casos cubiertos por el classifier:

1. snap_b.produced_by is None        → TransitionStatic (snapshot inicial).
2. snap_b corresponde a un FALLO     → TransitionFailure (drone en ERROR).
3. snap_b.produced_by is Move        → TransitionDroneMove (sin transporter).
4. snap_b.produced_by is MoveWithT…  → TransitionDroneMove (con transporter).
5. snap_b.produced_by is PickUp      → TransitionPackageMove (loc → arm).
6. snap_b.produced_by is Deliver     → TransitionPackageMove (arm → loc), persona recibe.
7. snap_b.produced_by is LoadIntoT…  → TransitionPackageMove (arm → transp).
8. snap_b.produced_by is UnloadFromT → TransitionPackageMove (transp → arm).
9. Snapshot start (drone.state cambió a MOVING/INTERACTING) en duración>0:
   se trata como pseudo-static visualmente (el dron está en su loc de
   origen "preparándose"); el render NO interpola en este tramo (pasa
   en bloque cuando el Timeline avanza al snap_start). Detectar este
   caso es importante: tras el snap_start del Move, el SIGUIENTE par
   (snap_start, snap_end) sí lleva la interpolación de movimiento.

Detección de "snap_b es el start de una acción durativa":

- snap_a y snap_b tienen el MISMO produced_by (el snap_start lleva el
  Command, y el snap_end también; ambos son del mismo Command, solo
  cambia el timestamp y el drone.state). Memoria C §4: para
  duration>0, el runner produce DOS snapshots con el mismo
  produced_by.
- Si detectamos snap_a.produced_by is snap_b.produced_by (identidad,
  porque los Command tienen command_id único), entonces snap_b es el
  END de la acción y snap_a es el START. Es el tramo de interpolación
  "rica". Esto es el caso típico.
- Caso alternativo: snap_a.produced_by != snap_b.produced_by → snap_a
  es el snap_end (o inicial) y snap_b es el snap_start del siguiente
  Command. Aquí la transición visual es "trivial": no hay nada que
  animar entre ambos (el world no cambió de manera significativa, solo
  drone.state pasó a MOVING/INTERACTING). Devolvemos TransitionStatic.

  Subcaso especial dentro de este alternativo: cuando snap_a es el
  snapshot INICIAL (sin produced_by) y snap_b es el snap_start del
  PRIMER Command durativo, no podemos detectarlo por comparación de
  command_id (snap_a.produced_by is None). En ese caso lo
  reconocemos porque el drone en snap_b está en MOVING/INTERACTING,
  no en IDLE. Esa propiedad es invariante por la FSM del drone
  (el runtime §2 + el runtime §4): un snap_end SIEMPRE deja el drone en
  IDLE, un snap_start lo deja en MOVING/INTERACTING. Sin este
  reconocimiento, la animación del primer Command durativo se
  ejecutaría dos veces: una en el tramo inicial→start (incorrecto) y
  otra en el tramo start→end (correcto). Bug detectado por la
  integración con el runtime (UI envolvente) tras el cierre.

Por simplicidad, este módulo devuelve la transición principal asumiendo
que la pareja (snap_a, snap_b) es "el cambio importante". El Timeline
del Paso 6 garantizará que las parejas pasadas al painter sean siempre
el par (start, end) del mismo Command o (end_anterior, end_actual) en
caso de Commands con duration=0.

API pública:

    classify_transition(snap_a, snap_b) -> Transition

    Transition = TransitionStatic
               | TransitionDroneMove
               | TransitionPackageMove
               | TransitionFailure

Cada subtipo es una dataclass frozen+slots con los datos mínimos para
que el painter dibuje el frame intermedio.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from droneplan_viz.commands import (
    Deliver,
    LoadIntoTransporter,
    Move,
    MoveWithTransporter,
    PickUp,
    UnloadFromTransporter,
)
from droneplan_viz.domain import AtLocation, HeldByArm, InTransporter
from droneplan_viz.domain.drone_state import DroneState
from droneplan_viz.history import WorldSnapshot


# ---------------------------------------------------------------------------
# Tipos: una dataclass por categoría de transición
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TransitionStatic:
    """Sin animación entre snap_a y snap_b. El painter dibuja snap_b directo.

    Casos:
    - snap_b es el snapshot inicial (produced_by is None).
    - snap_a y snap_b son el mismo snapshot (caso degenerado, Timeline
      en uno de los extremos).
    - snap_a es snap_end de un Command durativo y snap_b es snap_start
      del siguiente Command (sin cambio visual significativo entre
      ellos).
    """


@dataclass(frozen=True, slots=True)
class TransitionDroneMove:
    """El drone (y opcionalmente un transporter) cambia de localización.

    Atributos:
        drone_id: id del drone que se mueve.
        from_loc_id: localización origen.
        to_loc_id: localización destino.
        transporter_id: id del transporter arrastrado, o None si fue
            un Move simple. Para MoveWithTransporter, lleva el id del
            transporter; los paquetes que viajan dentro de él se
            mueven implícitamente (su posición se deriva del transp).
    """

    drone_id: str
    from_loc_id: str
    to_loc_id: str
    transporter_id: str | None


# Ubicación lógica de un paquete a efectos de interpolación.
# "loc"    → paquete libre en una localización (AtLocation).
# "arm"    → paquete sostenido por un brazo (HeldByArm).
# "transp" → paquete dentro de un transportador (InTransporter).
PackagePlaceKind = Literal["loc", "arm", "transp"]


@dataclass(frozen=True, slots=True)
class PackagePlace:
    """Ubicación lógica de un paquete, normalizada para interpolación.

    Atributos:
        kind: una de "loc", "arm", "transp".
        loc_id: para kind="loc", el loc_id donde está el paquete libre.
            None para "arm" y "transp".
        drone_id: para kind="arm", el drone que lo sostiene.
            None para "loc" y "transp".
        arm_id: para kind="arm", el brazo que lo sostiene.
            None para "loc" y "transp".
        transporter_id: para kind="transp", el transp que lo contiene.
            None para "loc" y "arm".

    El painter usa esta info junto con WorldLayout para resolver la
    posición física en pantalla del paquete en cada extremo.
    """

    kind: PackagePlaceKind
    loc_id: str | None = None
    drone_id: str | None = None
    arm_id: str | None = None
    transporter_id: str | None = None


@dataclass(frozen=True, slots=True)
class TransitionPackageMove:
    """Un paquete cambia de "manos": loc, brazo o transp.

    Cubre los cuatro Commands de manipulación: PickUp, Deliver,
    LoadIntoTransporter, UnloadFromTransporter.

    Atributos:
        drone_id: drone protagonista de la acción. Aunque la animación
            visual se centra en el paquete, el drone es quien la
            ejecuta y debe pintarse en estado INTERACTING durante el
            tramo.
        package_id: paquete que se mueve.
        from_place: dónde estaba el paquete al inicio.
        to_place: dónde está el paquete al final.
        person_id: solo para Deliver. None en los otros casos. Permite
            al painter dibujar al destinatario con alguna pista visual
            extra (recepción).
    """

    drone_id: str
    package_id: str
    from_place: PackagePlace
    to_place: PackagePlace
    person_id: str | None = None


@dataclass(frozen=True, slots=True)
class TransitionFailure:
    """El drone pasó a ERROR. Sin interpolación física.

    El painter dibuja snap_b directamente (con el drone en rojo y X).
    No tiene sentido interpolar entre "drone vivo" y "drone en ERROR".

    Atributos:
        drone_id: drone afectado.
        command_id: id del Command que falló. Útil para overlays
            futuros (no usado por el painter del motor de render;
            queda como dato).
    """

    drone_id: str
    command_id: str


# Unión de las categorías. El painter hace match exhaustivo sobre ella.
Transition = (
    TransitionStatic
    | TransitionDroneMove
    | TransitionPackageMove
    | TransitionFailure
)


# ---------------------------------------------------------------------------
# Clasificador principal
# ---------------------------------------------------------------------------


def classify_transition(
    snap_a: WorldSnapshot,
    snap_b: WorldSnapshot,
) -> Transition:
    """Clasifica la transición entre dos WorldSnapshots consecutivos.

    Args:
        snap_a: snapshot inicial del tramo (típicamente el snap_start
            de un Command durativo, o el snap_end del Command anterior).
        snap_b: snapshot final del tramo (típicamente el snap_end del
            mismo Command, o un snap_start/end siguiente).

    Returns:
        Una de las cuatro dataclasses Transition.

    Raises:
        TypeError: si snap_b.produced_by es un objeto que no es uno de
            los seis Commands conocidos. Esto indica una inconsistencia
            entre Sesiones B y D y debe ser corregida en código.
    """
    cmd = snap_b.produced_by

    # Caso 1: snapshot inicial (sin Command que lo produzca).
    if cmd is None:
        return TransitionStatic()

    # Caso 2: snap_a == snap_b (mismo snapshot). Degenerate, no animar.
    if snap_a is snap_b:
        return TransitionStatic()

    # Caso 3: el snap_b representa un FALLO.
    # Detección: el drone implicado por el Command está en estado ERROR
    # en snap_b.world (Memoria C §2.12). Esto siempre tiene prioridad
    # sobre el tipo de Command: si falló, no interpolamos movimiento.
    drone = snap_b.world.drones.get(cmd.drone_id)
    if drone is not None and drone.state == DroneState.ERROR:
        return TransitionFailure(
            drone_id=cmd.drone_id,
            command_id=cmd.command_id,
        )

    # Caso 4: snap_b solo refleja el efecto at-START de un Command durativo
    # (el drone protagonista entra a MOVING/INTERACTING sin que la geometría
    # haya cambiado aún). No hay nada que animar todavía: la geometría se
    # moverá en el tramo (snap_start → snap_end).
    #
    # El discriminador es la FSM del drone en snap_b (invariante de Sesión
    # C), NO la comparación de command_id. Razón: en planes CONCURRENTES dos
    # snapshots ADYACENTES pueden venir de Commands distintos (de drones
    # distintos, por el interleaving de eventos) y AUN ASÍ contener geometría
    # que animar — p.ej. el snap_end del Move de un drone seguido del
    # snap_start de otro. Comparar command_id clasificaba esos tramos como
    # Static y hacía que los drones "se teletransportaran". La FSM resuelve
    # ambos casos (secuencial y concurrente) de forma uniforme:
    #   - protagonista en MOVING/INTERACTING en snap_b  → snap_START → Static.
    #   - protagonista en IDLE en snap_b                → snap_END   → animar.
    b_drone = snap_b.world.drones.get(cmd.drone_id)
    if b_drone is not None and b_drone.state in (
        DroneState.MOVING, DroneState.INTERACTING,
    ):
        return TransitionStatic()

    # Caso 5: snap_b tiene efectos at-end completos (drone IDLE) y hay
    # diferencia geométrica que animar: snap_start→snap_end del MISMO
    # Command durativo, el único snapshot de un Command instantáneo (PDDL
    # parte 1-2), o —en concurrencia— el snap_end de un Command cuyo
    # snap_start quedó en un tramo anterior (no adyacente).
    return _classify_by_command(cmd, snap_a, snap_b)


def _classify_by_command(
    cmd,
    snap_a: WorldSnapshot,
    snap_b: WorldSnapshot,
) -> Transition:
    """Dispatch por tipo de Command. Asume cmd no es None y no es fallo.

    Asume invariante: la acción se completó correctamente, así que
    los efectos PDDL están aplicados en snap_b.world.

    Para PickUp/Deliver/Load/Unload, hay que reconstruir el "from" y
    "to" del paquete; algunos se infieren del Command, otros se leen
    del world.
    """
    # ---- Movimiento de drone ----
    if isinstance(cmd, Move):
        # snap_a.world tiene al drone en su origen. Lo leemos de ahí
        # porque cmd.destination_id es el destino, pero cmd no lleva
        # el origen (la API del Validator no lo necesita).
        from_loc = _drone_position_in(snap_a, cmd.drone_id)
        return TransitionDroneMove(
            drone_id=cmd.drone_id,
            from_loc_id=from_loc,
            to_loc_id=cmd.destination_id,
            transporter_id=None,
        )

    if isinstance(cmd, MoveWithTransporter):
        from_loc = _drone_position_in(snap_a, cmd.drone_id)
        return TransitionDroneMove(
            drone_id=cmd.drone_id,
            from_loc_id=from_loc,
            to_loc_id=cmd.destination_id,
            transporter_id=cmd.transporter_id,
        )

    # ---- Manipulación de paquetes ----
    if isinstance(cmd, PickUp):
        # Origen: paquete libre en la loc del drone (snap_a tiene el
        # paquete AtLocation(loc_id)). Destino: brazo del drone.
        pkg_a = snap_a.world.packages[cmd.package_id]
        from_loc_id = _atlocation_loc_id_or_raise(pkg_a, cmd.package_id, "snap_a", "PickUp")
        return TransitionPackageMove(
            drone_id=cmd.drone_id,
            package_id=cmd.package_id,
            from_place=PackagePlace(kind="loc", loc_id=from_loc_id),
            to_place=PackagePlace(
                kind="arm",
                drone_id=cmd.drone_id,
                arm_id=cmd.arm_id,
            ),
            person_id=None,
        )

    if isinstance(cmd, Deliver):
        # Origen: paquete en algún brazo del drone (snap_a tiene
        # HeldByArm). Para saber qué brazo, consultamos snap_a (porque
        # el Command de Deliver NO lleva arm_id, decisión 2.5).
        pkg_a = snap_a.world.packages[cmd.package_id]
        if not isinstance(pkg_a.at, HeldByArm):
            raise ValueError(
                f"Deliver inconsistente: el paquete {cmd.package_id!r} "
                f"debería estar HeldByArm en snap_a, pero está en "
                f"{type(pkg_a.at).__name__}."
            )
        from_drone = pkg_a.at.drone_id
        from_arm = pkg_a.at.arm_id
        # Destino: el paquete queda libre en la loc de la persona
        # (Memoria B §4: "package.at := AtLocation(person.position)").
        person = snap_b.world.persons[cmd.person_id]
        return TransitionPackageMove(
            drone_id=cmd.drone_id,
            package_id=cmd.package_id,
            from_place=PackagePlace(
                kind="arm",
                drone_id=from_drone,
                arm_id=from_arm,
            ),
            to_place=PackagePlace(kind="loc", loc_id=person.position),
            person_id=cmd.person_id,
        )

    if isinstance(cmd, LoadIntoTransporter):
        # Origen: paquete en algún brazo (Command sin arm_id). Destino: transp.
        pkg_a = snap_a.world.packages[cmd.package_id]
        if not isinstance(pkg_a.at, HeldByArm):
            raise ValueError(
                f"LoadIntoTransporter inconsistente: el paquete "
                f"{cmd.package_id!r} debería estar HeldByArm en snap_a, "
                f"pero está en {type(pkg_a.at).__name__}."
            )
        from_drone = pkg_a.at.drone_id
        from_arm = pkg_a.at.arm_id
        return TransitionPackageMove(
            drone_id=cmd.drone_id,
            package_id=cmd.package_id,
            from_place=PackagePlace(
                kind="arm",
                drone_id=from_drone,
                arm_id=from_arm,
            ),
            to_place=PackagePlace(
                kind="transp",
                transporter_id=cmd.transporter_id,
            ),
            person_id=None,
        )

    if isinstance(cmd, UnloadFromTransporter):
        # Origen: paquete en transp. Destino: brazo del drone.
        pkg_a = snap_a.world.packages[cmd.package_id]
        if not isinstance(pkg_a.at, InTransporter):
            raise ValueError(
                f"UnloadFromTransporter inconsistente: el paquete "
                f"{cmd.package_id!r} debería estar InTransporter en snap_a, "
                f"pero está en {type(pkg_a.at).__name__}."
            )
        from_transp = pkg_a.at.transporter_id
        return TransitionPackageMove(
            drone_id=cmd.drone_id,
            package_id=cmd.package_id,
            from_place=PackagePlace(
                kind="transp",
                transporter_id=from_transp,
            ),
            to_place=PackagePlace(
                kind="arm",
                drone_id=cmd.drone_id,
                arm_id=cmd.arm_id,
            ),
            person_id=None,
        )

    # Caso 6: Command desconocido. Fail-fast.
    raise TypeError(
        f"classify_transition no sabe clasificar el Command de tipo "
        f"{type(cmd).__name__}. Si se ha añadido un Command nuevo en "
        f"commands/, actualizar interpolation.py para cubrirlo."
    )


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------


def _drone_position_in(snap: WorldSnapshot, drone_id: str) -> str:
    """Lee la posición (loc_id) del drone en el snapshot dado.

    Encapsula el lookup para que cualquier futuro cambio en cómo se
    accede a la posición del drone se localice aquí.

    Raises:
        KeyError: si el drone no existe en snap.world.
    """
    return snap.world.drones[drone_id].position


def _atlocation_loc_id_or_raise(
    pkg,
    pkg_id: str,
    snap_label: str,
    cmd_label: str,
) -> str:
    """Si pkg.at es AtLocation, devuelve su loc_id; si no, lanza ValueError.

    Args:
        pkg: Package del dominio.
        pkg_id: id del paquete (para el mensaje de error).
        snap_label: "snap_a" o "snap_b" (para el mensaje de error).
        cmd_label: nombre del Command que esperaba esta condición.

    Raises:
        ValueError: si pkg.at no es AtLocation. Indica inconsistencia
            entre el Command y el snapshot, normalmente un bug en
            el runner o un Plan mal construido.
    """
    if not isinstance(pkg.at, AtLocation):
        raise ValueError(
            f"{cmd_label} inconsistente: el paquete {pkg_id!r} debería "
            f"estar AtLocation en {snap_label}, pero está en "
            f"{type(pkg.at).__name__}."
        )
    return pkg.at.loc_id
