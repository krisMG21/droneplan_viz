"""Validator: reglas físicas del dominio, fuente única de verdad.

Decisión 2 del diseño: una sola pieza decide si un comando es aplicable
al estado actual del mundo. Esto evita reglas dispersas entre handlers
de comandos, builders y renderer.

Organización:

    ValidationResult: resultado tipado y consultable como booleano.
    _check_drone_ready: helper para las tres reglas transversales que
        aplican a casi cualquier comando dirigido a un drone.
    validate_*: una función pública por tipo de acción del dominio.

Por qué funciones libres y no una clase Validator:

    Las reglas son puras (entrada → resultado), no tienen estado.
    Una clase sin atributos sería ruido. Si en el futuro hace falta
    configurabilidad (activar/desactivar reglas, niveles de severidad),
    se introduce entonces. YAGNI ahora.

Por qué cada función devuelve al primer fallo:

    El usuario solo necesita una razón clara: "no puedo mover porque el
    drone está en error". Listar todas las razones simultáneamente sería
    más información de la útil y complicaría los mensajes en la UI.

Las funciones validate_* serán llamadas desde el dispatch del Validator
en el runtime/C, una vez que existan los tipos Command. Por ahora se usan
directamente con sus argumentos sueltos, lo que permite tests completos
sin acoplar a las clases Command que aún no existen.
"""

from dataclasses import dataclass

from droneplan_viz.domain.package import AtLocation, HeldByArm, InTransporter
from droneplan_viz.domain.world import World


# ---------------------------------------------------------------------------
# ValidationResult
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Resultado de validar un comando contra un World.

    Atributos:
        ok: True si el comando es aplicable, False si no.
        reason: mensaje legible de la razón del fallo. None cuando ok=True.

    Soporta uso como booleano: `if validate_move(world, ...):` lee bien.
    """

    ok: bool
    reason: str | None = None

    @classmethod
    def valid(cls) -> "ValidationResult":
        """Resultado afirmativo, sin razón."""
        return cls(ok=True, reason=None)

    @classmethod
    def invalid(cls, reason: str) -> "ValidationResult":
        """Resultado negativo con razón obligatoria."""
        return cls(ok=False, reason=reason)

    def __bool__(self) -> bool:
        return self.ok


# ---------------------------------------------------------------------------
# Helpers privados: reglas transversales
# ---------------------------------------------------------------------------


def _check_drone_ready(world: World, drone_id: str) -> str | None:
    """Comprueba las tres reglas transversales a cualquier comando
    dirigido a un drone:

        1. El drone existe.
        2. El drone no está en ERROR (sumidero forward).
        3. El drone está IDLE (no MOVING ni INTERACTING).

    Devuelve la razón del primer fallo, o None si las tres pasan.
    Aislamos esto en un helper porque casi todas las acciones lo usan
    como primer chequeo y conviene mantener el mensaje uniforme.
    """
    if drone_id not in world.drones:
        return f"el drone '{drone_id}' no existe en el mundo"
    drone = world.drones[drone_id]
    if drone.state.is_terminal_forward():
        return f"el drone '{drone_id}' está en estado terminal (ERROR)"
    if drone.state.is_busy():
        return (
            f"el drone '{drone_id}' está ocupado "
            f"({drone.state.name}); no acepta comandos"
        )
    return None


def _check_move_target(
    world: World, from_loc_id: str, to_loc_id: str
) -> str | None:
    """Comprueba las dos reglas asociadas al destino de un movimiento:

        4. La localización destino existe.
        5. Existe coste declarado entre origen y destino (adyacencia
           implícita por declaración).

    Helper compartido entre validate_move y validate_move_with_transporter,
    cuyas reglas de "a dónde se puede ir" son idénticas.
    """
    if to_loc_id not in world.locations:
        return (
            f"la localización destino '{to_loc_id}' no existe en el mundo"
        )
    if (from_loc_id, to_loc_id) not in world.costs:
        return (
            f"no hay coste declarado entre '{from_loc_id}' y "
            f"'{to_loc_id}'; la arista no es transitable"
        )
    return None


# ---------------------------------------------------------------------------
# validate_move: acción PDDL "volar"
# ---------------------------------------------------------------------------


def validate_move(
    world: World, drone_id: str, to_loc_id: str
) -> ValidationResult:
    """Valida un comando de movimiento de un drone a una localización.

    Reglas comprobadas, en orden:

        1–3. Drone existe, no en ERROR, IDLE (vía _check_drone_ready).
        4. La localización destino existe en el mundo.
        5. Existe coste declarado entre la posición actual del drone y
           el destino. Modelamos adyacencia implícita por la existencia
           del coste: si no se declaró en world.costs, asumimos que la
           arista no es transitable.

    No se permite "moverse al mismo sitio": si el alumno declara un
    fly-cost (X, X) = 1 como hace el generador de la asignatura, será
    una arista transitable, pero típicamente eso es plumbing del PDDL y
    no aparecerá en planes reales. No emitimos juicio aquí.
    """
    razon = _check_drone_ready(world, drone_id)
    if razon is not None:
        return ValidationResult.invalid(razon)

    drone = world.drones[drone_id]
    razon = _check_move_target(world, drone.position, to_loc_id)
    if razon is not None:
        return ValidationResult.invalid(razon)

    return ValidationResult.valid()


# ---------------------------------------------------------------------------
# validate_move_with_transporter: acción PDDL "mover-transportador"
# ---------------------------------------------------------------------------


def validate_move_with_transporter(
    world: World,
    drone_id: str,
    to_loc_id: str,
    transporter_id: str,
) -> ValidationResult:
    """Valida un comando de movimiento de un drone arrastrando un
    transportador hacia una localización.

    Reglas comprobadas, en orden:

        1–3. Drone existe, no en ERROR, IDLE.
        4–5. Destino existe, arista transitable (compartidas con
             validate_move vía _check_move_target).
        6. El transportador existe en el mundo.
        7. El transportador está co-localizado con el drone. En el PDDL,
           `(en-transportador ?t ?u1)` con el mismo `?u1` que la posición
           del drone es precondition de la acción.
        8. El drone tiene al menos un brazo vacío. Refleja la
           precondition `(brazo-vacio ?d ?b)` del PDDL canónico. Un
           drone explorador (sin brazos) no podrá arrastrar
           transportadores: estructuralmente "ningún brazo libre"
           equivale a "no hay brazo con el que guiar el transportador".

    Por qué función separada en vez de un parámetro opcional en
    validate_move:

        En el PDDL son dos acciones distintas (volar vs
        mover-transportador) con efectos distintos. Mantenerlas
        separadas refleja eso, deja la firma de cada función limpia,
        y prepara el dispatch por tipo de Command.
    """
    razon = _check_drone_ready(world, drone_id)
    if razon is not None:
        return ValidationResult.invalid(razon)

    drone = world.drones[drone_id]
    razon = _check_move_target(world, drone.position, to_loc_id)
    if razon is not None:
        return ValidationResult.invalid(razon)

    if transporter_id not in world.transporters:
        return ValidationResult.invalid(
            f"el transportador '{transporter_id}' no existe en el mundo"
        )

    transporter = world.transporters[transporter_id]
    if transporter.position != drone.position:
        return ValidationResult.invalid(
            f"el transportador '{transporter_id}' está en "
            f"'{transporter.position}' y el drone '{drone_id}' está en "
            f"'{drone.position}'; no están co-localizados"
        )

    # Al menos un brazo vacío. Un brazo está vacío si no hay paquete
    # registrado en world como HeldByArm con ese (drone_id, arm.id).
    if not any(
        world.package_held_by(drone_id, arm.id) is None
        for arm in drone.arms
    ):
        return ValidationResult.invalid(
            f"el drone '{drone_id}' no tiene ningún brazo libre para "
            f"guiar el transportador"
        )

    return ValidationResult.valid()


# ---------------------------------------------------------------------------
# validate_pick_up: acción PDDL "recoger"
# ---------------------------------------------------------------------------


def validate_pick_up(
    world: World, drone_id: str, package_id: str, arm_id: str
) -> ValidationResult:
    """Valida un comando de recogida: el drone agarra un paquete con
    un brazo concreto.

    Reglas comprobadas, en orden:

        1–3. Drone existe, no en ERROR, IDLE (vía _check_drone_ready).
        9. El paquete existe en el mundo.
        10. El brazo identificado existe en el drone. arm_id se busca
            por igualdad estructural en la tupla drone.arms.
        11. El brazo está libre. Refleja la precondition PDDL
            `(brazo-vacio ?d ?b)`. Se comprueba consultando world:
            si algún paquete tiene HeldByArm(drone_id, arm_id), el
            brazo está ocupado.
        12. El paquete está libre y co-localizado con el drone. Refleja
            la precondition PDDL `(en-caja ?c ?u)` con `?u =
            drone.position`. Implica dos cosas a la vez: el paquete
            está en estado AtLocation (no sostenido ni en transporter)
            y su loc_id coincide con el del drone.

    El orden refleja fundamentalidad: existencia de las cosas, después
    estado de las cosas.
    """
    razon = _check_drone_ready(world, drone_id)
    if razon is not None:
        return ValidationResult.invalid(razon)

    if package_id not in world.packages:
        return ValidationResult.invalid(
            f"el paquete '{package_id}' no existe en el mundo"
        )

    drone = world.drones[drone_id]
    if not any(arm.id == arm_id for arm in drone.arms):
        return ValidationResult.invalid(
            f"el drone '{drone_id}' no tiene un brazo llamado '{arm_id}'"
        )

    ocupando = world.package_held_by(drone_id, arm_id)
    if ocupando is not None:
        return ValidationResult.invalid(
            f"el brazo '{arm_id}' del drone '{drone_id}' ya sostiene "
            f"el paquete '{ocupando.id}'"
        )

    package = world.packages[package_id]
    if not isinstance(package.at, AtLocation):
        return ValidationResult.invalid(
            f"el paquete '{package_id}' no está libre en ninguna "
            f"localización; no se puede recoger"
        )
    if package.at.loc_id != drone.position:
        return ValidationResult.invalid(
            f"el paquete '{package_id}' está en '{package.at.loc_id}' "
            f"y el drone '{drone_id}' está en '{drone.position}'; no "
            f"están co-localizados"
        )

    return ValidationResult.valid()


# ---------------------------------------------------------------------------
# validate_deliver: acción PDDL "entregar"
# ---------------------------------------------------------------------------


def validate_deliver(
    world: World, drone_id: str, package_id: str, person_id: str
) -> ValidationResult:
    """Valida un comando de entrega: el drone entrega un paquete (que
    sostiene en algún brazo) a una persona co-localizada.

    Reglas comprobadas, en orden:

        1–3. Drone existe, no en ERROR, IDLE (vía _check_drone_ready).
        13. La persona existe en el mundo.
        14. El paquete existe en el mundo.
        15. El paquete está sostenido por algún brazo de este drone.
            Refleja la precondition PDDL `(sujetando ?d ?b ?c)`. No se
            comprueba qué brazo concreto: la API de la facade no lo
            exige porque el efecto de la entrega libera el brazo que sea.
        16. La persona está co-localizada con el drone. Refleja
            `(en-persona ?p ?u)` con `?u = drone.position`.
        17. La persona necesita el contenido del paquete. Refleja
            `(necesita ?p ?co)` como precondition. Esta regla es la
            traducción más estricta del dominio: el alumno no puede
            entregar comida a alguien que no la necesita; sería un
            comando que el planificador real nunca habría producido.

    Orden de fundamentalidad: existencias antes que estados, identidad
    antes que contenido semántico.
    """
    razon = _check_drone_ready(world, drone_id)
    if razon is not None:
        return ValidationResult.invalid(razon)

    if person_id not in world.persons:
        return ValidationResult.invalid(
            f"la persona '{person_id}' no existe en el mundo"
        )

    if package_id not in world.packages:
        return ValidationResult.invalid(
            f"el paquete '{package_id}' no existe en el mundo"
        )

    package = world.packages[package_id]
    if not (
        isinstance(package.at, HeldByArm) and package.at.drone_id == drone_id
    ):
        return ValidationResult.invalid(
            f"el paquete '{package_id}' no está siendo sostenido por "
            f"el drone '{drone_id}'"
        )

    drone = world.drones[drone_id]
    person = world.persons[person_id]
    if person.position != drone.position:
        return ValidationResult.invalid(
            f"la persona '{person_id}' está en '{person.position}' y "
            f"el drone '{drone_id}' está en '{drone.position}'; no "
            f"están co-localizados"
        )

    if package.contains not in person.needs:
        return ValidationResult.invalid(
            f"la persona '{person_id}' no necesita "
            f"'{package.contains.id}'; el paquete '{package_id}' no "
            f"se puede entregar"
        )

    return ValidationResult.valid()


# ---------------------------------------------------------------------------
# validate_load_into_transporter: acción PDDL "poner-caja-en-transportador"
# ---------------------------------------------------------------------------


def validate_load_into_transporter(
    world: World,
    drone_id: str,
    package_id: str,
    transporter_id: str,
) -> ValidationResult:
    """Valida un comando de carga: el drone deposita un paquete que
    sostiene en un transportador co-localizado.

    Reglas comprobadas, en orden:

        1–3. Drone existe, no en ERROR, IDLE (vía _check_drone_ready).
        18. El paquete existe en el mundo.
        19a. El transportador existe en el mundo.
        19b. El transportador está co-localizado con el drone. Refleja
             la precondition `(en-transportador ?t ?u)` con `?u` igual
             a la posición del drone.
        20. El paquete está sostenido por algún brazo de este drone.
            Refleja `(sujetando ?d ?b ?c)`. Misma comprobación que en
            validate_deliver: no se especifica brazo, se libera el que
            sea al cargar.
        21. El transportador no está lleno. Refleja el mecanismo
            `(capacidad-disponible ?t ?n) (siguiente ?n0 ?n)` del PDDL:
            si la cuenta de paquetes ya en el transporter iguala su
            capacidad, no hay sucesor numérico disponible y la acción
            no aplica. Lo expresamos directamente como comparación
            entera, equivalente y más legible.

    Orden de fundamentalidad: drone → paquete → transportador
    (existencia, después posición) → estado del paquete → capacidad.
    """
    razon = _check_drone_ready(world, drone_id)
    if razon is not None:
        return ValidationResult.invalid(razon)

    if package_id not in world.packages:
        return ValidationResult.invalid(
            f"el paquete '{package_id}' no existe en el mundo"
        )

    if transporter_id not in world.transporters:
        return ValidationResult.invalid(
            f"el transportador '{transporter_id}' no existe en el mundo"
        )

    drone = world.drones[drone_id]
    transporter = world.transporters[transporter_id]
    if transporter.position != drone.position:
        return ValidationResult.invalid(
            f"el transportador '{transporter_id}' está en "
            f"'{transporter.position}' y el drone '{drone_id}' está en "
            f"'{drone.position}'; no están co-localizados"
        )

    package = world.packages[package_id]
    if not (
        isinstance(package.at, HeldByArm) and package.at.drone_id == drone_id
    ):
        return ValidationResult.invalid(
            f"el paquete '{package_id}' no está siendo sostenido por "
            f"el drone '{drone_id}'"
        )

    cargados = len(world.packages_in_transporter(transporter_id))
    if cargados >= transporter.capacity:
        return ValidationResult.invalid(
            f"el transportador '{transporter_id}' está lleno "
            f"({cargados}/{transporter.capacity}); no se puede cargar "
            f"el paquete '{package_id}'"
        )

    return ValidationResult.valid()


# ---------------------------------------------------------------------------
# validate_unload_from_transporter:
#     acción PDDL "coger-caja-del-transportador"
# ---------------------------------------------------------------------------


def validate_unload_from_transporter(
    world: World,
    drone_id: str,
    package_id: str,
    transporter_id: str,
    arm_id: str,
) -> ValidationResult:
    """Valida un comando de descarga: el drone saca un paquete de un
    transportador co-localizado, agarrándolo con un brazo concreto.

    Reglas comprobadas, en orden:

        1–3. Drone existe, no en ERROR, IDLE (vía _check_drone_ready).
        22. El paquete existe en el mundo.
        23a. El transportador existe en el mundo.
        23b. El transportador está co-localizado con el drone. Refleja
             la precondition `(en-transportador ?t ?u)` con `?u` igual
             a la posición del drone.
        24. El paquete está dentro de ese transportador. Refleja la
            precondition `(en-transportador-caja ?t ?c)`.
        25a. El brazo identificado existe en el drone. Mismo chequeo
             que en validate_pick_up.
        25b. El brazo está libre. Refleja `(brazo-vacio ?d ?b)`. El
             efecto de la descarga será `(sujetando ?d ?b ?c)`.

    A diferencia del comando "entregar" (donde no especificamos brazo
    porque el efecto libera "el brazo que sea"), aquí el efecto es
    ocupar un brazo concreto, así que el arm_id es obligatorio. Esto
    mantiene paralelismo con "recoger".

    Orden de fundamentalidad: drone → paquete → transporter (existencia,
    posición) → relación paquete-transporter → brazo (existencia, libre).
    """
    razon = _check_drone_ready(world, drone_id)
    if razon is not None:
        return ValidationResult.invalid(razon)

    if package_id not in world.packages:
        return ValidationResult.invalid(
            f"el paquete '{package_id}' no existe en el mundo"
        )

    if transporter_id not in world.transporters:
        return ValidationResult.invalid(
            f"el transportador '{transporter_id}' no existe en el mundo"
        )

    drone = world.drones[drone_id]
    transporter = world.transporters[transporter_id]
    if transporter.position != drone.position:
        return ValidationResult.invalid(
            f"el transportador '{transporter_id}' está en "
            f"'{transporter.position}' y el drone '{drone_id}' está en "
            f"'{drone.position}'; no están co-localizados"
        )

    package = world.packages[package_id]
    if not (
        isinstance(package.at, InTransporter)
        and package.at.transporter_id == transporter_id
    ):
        return ValidationResult.invalid(
            f"el paquete '{package_id}' no está en el transportador "
            f"'{transporter_id}'"
        )

    if not any(arm.id == arm_id for arm in drone.arms):
        return ValidationResult.invalid(
            f"el drone '{drone_id}' no tiene un brazo llamado '{arm_id}'"
        )

    ocupando = world.package_held_by(drone_id, arm_id)
    if ocupando is not None:
        return ValidationResult.invalid(
            f"el brazo '{arm_id}' del drone '{drone_id}' ya sostiene "
            f"el paquete '{ocupando.id}'"
        )

    return ValidationResult.valid()
