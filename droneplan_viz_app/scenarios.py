"""Factorías de escenarios (World, Plan) para la app interactiva.

Desde el runtime, el escenario de ÉXITO (build_demo_scenario) se construye a
través de la fachada pública DronePlanViz: es, además de la demo de la app,
el primer cliente real de la fachada y un test vivo de que reproduce un
World/Plan idéntico al que se montaría a mano. El escenario de FALLO
(build_failure_demo_scenario) se mantiene montado a mano sobre domain/
commands/runtime como ANTES: actúa de regresión del camino de construcción
directa (que sigue siendo API legítima) y porque su plan mezcla solapes de
tiempo y un fallo deliberado que es más claro de leer explícitamente.

Diseño consciente — DUPLICACIÓN frente a extracción de scripts/demo_render.py:

Existe la tentación de mover build_demo_world() / build_demo_plan() desde
scripts/demo_render.py a este módulo y hacer que el script importe desde
aquí. NO lo hago: el script demo_render.py es material de validación de
el runtime y ha de poder ejecutarse aunque droneplan_viz_app no esté
instalado (la librería es pygame-only; el script no necesita pygame_gui).
Hacer que el script dependa del paquete app introduciría un acoplamiento
inverso que rompe la frontera limpia. Pago el coste de duplicar ~80
líneas de construcción de World y Plan a cambio de mantener la frontera.

Si el escenario cambia en el futuro (nuevos contenidos, más drones), se
actualizan AMBOS sitios y el test test_demo_world_topologia documenta el
contrato. El test test_demo_world_igual_a_construccion_manual aserta además
que la versión por fachada y la manual (_build_demo_world) coinciden
estructuralmente, así que una divergencia accidental se detecta.

Las dos factorías:

- build_demo_scenario() -> (World, Plan)
    Escenario "feliz", construido VÍA FACHADA: 4 locations, 2 drones, 1
    transporter, 3 paquetes, 2 personas. Plan durativo de 3 Commands
    (PickUp → Move → Deliver) que se ejecuta sin fallos. Equivalente al de
    scripts/demo_render.py y a _build_demo_world().

- build_failure_demo_scenario() -> (World, Plan)
    Escenario "didáctico", montado A MANO: misma topología pero el Plan
    combina Commands exitosos con uno que falla. Concretamente, d2 (en
    casa1) intenta recoger pkg_water1 (en deposito): fallo PDDL "no
    co-localizados". Esto deja a d2 en ERROR pero d1 continúa exitosamente.
    Útil para que la UI muestre simultáneamente:
      * la X roja sobre d2 (cortesía del render),
      * un item en el panel de fallos,
      * la marca roja en la barra de progreso,
      * un Move/Deliver exitoso después (la ejecución no se cancela
        globalmente; documentado en runtime/runner.py).
"""
from __future__ import annotations

from droneplan_viz.commands import Deliver, Move, PickUp
from droneplan_viz.domain import (
    AtLocation,
    Content,
    Package,
    Transporter,
    World,
)
from droneplan_viz.domain.arm import Arm
from droneplan_viz.domain.drone import Drone
from droneplan_viz.domain.drone_state import DroneState
from droneplan_viz.domain.location import Location
from droneplan_viz.domain.person import Person
from droneplan_viz.runtime import Plan, ScheduledCommand

# La fachada pública: build_demo_scenario se construye con ella (es su
# primer cliente real). El import es seguro (no circular): droneplan_viz no
# importa droneplan_viz_app, y el run() de la fachada importa la app de
# forma lazy.
from droneplan_viz import DronePlanViz


# ---------------------------------------------------------------------------
# World docente compartido
# ---------------------------------------------------------------------------


def _build_demo_world() -> World:
    """World docente con todas las clases de entidad presentes.

    Topología (idéntica a scripts/demo_render.py por consistencia visual):
        Locations (4): casa1, casa2, deposito, hospital.
        Drones (2): d1 (2 brazos, en deposito), d2 (1 brazo, en casa1).
        Transporter (1): t1 en deposito, capacidad 4.
        Persons (2): p1 en casa1 (necesita medicina + comida),
                     p2 en casa2 (necesita agua).
        Packages (3):
            pkg_med1 (medicina) en deposito,
            pkg_food1 (comida) en deposito,
            pkg_water1 (agua) en deposito.
        Costs: grafo completo entre las 4 locs.
    """
    contents = {
        "medicina": Content(id="medicina"),
        "comida": Content(id="comida"),
        "agua": Content(id="agua"),
    }
    locs = {
        "casa1": Location(id="casa1"),
        "casa2": Location(id="casa2"),
        "deposito": Location(id="deposito"),
        "hospital": Location(id="hospital"),
    }
    drones = {
        "d1": Drone(
            id="d1",
            position="deposito",
            arms=(Arm(id="izq"), Arm(id="der")),
            state=DroneState.IDLE,
        ),
        "d2": Drone(
            id="d2",
            position="casa1",
            arms=(Arm(id="izq"),),
            state=DroneState.IDLE,
        ),
    }
    transporters = {
        "t1": Transporter(id="t1", position="deposito", capacity=4),
    }
    persons = {
        "p1": Person(
            id="p1",
            position="casa1",
            needs=(contents["medicina"], contents["comida"]),
        ),
        "p2": Person(
            id="p2",
            position="casa2",
            needs=(contents["agua"],),
        ),
    }
    packages = {
        "pkg_med1": Package(
            id="pkg_med1",
            contains=contents["medicina"],
            at=AtLocation(loc_id="deposito"),
        ),
        "pkg_food1": Package(
            id="pkg_food1",
            contains=contents["comida"],
            at=AtLocation(loc_id="deposito"),
        ),
        "pkg_water1": Package(
            id="pkg_water1",
            contains=contents["agua"],
            at=AtLocation(loc_id="deposito"),
        ),
    }
    pairs = [
        ("casa1", "casa2", 5.0),
        ("casa1", "deposito", 8.0),
        ("casa1", "hospital", 10.0),
        ("casa2", "deposito", 12.0),
        ("casa2", "hospital", 9.0),
        ("deposito", "hospital", 6.0),
    ]
    costs: dict[tuple[str, str], float] = {}
    for o, d, c in pairs:
        costs[(o, d)] = c
        costs[(d, o)] = c

    return World(
        locations=locs,
        drones=drones,
        transporters=transporters,
        persons=persons,
        packages=packages,
        contents=contents,
        costs=costs,
    )


# ---------------------------------------------------------------------------
# Escenario de éxito
# ---------------------------------------------------------------------------


def build_demo_scenario() -> tuple[World, Plan]:
    """Escenario de éxito puro: d1 entrega medicina a p1. VÍA FACHADA.

        t=0   PickUp d1·izq pkg_med1 en deposito  (5s)
        t=5   Move d1 deposito → casa1            (8s)
        t=13  Deliver pkg_med1 a p1               (5s)
        t=18  fin

    Tras este plan, p1 recibió medicina y aún necesita comida; pkg_med1
    aparece en casa1. PlanRunner.execute() devuelve succeeded=True,
    failures=().

    Construido con DronePlanViz para reproducir EXACTAMENTE el World y el
    Plan de _build_demo_world() + la demo de scripts/demo_render.py:

      - Topología completa declarada (las 4 locations, los 3 contenidos,
        las 2 personas, los 3 paquetes, d1/d2/t1), aunque el plan feliz solo
        ejerza d1·pkg_med1·p1: el contrato de topología documentado lo exige
        y la app la dibuja entera.
      - Grafo de costes completo y simétrico entre las 4 locations. No es
        decorativo: la arista deposito↔casa1 es la que hace transitable el
        Move; el resto mantiene paridad con la demo.
      - command_id explícitos (step_pickup/step_move/step_deliver) vía id=,
        para conservar las etiquetas originales del plan.
      - inicio= explícito en las tres acciones: plan TEMPORAL con tiempos
        absolutos 0/5/13, idénticos a la demo durativa original.
    """
    viz = DronePlanViz()

    # Localizaciones (grafo completo, simétrico).
    for loc in ("casa1", "casa2", "deposito", "hospital"):
        viz.world.location(loc)
    viz.world.costes(
        {
            ("casa1", "casa2"): 5.0,
            ("casa1", "deposito"): 8.0,
            ("casa1", "hospital"): 10.0,
            ("casa2", "deposito"): 12.0,
            ("casa2", "hospital"): 9.0,
            ("deposito", "hospital"): 6.0,
        },
        simetrico=True,
    )

    # Contenidos.
    for c in ("medicina", "comida", "agua"):
        viz.world.content(c)

    # Personas y sus necesidades.
    viz.world.person("p1", at="casa1", necesita=["medicina", "comida"])
    viz.world.person("p2", at="casa2", necesita=["agua"])

    # Paquetes (los tres depositados en deposito).
    viz.world.package("pkg_med1", contiene="medicina", at="deposito")
    viz.world.package("pkg_food1", contiene="comida", at="deposito")
    viz.world.package("pkg_water1", contiene="agua", at="deposito")

    # Agentes: d1 dos brazos en deposito, d2 un brazo en casa1, t1 cap. 4.
    viz.agents.drone("d1", at="deposito", arms=["izq", "der"])
    viz.agents.drone("d2", at="casa1", arms=["izq"])
    viz.agents.transporter("t1", capacidad=4, at="deposito")

    # Plan durativo (PickUp → Move → Deliver), ids y tiempos originales.
    viz.recoger(
        "d1", caja="pkg_med1", brazo="izq",
        inicio=0.0, duracion=5.0, id="step_pickup",
    )
    viz.mover(
        "d1", a="casa1",
        inicio=5.0, duracion=8.0, id="step_move",
    )
    viz.entregar(
        "d1", caja="pkg_med1", a="p1",
        inicio=13.0, duracion=5.0, id="step_deliver",
    )

    return viz.build()


# ---------------------------------------------------------------------------
# Escenario de fallo
# ---------------------------------------------------------------------------


def build_failure_demo_scenario() -> tuple[World, Plan]:
    """Escenario didáctico mixto: un éxito, un fallo PDDL, otro éxito.

    Topología idéntica a la del escenario feliz. El Plan combina:

        t=0   PickUp d1·izq pkg_med1 en deposito  (5s, ÉXITO)
              → d1 ahora sostiene pkg_med1.

        t=2   PickUp d2·izq pkg_water1            (5s, FALLO)
              → d2 está en casa1, pkg_water1 en deposito.
              → fallo PDDL: "no co-localizados".
              → d2 pasa a ERROR. failed_commands += 1.

        t=5   Move d1 deposito → casa1            (8s, ÉXITO)
              → continúa en paralelo, d1 no afectado por el fallo de d2.

        t=13  Deliver pkg_med1 a p1 con d1        (5s, ÉXITO)
              → p1 recibe medicina; pkg_med1 acaba en casa1.

    Resultado: RunResult.succeeded=False, len(failures)=1, makespan=18s.
    El historial contiene snapshots intercalados (los Commands a t=0 y
    t=2 producen snapshots con start_times distintos pese al solape) que
    Timeline.sample() recorre limpiamente.

    Por qué este caso (PDDL no-colocalizado) en concreto:
      * El fallo es semánticamente diáfano para mostrar en el HUD:
        "pkg_water1 está en 'deposito', d2 está en 'casa1'; no están
        co-localizados". Frase autoexplicativa para el TFG.
      * Es 100% reproducible: no depende de timing, no es race condition.
      * No interfiere con el resto del plan (d1 sigue su curso), así que
        el usuario ve ambas cosas a la vez.

    Alternativas que rechacé:
      * Fallo de concurrencia: requiere construir un caso con la
        ResourceTable que es más opaco para mostrar en el HUD.
      * Drone inexistente: el mensaje sería "drone 'xxx' no existe",
        menos didáctico que un fallo de co-localización.
      * Plan que solo falla: pierde la mitad de la lección visual (no
        habría animación exitosa que ver).
    """
    world = _build_demo_world()
    plan = Plan(scheduled=(
        # ÉXITO inicial: d1 recoge pkg_med1.
        ScheduledCommand(
            command=PickUp(
                drone_id="d1",
                arm_id="izq",
                package_id="pkg_med1",
                duration=5.0,
                command_id="ok_pickup_med",
            ),
            start_time=0.0,
        ),
        # FALLO: d2 intenta recoger pkg_water1 sin co-localización.
        ScheduledCommand(
            command=PickUp(
                drone_id="d2",
                arm_id="izq",
                package_id="pkg_water1",
                duration=5.0,
                command_id="fail_pickup_water",
            ),
            start_time=2.0,
        ),
        # ÉXITO: d1 vuela a casa1.
        ScheduledCommand(
            command=Move(
                drone_id="d1",
                destination_id="casa1",
                duration=8.0,
                command_id="ok_move_casa1",
            ),
            start_time=5.0,
        ),
        # ÉXITO: d1 entrega pkg_med1 a p1.
        ScheduledCommand(
            command=Deliver(
                drone_id="d1",
                package_id="pkg_med1",
                person_id="p1",
                duration=5.0,
                command_id="ok_deliver_med",
            ),
            start_time=13.0,
        ),
    ))
    return world, plan


# ---------------------------------------------------------------------------
# Registro de escenarios disponibles por nombre (consumido por la CLI)
# ---------------------------------------------------------------------------

#: Mapeo nombre → factoría. Lo consumirá main.py para resolver --scenario.
#: Mantener al final del módulo para que las funciones de arriba estén
#: definidas cuando este diccionario se construye.
SCENARIOS: dict[str, "ScenarioFactory"] = {
    "demo": build_demo_scenario,
    "failure_demo": build_failure_demo_scenario,
}


# Type alias informal para la firma de las factorías. No es un Protocol
# formal porque solo hay un consumidor (la CLI) y aún no necesitamos la
# extensibilidad; YAGNI estricto.
from typing import Callable  # noqa: E402

ScenarioFactory = Callable[[], tuple[World, Plan]]
