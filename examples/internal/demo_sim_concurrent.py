"""Demo de simulación: DOS drones actuando CONCURRENTEMENTE.

d1 y d2 parten del depósito y entregan a la vez, cada uno a su casa, sobre
recursos disjuntos (drones y paquetes distintos): el runner lo permite y el
makespan refleja el PARALELISMO (máximo de las dos pistas), no la suma.

Controles: ESPACIO pausa · ← → snapshots · R reinicia · ESC/Q sale.

Uso (ventana):           python scripts/demo_sim_concurrent.py
Uso (humo headless):     SDL_VIDEODRIVER=dummy python scripts/demo_sim_concurrent.py --smoke
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from droneplan_viz.commands import Deliver, Move, PickUp  # noqa: E402
from droneplan_viz.domain import (  # noqa: E402
    Arm, AtLocation, Content, Drone, DroneState, Location, Package, Person, World,
)
from droneplan_viz.runtime import Plan, ScheduledCommand  # noqa: E402
from examples.internal._demo_common import run_replay  # noqa: E402


def build_world() -> World:
    med = Content(id="medicina")
    com = Content(id="comida")
    contents = {"medicina": med, "comida": com}
    locs = {
        "deposito": Location(id="deposito"),
        "casa1": Location(id="casa1"),
        "casa2": Location(id="casa2"),
    }
    drones = {
        "d1": Drone(id="d1", position="deposito",
                    arms=(Arm(id="izq"), Arm(id="der")), state=DroneState.IDLE),
        "d2": Drone(id="d2", position="deposito",
                    arms=(Arm(id="izq"), Arm(id="der")), state=DroneState.IDLE),
    }
    packages = {
        "pkg_med": Package(id="pkg_med", contains=med, at=AtLocation(loc_id="deposito")),
        "pkg_com": Package(id="pkg_com", contains=com, at=AtLocation(loc_id="deposito")),
    }
    persons = {
        "ana": Person(id="ana", position="casa1", needs=(med,)),
        "bob": Person(id="bob", position="casa2", needs=(com,)),
    }
    pairs = [("deposito", "casa1", 8.0), ("deposito", "casa2", 10.0),
             ("casa1", "casa2", 6.0)]
    costs: dict[tuple[str, str], float] = {}
    for o, d, c in pairs:
        costs[(o, d)] = c
        costs[(d, o)] = c
    return World(locations=locs, drones=drones, packages=packages,
                 persons=persons, contents=contents, costs=costs)


def build_plan() -> Plan:
    def S(cmd, t):
        return ScheduledCommand(command=cmd, start_time=t)

    return Plan(scheduled=(
        # Pista de d1: medicina a ana (casa1). Arranca en t=0.
        S(PickUp(drone_id="d1", arm_id="izq", package_id="pkg_med",
                 duration=2.0, command_id="d1_pk"), 0.0),
        S(Move(drone_id="d1", destination_id="casa1",
               duration=6.0, command_id="d1_mv"), 2.0),
        S(Deliver(drone_id="d1", package_id="pkg_med", person_id="ana",
                  duration=2.0, command_id="d1_dl"), 8.0),
        # Pista de d2: comida a bob (casa2). Arranca TAMBIÉN en t=0 (paralelo).
        S(PickUp(drone_id="d2", arm_id="izq", package_id="pkg_com",
                 duration=2.0, command_id="d2_pk"), 0.0),
        S(Move(drone_id="d2", destination_id="casa2",
               duration=8.0, command_id="d2_mv"), 2.0),
        S(Deliver(drone_id="d2", package_id="pkg_com", person_id="bob",
                  duration=2.0, command_id="d2_dl"), 10.0),
    ))


def main() -> int:
    return run_replay(
        build_world(), build_plan(),
        caption="droneplan_viz · demo_sim_concurrent (2 drones en paralelo)",
        smoke="--smoke" in sys.argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
