"""Demo de simulación: un PLAN con un FALLO.

d1 entrega correctamente (medicina a ana en casa1). d2, en cambio, intenta
recoger un paquete que NO está co-localizado con él: pkg_lejos está en
casa2 pero d2 está en el depósito. El Validador rechaza la acción (fallo
PDDL), d2 pasa a estado ERROR y el render lo marca con la X roja; el resto
del plan (d1) sigue su curso. La ejecución global devuelve succeeded=False
con un CommandFailure.

Controles: ESPACIO pausa · ← → snapshots · R reinicia · ESC/Q sale.

Uso (ventana):           python scripts/demo_sim_failure.py
Uso (humo headless):     SDL_VIDEODRIVER=dummy python scripts/demo_sim_failure.py --smoke
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
    contents = {"medicina": med}
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
        # Paquete que d2 intentará recoger desde OTRA loc: está en casa1.
        "pkg_lejos": Package(id="pkg_lejos", contains=med, at=AtLocation(loc_id="casa1")),
    }
    persons = {
        "ana": Person(id="ana", position="casa1", needs=(med,)),
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
        # d1: entrega correcta de medicina a ana.
        S(PickUp(drone_id="d1", arm_id="izq", package_id="pkg_med",
                 duration=2.0, command_id="d1_pk"), 0.0),
        S(Move(drone_id="d1", destination_id="casa1",
               duration=6.0, command_id="d1_mv"), 2.0),
        S(Deliver(drone_id="d1", package_id="pkg_med", person_id="ana",
                  duration=2.0, command_id="d1_dl"), 8.0),
        # d2: vuela a casa2 (éxito) y allí, AISLADO, intenta recoger
        # pkg_lejos, que está en casa1 → FALLO PDDL "no co-localizados".
        # d2 queda en ERROR en casa2, bien visible (X roja).
        S(Move(drone_id="d2", destination_id="casa2",
               duration=6.0, command_id="d2_mv"), 0.0),
        S(PickUp(drone_id="d2", arm_id="izq", package_id="pkg_lejos",
                 duration=2.0, command_id="d2_pk_fail"), 6.0),
    ))


def main() -> int:
    return run_replay(
        build_world(), build_plan(),
        caption="droneplan_viz · demo_sim_failure (un dron falla)",
        smoke="--smoke" in sys.argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
