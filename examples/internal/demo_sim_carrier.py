"""Demo de simulación: un CARRIER de capacidad 5 con 5 paquetes.

Un único dron usa un transporter (carrier) de capacidad 5 para llevar 5
paquetes del depósito al hospital:

  1. Carga: para cada paquete, lo RECOGE y lo CARGA en el carrier
     (LoadIntoTransporter exige sostenerlo antes: PickUp → Load). Al
     terminar, el carrier va lleno (5/5).
  2. Traslado: arrastra el carrier (MoveWithTransporter) al hospital;
     los 5 paquetes viajan dentro.
  3. Descarga: para cada paquete, lo DESCARGA del carrier y lo ENTREGA
     al paciente que lo necesita (UnloadFromTransporter → Deliver).

Todo lo hace el mismo dron sobre el mismo carrier, así que los Commands van
secuenciados (no podrían solaparse sin chocar por concurrencia de recursos).

Controles: ESPACIO pausa · ← → snapshots · R reinicia · ESC/Q sale.

Uso (ventana):           python scripts/demo_sim_carrier.py
Uso (humo headless):     SDL_VIDEODRIVER=dummy python scripts/demo_sim_carrier.py --smoke
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from droneplan_viz.commands import (  # noqa: E402
    Deliver, LoadIntoTransporter, MoveWithTransporter, PickUp, UnloadFromTransporter,
)
from droneplan_viz.domain import (  # noqa: E402
    Arm, AtLocation, Content, Drone, DroneState, Location, Package, Person,
    Transporter, World,
)
from droneplan_viz.runtime import Plan, ScheduledCommand  # noqa: E402
from examples.internal._demo_common import run_replay  # noqa: E402

N = 5  # paquetes = capacidad del carrier


def build_world() -> World:
    # 5 contenidos distintos, un paquete y un paciente por cada uno.
    contents = {f"c{i}": Content(id=f"c{i}") for i in range(N)}
    locs = {
        "deposito": Location(id="deposito"),
        "hospital": Location(id="hospital"),
    }
    drones = {
        "d1": Drone(id="d1", position="deposito",
                    arms=(Arm(id="izq"), Arm(id="der")), state=DroneState.IDLE),
    }
    transporters = {
        "carrier": Transporter(id="carrier", position="deposito", capacity=N),
    }
    packages = {
        f"pkg{i}": Package(id=f"pkg{i}", contains=contents[f"c{i}"],
                           at=AtLocation(loc_id="deposito"))
        for i in range(N)
    }
    persons = {
        f"p{i}": Person(id=f"p{i}", position="hospital", needs=(contents[f"c{i}"],))
        for i in range(N)
    }
    costs = {("deposito", "hospital"): 8.0, ("hospital", "deposito"): 8.0}
    return World(locations=locs, drones=drones, transporters=transporters,
                 packages=packages, persons=persons, contents=contents, costs=costs)


def build_plan() -> Plan:
    sched: list[ScheduledCommand] = []
    t = 0.0

    def add(cmd, dur):
        nonlocal t
        sched.append(ScheduledCommand(command=cmd, start_time=t))
        t += dur

    # 1. Carga: recoger + cargar cada paquete en el carrier (en el depósito).
    for i in range(N):
        add(PickUp(drone_id="d1", arm_id="izq", package_id=f"pkg{i}",
                   duration=1.0, command_id=f"pk{i}"), 1.0)
        add(LoadIntoTransporter(drone_id="d1", package_id=f"pkg{i}",
                                transporter_id="carrier", duration=1.0,
                                command_id=f"ld{i}"), 1.0)

    # 2. Traslado del carrier lleno al hospital.
    add(MoveWithTransporter(drone_id="d1", transporter_id="carrier",
                            destination_id="hospital", duration=4.0,
                            command_id="mv"), 4.0)

    # 3. Descarga + entrega de cada paquete a su paciente.
    for i in range(N):
        add(UnloadFromTransporter(drone_id="d1", arm_id="izq", package_id=f"pkg{i}",
                                  transporter_id="carrier", duration=1.0,
                                  command_id=f"ul{i}"), 1.0)
        add(Deliver(drone_id="d1", package_id=f"pkg{i}", person_id=f"p{i}",
                    duration=1.0, command_id=f"dl{i}"), 1.0)

    return Plan(scheduled=tuple(sched))


def main() -> int:
    return run_replay(
        build_world(), build_plan(),
        caption="droneplan_viz · demo_sim_carrier (carrier x5)",
        smoke="--smoke" in sys.argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
