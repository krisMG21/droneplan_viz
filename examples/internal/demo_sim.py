"""Demo de "ejecución tipo" animada — equivalente moderno de test_sim.py.

El antiguo test_sim.py de raíz estaba escrito contra una API que YA NO
EXISTE en el proyecto:

    El espacio se modela como un grafo de localizaciones discretas con
    costes entre ellas, y el render vive en droneplan_viz.render
    (render_frame + Timeline). Este script muestra la API real:

    1. Construir un World (locations + drone + transporter + paquetes +
       persona).
    2. Declarar un Plan durativo de Commands (Move / PickUp /
       LoadIntoTransporter / MoveWithTransporter / UnloadFromTransporter /
       Deliver) con tiempos de inicio.
    3. Ejecutarlo con PlanRunner.execute() → historial de snapshots.
    4. Animar el historial con Timeline + render_frame en una ventana
       pygame, con controles básicos.

Controles:
    ESPACIO   reproducir / pausar
    ←  →      retroceder / avanzar al snapshot anterior/siguiente
    R         reiniciar al principio
    ESC / ✕   salir

Uso (ventana):
    python scripts/demo_sim.py

Uso (humo headless, sin ventana, renderiza N frames y sale):
    SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy python scripts/demo_sim.py --smoke

Nota: este demo es LIBRARY-ONLY (no depende de droneplan_viz_app), igual
que demo_render.py y demo_camera.py. Por eso las etiquetas usan la fuente
del sistema; la app interactiva inyecta Monogram en el Theme de render.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pygame  # noqa: E402

from droneplan_viz.commands import (  # noqa: E402
    Deliver,
    LoadIntoTransporter,
    MoveWithTransporter,
    PickUp,
    UnloadFromTransporter,
)
from droneplan_viz.domain import (  # noqa: E402
    Arm,
    AtLocation,
    Content,
    Drone,
    DroneState,
    Location,
    Package,
    Person,
    Transporter,
    World,
)
from droneplan_viz.render import Theme, Timeline, render_world_at  # noqa: E402
from droneplan_viz.render.sprite_manager import SpriteManager  # noqa: E402
from droneplan_viz.runtime import Plan, PlanRunner, ScheduledCommand  # noqa: E402


# ---------------------------------------------------------------------------
# 1. World: grafo de locations (NO grid), drone, transporter, paquetes, persona
# ---------------------------------------------------------------------------
def build_world() -> World:
    contents = {
        "medicina": Content(id="medicina"),
        "comida": Content(id="comida"),
    }
    locations = {
        "base": Location(id="base"),
        "almacen": Location(id="almacen"),
        "hospital": Location(id="hospital"),
    }
    drones = {
        "drone1": Drone(
            id="drone1",
            position="base",
            arms=(Arm(id="izq"), Arm(id="der")),
            state=DroneState.IDLE,
        ),
    }
    transporters = {
        "carrier1": Transporter(id="carrier1", position="base", capacity=4),
    }
    packages = {
        "pkg_med": Package(
            id="pkg_med", contains=contents["medicina"],
            at=AtLocation(loc_id="base"),
        ),
        "pkg_food": Package(
            id="pkg_food", contains=contents["comida"],
            at=AtLocation(loc_id="base"),
        ),
    }
    persons = {
        "paciente": Person(
            id="paciente", position="hospital",
            needs=(contents["medicina"], contents["comida"]),
        ),
    }
    pairs = [
        ("base", "almacen", 6.0),
        ("base", "hospital", 9.0),
        ("almacen", "hospital", 7.0),
    ]
    costs: dict[tuple[str, str], float] = {}
    for o, d, c in pairs:
        costs[(o, d)] = c
        costs[(d, o)] = c
    return World(
        locations=locations,
        drones=drones,
        transporters=transporters,
        persons=persons,
        packages=packages,
        contents=contents,
        costs=costs,
    )


# ---------------------------------------------------------------------------
# 2. Plan: el drone usa el transporter para llevar dos paquetes al hospital
# ---------------------------------------------------------------------------
def build_plan() -> Plan:
    """drone1 mete dos paquetes en carrier1 (en la base), arrastra el
    transporter al hospital, descarga y entrega ambos al paciente.

    (Cargar en el transporter exige sostener antes el paquete: el ciclo
    intercala PickUp → LoadIntoTransporter. Todo lo hace el mismo drone
    sobre el mismo transporter, así que va estrictamente secuenciado.)
    """
    def S(cmd, t):
        return ScheduledCommand(command=cmd, start_time=t)

    return Plan(scheduled=(
        # Cargar pkg_med en el transporter (recoger + cargar), en la base.
        S(PickUp(drone_id="drone1", arm_id="izq", package_id="pkg_med",
                 duration=1.5, command_id="pk_med"), 0.0),
        S(LoadIntoTransporter(drone_id="drone1", package_id="pkg_med",
                              transporter_id="carrier1", duration=1.5,
                              command_id="ld_med"), 1.5),
        # Cargar pkg_food.
        S(PickUp(drone_id="drone1", arm_id="izq", package_id="pkg_food",
                 duration=1.5, command_id="pk_food"), 3.0),
        S(LoadIntoTransporter(drone_id="drone1", package_id="pkg_food",
                              transporter_id="carrier1", duration=1.5,
                              command_id="ld_food"), 4.5),
        # Arrastrar el transporter al hospital ("el drone vuela con la carga").
        S(MoveWithTransporter(drone_id="drone1", transporter_id="carrier1",
                              destination_id="hospital", duration=4.0,
                              command_id="mv_hosp"), 6.0),
        # Descargar y entregar ambos paquetes al paciente.
        S(UnloadFromTransporter(drone_id="drone1", arm_id="izq",
                                package_id="pkg_med", transporter_id="carrier1",
                                duration=1.5, command_id="ul_med"), 10.0),
        S(Deliver(drone_id="drone1", package_id="pkg_med",
                  person_id="paciente", duration=1.5, command_id="dl_med"), 11.5),
        S(UnloadFromTransporter(drone_id="drone1", arm_id="izq",
                                package_id="pkg_food", transporter_id="carrier1",
                                duration=1.5, command_id="ul_food"), 13.0),
        S(Deliver(drone_id="drone1", package_id="pkg_food",
                  person_id="paciente", duration=1.5, command_id="dl_food"), 14.5),
    ))


# ---------------------------------------------------------------------------
# 3 + 4. Ejecutar y animar
# ---------------------------------------------------------------------------
def main() -> int:
    smoke = "--smoke" in sys.argv
    if smoke:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    world = build_world()
    plan = build_plan()

    runner = PlanRunner(world)
    result = runner.execute(plan)
    print(f"Ejecución: succeeded={result.succeeded} "
          f"makespan={result.makespan} snapshots={result.history_length}")
    if not result.succeeded:
        for f in result.failures:
            print(f"  FALLO [{f.kind}] {f.reason}")

    theme = Theme.default()
    timeline = Timeline(runner.history, theme=theme)

    pygame.init()
    size = (900, 600)
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("droneplan_viz · demo_sim (ejecución tipo)")
    clock = pygame.time.Clock()
    sprite_manager = SpriteManager(theme)

    t = 0.0
    paused = False
    running = True
    frames = 0
    while running:
        dt = clock.tick(60) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    t = 0.0
                elif event.key == pygame.K_RIGHT:
                    # Avanzar al siguiente snapshot.
                    nxt = [s for s in timeline.snapshot_times if s > t + 1e-6]
                    t = nxt[0] if nxt else timeline.duration
                elif event.key == pygame.K_LEFT:
                    prv = [s for s in timeline.snapshot_times if s < t - 1e-6]
                    t = prv[-1] if prv else 0.0

        if not paused:
            t = min(t + dt, timeline.duration)

        screen.fill(theme.background)
        render_world_at(screen, timeline, t,
                        theme=theme, sprite_manager=sprite_manager,
                        anim_time=pygame.time.get_ticks() / 1000.0)
        pygame.display.flip()

        frames += 1
        if smoke and frames >= 5:
            running = False  # humo headless: unos pocos frames y salir.

    pygame.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
