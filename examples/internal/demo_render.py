"""
Demo headless del paquete render: genera varios PNGs en examples/internal/out/.

Sirve como:
- Verificación visual manual del aspecto del render.
- Material gráfico para la memoria de Sesión D.
- Smoke test end-to-end ejecutable (si crashea, algo está roto).

Uso:
    python scripts/demo_render.py

Salida:
    examples/internal/out/01_world_inicial.png      Render estático del estado inicial.
    examples/internal/out/02_pickup_progress_50.png Frame medio durante un PickUp.
    examples/internal/out/03_move_progress_25.png   Drone empezando a volar.
    examples/internal/out/04_move_progress_50.png   Drone a medio camino.
    examples/internal/out/05_move_progress_75.png   Drone llegando al destino.
    examples/internal/out/06_deliver_progress_50.png Paquete a medio entregar.
    examples/internal/out/07_world_final.png        Estado final tras el plan.
    examples/internal/out/08_drone_error.png        Bonus: drone en ERROR (rojo + X).

El script es completamente headless: configura SDL_VIDEODRIVER=dummy
antes de importar pygame, así que funciona en CI, contenedores y
sandboxes sin display real.

No depende de assets externos: todos los sprites se generan
programáticamente desde el render (decisión §2.6 de la propuesta de
Sesión D).
"""
import os
import sys
from pathlib import Path

# Headless: imprescindible llamar a esto ANTES de importar pygame.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

# Permitir ejecutar el script directamente (python scripts/demo_render.py):
# aseguramos que la RAÍZ del repo esté en sys.path para importar el paquete.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pygame  # noqa: E402

from droneplan_viz.commands import (  # noqa: E402
    Deliver,
    Move,
    PickUp,
)
from droneplan_viz.domain import (  # noqa: E402
    AtLocation,
    Content,
    MetricsTracker,
    Package,
    Transporter,
    World,
)
from droneplan_viz.domain.arm import Arm  # noqa: E402
from droneplan_viz.domain.drone import Drone  # noqa: E402
from droneplan_viz.domain.drone_state import DroneState  # noqa: E402
from droneplan_viz.domain.location import Location  # noqa: E402
from droneplan_viz.domain.person import Person  # noqa: E402
from droneplan_viz.history import WorldSnapshot  # noqa: E402
from droneplan_viz.render import (  # noqa: E402
    Theme,
    Timeline,
    render_frame,
    render_snapshot,
)
from droneplan_viz.runtime import Plan, PlanRunner, ScheduledCommand  # noqa: E402


# ---------------------------------------------------------------------------
# Configuración de la demo
# ---------------------------------------------------------------------------

CANVAS_SIZE = (900, 600)
OUT_DIR = Path(__file__).parent / "out"


# ---------------------------------------------------------------------------
# Construcción del world docente
# ---------------------------------------------------------------------------


def build_demo_world() -> World:
    """World con todas las clases de entidad presentes para una demo rica.

    Topología:
        Locations (4): casa1, casa2, deposito, hospital.
        Drones (2): d1 (2 brazos, en deposito), d2 (1 brazo, en casa1).
        Transporter (1): t1 en deposito, capacidad 4.
        Persons (2): p1 en casa1 (necesita medicina + comida),
                     p2 en casa2 (necesita agua).
        Packages (3):
            pkg_med1 (medicina) AtLocation deposito.
            pkg_food1 (comida) AtLocation deposito.
            pkg_water1 (agua) AtLocation deposito.
        Costs: aristas entre las 4 locs (grafo completo).
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
    # Grafo completo entre 4 locs (12 aristas dirigidas, 6 no dirigidas):
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


def build_demo_plan() -> Plan:
    """Plan durativo realista que d1 ejecuta:

        t=0  → PickUp pkg_med1 con brazo izq (5s)
        t=5  → Move deposito → casa1 (8s)
        t=13 → Deliver pkg_med1 a p1 (5s)

    Total: 18s. Tras este plan, p1 recibió medicina y aún necesita comida.
    Diseñado para que se vean los tres tipos de transición (PickUp,
    Move, Deliver) en frames intermedios.
    """
    return Plan(scheduled=(
        ScheduledCommand(
            command=PickUp(
                drone_id="d1",
                arm_id="izq",
                package_id="pkg_med1",
                duration=5.0,
                command_id="step_pickup",
            ),
            start_time=0.0,
        ),
        ScheduledCommand(
            command=Move(
                drone_id="d1",
                destination_id="casa1",
                duration=8.0,
                command_id="step_move",
            ),
            start_time=5.0,
        ),
        ScheduledCommand(
            command=Deliver(
                drone_id="d1",
                package_id="pkg_med1",
                person_id="p1",
                duration=5.0,
                command_id="step_deliver",
            ),
            start_time=13.0,
        ),
    ))


def build_error_world() -> World:
    """World con un drone YA en estado ERROR para demostrar la X roja."""
    contents = {"medicina": Content(id="medicina")}
    locs = {
        "casa1": Location(id="casa1"),
        "casa2": Location(id="casa2"),
        "hospital": Location(id="hospital"),
    }
    drones = {
        "d1": Drone(
            id="d1",
            position="casa1",
            arms=(Arm(id="izq"),),
            state=DroneState.ERROR,
        ),
        "d2": Drone(
            id="d2",
            position="casa2",
            arms=(),
            state=DroneState.IDLE,
        ),
    }
    persons = {
        "p1": Person(id="p1", position="hospital", needs=(contents["medicina"],)),
    }
    costs = {
        ("casa1", "casa2"): 5.0, ("casa2", "casa1"): 5.0,
        ("casa1", "hospital"): 8.0, ("hospital", "casa1"): 8.0,
        ("casa2", "hospital"): 7.0, ("hospital", "casa2"): 7.0,
    }
    return World(
        locations=locs,
        drones=drones,
        persons=persons,
        contents=contents,
        costs=costs,
    )


# ---------------------------------------------------------------------------
# Generación de PNGs
# ---------------------------------------------------------------------------


def save_render_snapshot(
    out_path: Path,
    snapshot: WorldSnapshot,
    theme: Theme,
    sprite_manager=None,
) -> None:
    """Renderiza un snapshot estático y lo guarda como PNG."""
    surf = pygame.Surface(CANVAS_SIZE)
    render_snapshot(surf, snapshot, theme=theme, sprite_manager=sprite_manager)
    pygame.image.save(surf, str(out_path))
    print(f"  → {out_path.name} ({out_path.stat().st_size} bytes)")


def save_render_frame(
    out_path: Path,
    snap_a: WorldSnapshot,
    snap_b: WorldSnapshot,
    progress: float,
    theme: Theme,
    sprite_manager=None,
) -> None:
    """Renderiza un frame interpolado y lo guarda como PNG."""
    surf = pygame.Surface(CANVAS_SIZE)
    render_frame(surf, snap_a, snap_b, progress, theme=theme, sprite_manager=sprite_manager)
    pygame.image.save(surf, str(out_path))
    print(f"  → {out_path.name} ({out_path.stat().st_size} bytes)")


def find_pair_for_command(history, command_id: str) -> tuple[WorldSnapshot, WorldSnapshot] | None:
    """Localiza el par (snap_start, snap_end) que produce el Command con
    el id dado. Para Commands con duration>0 hay DOS snapshots con el
    mismo produced_by.command_id; los devolvemos en orden.
    """
    matches: list[WorldSnapshot] = []
    for i in range(len(history)):
        snap = history.at(i)
        if snap.produced_by is not None and snap.produced_by.command_id == command_id:
            matches.append(snap)
    if len(matches) != 2:
        return None
    return (matches[0], matches[1])


def main() -> None:
    print(f"droneplan_viz · demo_render · canvas {CANVAS_SIZE[0]}x{CANVAS_SIZE[1]}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    theme = Theme.default()

    # SpriteManager opcional: si DRONEPLAN_SPRITES=1 y hay assets, el demo
    # usa el camino sprite (composición por capas); si no, primitivas.
    sprite_manager = None
    if os.environ.get("DRONEPLAN_SPRITES") == "1":
        from droneplan_viz.render.sprite_manager import SpriteManager
        sprite_manager = SpriteManager(theme)
        print("  (sprites activados vía DRONEPLAN_SPRITES=1)")

    # ----- Plan principal -----
    world = build_demo_world()
    plan = build_demo_plan()
    runner = PlanRunner(world)
    result = runner.execute(plan)
    if not result.succeeded:
        print(f"ERROR: el plan falló en demo: {result.failures}", file=sys.stderr)
        sys.exit(1)

    history = runner.history
    print(f"  plan ejecutado: {len(history)} snapshots, makespan={result.makespan:.1f}s")

    # ----- 01: estado inicial -----
    print("Generando snapshots...")
    save_render_snapshot(OUT_DIR / "01_world_inicial.png", history.at(0), theme, sprite_manager)

    # ----- 02: PickUp a progress=0.5 -----
    pickup_pair = find_pair_for_command(history, "step_pickup")
    if pickup_pair is not None:
        snap_a, snap_b = pickup_pair
        save_render_frame(
            OUT_DIR / "02_pickup_progress_50.png",
            snap_a, snap_b, 0.5, theme, sprite_manager,
        )

    # ----- 03-05: Move a progress 0.25, 0.50, 0.75 -----
    move_pair = find_pair_for_command(history, "step_move")
    if move_pair is not None:
        snap_a, snap_b = move_pair
        save_render_frame(
            OUT_DIR / "03_move_progress_25.png",
            snap_a, snap_b, 0.25, theme, sprite_manager,
        )
        save_render_frame(
            OUT_DIR / "04_move_progress_50.png",
            snap_a, snap_b, 0.50, theme, sprite_manager,
        )
        save_render_frame(
            OUT_DIR / "05_move_progress_75.png",
            snap_a, snap_b, 0.75, theme, sprite_manager,
        )

    # ----- 06: Deliver a progress=0.5 -----
    deliver_pair = find_pair_for_command(history, "step_deliver")
    if deliver_pair is not None:
        snap_a, snap_b = deliver_pair
        save_render_frame(
            OUT_DIR / "06_deliver_progress_50.png",
            snap_a, snap_b, 0.5, theme, sprite_manager,
        )

    # ----- 07: estado final -----
    save_render_snapshot(
        OUT_DIR / "07_world_final.png",
        history.at(len(history) - 1),
        theme, sprite_manager,
    )

    # ----- 08: bonus, drone en ERROR -----
    print("Generando demo de fallo...")
    error_world = build_error_world()
    error_snap = WorldSnapshot(
        world=error_world,
        metrics=MetricsTracker(),
        produced_by=None,
        timestamp=0.0,
    )
    save_render_snapshot(OUT_DIR / "08_drone_error.png", error_snap, theme, sprite_manager)

    # ----- Demo opcional: timeline sample en t=duration/2 -----
    timeline = Timeline(history, theme=theme)
    print(f"  timeline duration={timeline.duration:.2f}s (con fallback de {theme.fallback_step_duration}s)")
    save_render_frame_via_timeline(
        OUT_DIR / "09_timeline_mid.png",
        timeline,
        timeline.duration / 2.0,
        theme,
    )

    print(f"\nTerminado. PNGs en {OUT_DIR.resolve()}")


def save_render_frame_via_timeline(
    out_path: Path,
    timeline: Timeline,
    playback_time: float,
    theme: Theme,
) -> None:
    """Demo del flujo real que usará la UI: muestra cómo Timeline.sample
    + render_frame se componen para producir un frame a un tiempo dado.
    """
    snap_a, snap_b, progress = timeline.sample(playback_time)
    surf = pygame.Surface(CANVAS_SIZE)
    render_frame(surf, snap_a, snap_b, progress, theme=theme)
    pygame.image.save(surf, str(out_path))
    print(
        f"  → {out_path.name} "
        f"(playback_time={playback_time:.2f}, progress={progress:.3f}, "
        f"{out_path.stat().st_size} bytes)"
    )


if __name__ == "__main__":
    main()
