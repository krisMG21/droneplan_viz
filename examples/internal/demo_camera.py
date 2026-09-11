"""Demo de la cámara (pan + zoom) sobre el escenario docente.

Genera capturas que ilustran el encargo D2 y verifica programáticamente
el invariante crítico (zoom=1, pan=(0,0) es píxel-idéntico al render sin
cámara). Pensado para la verificación end-to-end §9 de la nota de traspaso.

Headless: SDL dummy. Reutiliza build_demo_world de demo_render para no
duplicar el escenario (4 locations, 2 drones, transporter, paquetes,
personas).

Uso:
    SDL_VIDEODRIVER=dummy SDL_AUDIODRIVER=dummy python scripts/demo_camera.py

Variable de entorno opcional:
    DRONEPLAN_SPRITES=1  → usa SpriteManager con los assets reales.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np  # noqa: E402
import pygame  # noqa: E402
import pygame.surfarray as surfarray  # noqa: E402

from droneplan_viz.domain import MetricsTracker  # noqa: E402
from droneplan_viz.history import WorldSnapshot  # noqa: E402
from droneplan_viz.render import (  # noqa: E402
    ZOOM_MAX,
    ZOOM_MIN,
    Theme,
    locate_entity,
    render_snapshot,
)

# Reutilizamos el escenario docente del demo principal.
from examples.internal.demo_render import build_demo_world  # noqa: E402

CANVAS_SIZE = (900, 600)
OUT_DIR = Path(__file__).parent / "out"


def _snapshot(world) -> WorldSnapshot:
    return WorldSnapshot(
        world=world, metrics=MetricsTracker(), produced_by=None, timestamp=0.0
    )


def _save(path: Path, snapshot, theme, sprite_manager, *, zoom=1.0, pan=(0, 0)):
    surf = pygame.Surface(CANVAS_SIZE)
    render_snapshot(
        surf, snapshot, theme=theme, sprite_manager=sprite_manager, zoom=zoom, pan=pan
    )
    pygame.image.save(surf, str(path))
    return surf


def main() -> None:
    pygame.init()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    theme = Theme.default()
    sprite_manager = None
    if os.environ.get("DRONEPLAN_SPRITES") == "1":
        from droneplan_viz.render.sprite_manager import SpriteManager

        sprite_manager = SpriteManager(theme)

    world = build_demo_world()
    snap = _snapshot(world)

    print(f"ZOOM_MIN={ZOOM_MIN}  ZOOM_MAX={ZOOM_MAX}")

    # --- Capturas que pide la nota §9 ---
    base = _save(OUT_DIR / "cam_01_zoom1_pan0.png", snap, theme, sprite_manager,
                 zoom=1.0, pan=(0, 0))
    _save(OUT_DIR / "cam_02_zoom2.png", snap, theme, sprite_manager,
          zoom=2.0, pan=(0, 0))
    _save(OUT_DIR / "cam_03_zoom05.png", snap, theme, sprite_manager,
          zoom=0.5, pan=(0, 0))
    _save(OUT_DIR / "cam_04_pan_x100.png", snap, theme, sprite_manager,
          zoom=1.0, pan=(100, 0))
    _save(OUT_DIR / "cam_05_zoom2_pan.png", snap, theme, sprite_manager,
          zoom=2.0, pan=(100, 50))

    # Click-to-focus sobre un drone concreto (lo que hará E2).
    drone_id = next(iter(world.drones))
    lx, ly = locate_entity(world, drone_id, CANVAS_SIZE, theme=theme)
    pan = (CANVAS_SIZE[0] // 2 - lx, CANVAS_SIZE[1] // 2 - ly)
    _save(OUT_DIR / "cam_06_focus_drone.png", snap, theme, sprite_manager,
          zoom=1.0, pan=pan)
    print(f"locate_entity({drone_id!r}) = ({lx}, {ly}) → pan focus = {pan}")

    # --- Verificación del invariante: zoom=1, pan=0 == render sin cámara ---
    surf_no_cam = pygame.Surface(CANVAS_SIZE)
    render_snapshot(surf_no_cam, snap, theme=theme, sprite_manager=sprite_manager)
    identical = np.array_equal(
        surfarray.array3d(surf_no_cam), surfarray.array3d(base)
    )
    print(f"INVARIANTE (zoom=1,pan=0 == sin cámara): {identical}")
    if not identical:
        raise SystemExit("FALLO DE INVARIANTE: la cámara por defecto NO es transparente")

    print(f"Capturas escritas en {OUT_DIR}")


if __name__ == "__main__":
    main()
