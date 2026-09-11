"""Harness de prueba: renderiza N cajas libres y M carriers en una location
para inspeccionar el apilado y el orden de pintado. Headless -> PNG.

Uso:
    python scripts/_stack_probe.py N_BOXES M_CARRIERS out.png
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pygame  # noqa: E402

from droneplan_viz.domain import (  # noqa: E402
    AtLocation, Content, Location, MetricsTracker, Package, Transporter, World,
)
from droneplan_viz.history import WorldSnapshot  # noqa: E402
from droneplan_viz.render import Theme, render_snapshot  # noqa: E402
from droneplan_viz.render.sprite_manager import SpriteManager  # noqa: E402

CANVAS = (520, 440)


def build_world(n_boxes: int, m_carriers: int) -> World:
    locs = {"base": Location(id="base")}
    contents = {"c": Content(id="c")}
    packages = {
        f"pkg{i:02d}": Package(id=f"pkg{i:02d}", contains=contents["c"],
                               at=AtLocation(loc_id="base"))
        for i in range(n_boxes)
    }
    transporters = {
        f"car{i:02d}": Transporter(id=f"car{i:02d}", position="base", capacity=5)
        for i in range(m_carriers)
    }
    return World(locations=locs, drones={}, transporters=transporters,
                 packages=packages, persons={}, contents=contents, costs={})


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    m = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    out = sys.argv[3] if len(sys.argv) > 3 else "examples/internal/out/_probe.png"
    pygame.init()
    world = build_world(n, m)
    snap = WorldSnapshot(world=world, metrics=MetricsTracker(), produced_by=None)
    theme = Theme()
    surf = pygame.Surface(CANVAS)
    sm = SpriteManager(theme)
    render_snapshot(surf, snap, theme=theme, sprite_manager=sm)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    pygame.image.save(surf, out)
    print(f"boxes={n} carriers={m} -> {out}")


if __name__ == "__main__":
    main()
