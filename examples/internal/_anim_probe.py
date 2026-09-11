"""Reproduce el camino de pintado REAL de la app (render_world_at) con un
depósito que rebosa cajas (>15), para verificar el filtro de capacidad y el
orden por capas en las rutas animadas. Headless -> PNG.

    python scripts/_anim_probe.py [t] [out.png]
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pygame  # noqa: E402

from droneplan_viz import DronePlanViz  # noqa: E402
from droneplan_viz.runtime import PlanRunner  # noqa: E402
from droneplan_viz.render import Theme, Timeline, render_world_at  # noqa: E402
from droneplan_viz.render.sprite_manager import SpriteManager  # noqa: E402

_CONTENIDOS = ("medicina", "comida", "agua")


def construir(n_casas: int, n_drones: int) -> DronePlanViz:
    viz = DronePlanViz()
    viz.world.location("deposito")
    for c in _CONTENIDOS:
        viz.world.content(c)
    costes = {}
    for k in range(n_casas):
        casa = f"casa{k:02d}"
        viz.world.location(casa)
        costes[("deposito", casa)] = 6.0 + (k % 7)
        viz.world.person(f"vecino{k:02d}", at=casa, necesita=[_CONTENIDOS[k % 3]])
        viz.world.package(f"caja{k:02d}", contiene=_CONTENIDOS[k % 3], at="deposito")
    viz.world.costes(costes, simetrico=True)
    for d in range(n_drones):
        viz.agents.drone(f"dron{d}", at="deposito")
    for d in range(n_drones):
        casas = list(range(d, n_casas, n_drones))
        t = 0.0
        for pos, k in enumerate(casas):
            dron, caja, casa = f"dron{d}", f"caja{k:02d}", f"casa{k:02d}"
            viz.recoger(dron, caja=caja, brazo="izq", inicio=t, duracion=5.0,
                        id=f"r_{d}_{k}")
            t += 5.0
            viz.mover(dron, a=casa, inicio=t, duracion=8.0, id=f"ir_{d}_{k}")
            t += 8.0
            viz.entregar(dron, caja=caja, a=f"vecino{k:02d}", inicio=t,
                         duracion=5.0, id=f"e_{d}_{k}")
            t += 5.0
            if pos < len(casas) - 1:
                viz.mover(dron, a="deposito", inicio=t, duracion=8.0,
                          id=f"v_{d}_{k}")
                t += 8.0
    return viz


def main() -> None:
    t = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
    out = sys.argv[2] if len(sys.argv) > 2 else "examples/internal/out/_anim_t0.png"
    pygame.init()
    viz = construir(18, 6)
    world, plan = viz.build()
    runner = PlanRunner(world)
    runner.execute(plan)
    theme = Theme.default()
    sm = SpriteManager(theme)
    timeline = Timeline(runner.history, theme=theme)
    n_dep = sum(1 for p in world.packages.values()
                if getattr(p.at, "loc_id", None) == "deposito")
    surf = pygame.Surface((900, 700))
    render_world_at(surf, timeline, t, theme=theme, sprite_manager=sm)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    pygame.image.save(surf, out)
    print(f"cajas en deposito (t inicial) = {n_dep}; t={t} -> {out}")


if __name__ == "__main__":
    main()
