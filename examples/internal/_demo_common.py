"""Bucle de animación compartido por las demos de simulación (demo_sim*).

Library-only (no depende de droneplan_viz_app): ejecuta un Plan con
PlanRunner y anima el historial con Timeline + render_frame en una ventana
pygame, con controles básicos y un modo --smoke headless (renderiza unos
pocos frames y sale, para CI/sandboxes sin display).

Cada demo aporta su World y su Plan y llama a run_replay(...). El patrón
de import entre scripts es el mismo que usa demo_camera con demo_render:
`from examples.internal._demo_common import run_replay` (con la raíz del repo en
sys.path, scripts/ funciona como namespace package).

Controles:
    ESPACIO   reproducir / pausar
    ←  →      saltar al snapshot anterior / siguiente
    R         reiniciar al principio
    ESC / Q   salir
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def run_replay(world, plan, *, caption: str, smoke: bool = False) -> int:
    """Ejecuta `plan` sobre `world` y anima el resultado.

    Args:
        world: World inicial.
        plan: Plan a ejecutar.
        caption: título de la ventana.
        smoke: si True, fuerza SDL dummy y renderiza solo unos frames.

    Returns:
        0 al terminar.
    """
    if smoke:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    import pygame  # tras fijar SDL dummy en modo smoke.

    from droneplan_viz.render import Theme, Timeline, render_world_at
    from droneplan_viz.render.sprite_manager import SpriteManager
    from droneplan_viz.runtime import PlanRunner

    runner = PlanRunner(world)
    result = runner.execute(plan)
    print(
        f"Ejecución: succeeded={result.succeeded} "
        f"makespan={result.makespan} snapshots={result.history_length}"
    )
    if not result.succeeded:
        for f in result.failures:
            print(f"  FALLO [{f.kind}] {f.reason}")

    theme = Theme.default()
    timeline = Timeline(runner.history, theme=theme)

    pygame.init()
    size = (900, 600)
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption(caption)
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
            running = False

    pygame.quit()
    return 0
