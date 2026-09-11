"""Genera capturas de pantalla de la app para la memoria de Sesión E.

Produce PNGs en examples/internal/out/app/ que documentan:

  01_initial_pause       Estado inicial: app pausada en t=0.
  02_play_mid             Reproducción a t≈9s con velocidad 2×.
  03_failure_selected     Escenario de fallo con el fallo seleccionado.
  04_step_navigation      Tras pulsar -> 3 veces desde el inicio.
  05_resize_wide          Tras resize a 1600×900.
  06_resize_small         Tras resize al mínimo 1024×640.
  07_end_state            Tras pulsar End: snapshot final del plan feliz.
  08_speed_active_4x      Botón 4× activo (selected).
  09_failure_panel_full   Vista cercana del panel lateral con métricas + fallos.

Uso:
    python scripts/demo_app.py

El script es headless (SDL dummy). No abre ventana real; produce los
PNGs leyendo el framebuffer interno de pygame.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

# Permitir ejecutar el script directamente (python scripts/demo_app.py).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pygame  # noqa: E402

from droneplan_viz_app.app import DroneplanVizApp  # noqa: E402
from droneplan_viz_app.app_state import ALLOWED_SPEEDS  # noqa: E402

OUT_DIR = Path(__file__).parent / "out" / "app"


def _save(app: DroneplanVizApp, name: str) -> None:
    """Guarda un screenshot bajo OUT_DIR con el nombre dado."""
    path = OUT_DIR / f"{name}.png"
    app.save_screenshot(str(path))
    print(f"  → {path.name} ({path.stat().st_size} bytes)")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"droneplan_viz · demo_app · capturas en {OUT_DIR}")

    # 01. Estado inicial pausado.
    app = DroneplanVizApp(scenario_name="demo")
    _save(app, "01_initial_pause")

    # 02. Reproducción en marcha a velocidad 2× en t≈9s (mitad de Move).
    app.state.playback_time = 9.0
    app.state.playback_speed = 2.0
    _save(app, "02_play_mid")

    # 04. Step navigation (avanzar al 3er snapshot).
    app.state.playback_time = app.timeline.snapshot_times[3]
    app.state.paused = True
    _save(app, "04_step_navigation")

    # 07. Final del plan feliz.
    app.state.playback_time = app.timeline.duration
    _save(app, "07_end_state")

    # 08. Botón 4× activo.
    app.state.playback_speed = 4.0
    app.state.playback_time = 5.0
    _save(app, "08_speed_active_4x")

    pygame.quit()

    # 03. Escenario de fallo con fallo seleccionado.
    app2 = DroneplanVizApp(scenario_name="failure_demo")
    app2.state.playback_time = 2.0
    app2.state.paused = True
    app2.state.selected_failure = 0
    # Revelar la pestaña Métricas+Fallos del panel lateral: con el rediseño
    # del inventario el panel pasó a tener dos pestañas y por defecto muestra
    # Inventario, así que sin esto la captura del fallo no mostraría el panel.
    app2.hud.show_metrics_failures_tab()
    _save(app2, "03_failure_selected")
    _save(app2, "09_failure_panel_full")  # mismo frame, nombre alternativo.

    # 05. Resize wide.
    pygame.event.post(pygame.event.Event(
        pygame.VIDEORESIZE, w=1600, h=900, size=(1600, 900)
    ))
    app2.run_one_frame(dt_override=0.0)
    app2.hud.show_metrics_failures_tab()  # relayout reconstruye el panel: reabrir.
    _save(app2, "05_resize_wide")

    # 06. Resize al mínimo.
    pygame.event.post(pygame.event.Event(
        pygame.VIDEORESIZE, w=1024, h=640, size=(1024, 640)
    ))
    app2.run_one_frame(dt_override=0.0)
    app2.hud.show_metrics_failures_tab()
    _save(app2, "06_resize_small")

    pygame.quit()

    print(f"\nTerminado. {len(list(OUT_DIR.glob('*.png')))} PNGs en {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
