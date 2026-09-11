"""Demostración interactiva del comportamiento ante un error de plan.

Abre la aplicación droneplan_viz en su funcionalidad completa (ventana
real, controles de reproducción, HUD y panel lateral) cargando el
escenario didáctico de fallo. A diferencia de examples/internal/generar_capturas_app.py, que es
headless y solo genera capturas PNG, este ejemplo lanza la interfaz para
que el usuario interactúe con ella.

Qué muestra el escenario
-------------------------
El plan mezcla acciones correctas con una que falla, de modo que la
interfaz exhibe a la vez la ejecución normal y la gestión del error:

    t=0   PickUp  d1 recoge pkg_med1 en deposito        (correcto)
    t=2   PickUp  d2 intenta recoger pkg_water1         (FALLA)
                  d2 está en casa1 y el paquete en deposito: no están
                  co-localizados, así que la precondición PDDL no se
                  cumple. d2 transita al estado ERROR.
    t=5   Move    d1 vuela a casa1                        (correcto)
    t=13  Deliver d1 entrega pkg_med1 a la persona p1     (correcto)

El fallo de d2 no cancela el resto del plan: d1 completa su recorrido en
paralelo. Esto permite observar simultáneamente una rama de ejecución
correcta y otra fallida.

Manifestaciones del error en la interfaz
----------------------------------------
Al reproducir el plan, el error se hace visible mediante:

    * una X roja sobre el dron d2 en el área de simulación,
    * una entrada en el panel de fallos (pestaña Métricas+Fallos),
    * una marca roja en la barra de progreso, en el instante del fallo.

Controles
---------
La aplicación arranca en su estado inicial habitual. Los controles son
los estándar de la interfaz:

    Barra espaciadora   reproducir / pausar
    Flechas izq/der     saltar entre los instantes clave del plan
    , .                 avanzar / retroceder fotograma a fotograma
    Inicio / Fin        saltar al principio / final del plan
    Rueda del ratón     zoom sobre la escena
    Arrastrar           desplazar la cámara

Uso
---
    python examples/diagnostico_visual_de_fallo.py

Equivale a lanzar la aplicación con el escenario de fallo:

    python -m droneplan_viz_app --scenario failure_demo
"""
from __future__ import annotations

import sys
from pathlib import Path

# Permitir la ejecución directa (python examples/diagnostico_visual_de_fallo.py)
# sin necesidad de instalar el paquete.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from droneplan_viz_app.app import DroneplanVizApp  # noqa: E402


def main() -> int:
    """Lanza la aplicación con el escenario de fallo."""
    app = DroneplanVizApp(scenario_name="failure_demo", window_size=(1280, 800))
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
