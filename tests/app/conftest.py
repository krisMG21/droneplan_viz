"""Configuración de tests del paquete droneplan_viz_app.

Establece los drivers SDL "dummy" ANTES de cualquier import de pygame
para que los tests funcionen en entornos sin display (CI, contenedores,
sandbox). Mismo patrón que tests/render/conftest.py de Sesión D.

Uso de `setdefault` (no asignación directa): si el usuario ejecuta los
tests en una máquina con display real y quiere ver la ventana, puede
exportar SDL_VIDEODRIVER=x11 (o lo que corresponda) y los tests lo
respetan.

Nota: la APP real (no los tests) NO usa SDL dummy. Cuando un usuario
lanza `droneplan-viz` desde terminal, pygame crea una ventana real.
Lo dummy es exclusivo de los tests de smoke del HUD que necesitan
pygame.display.set_mode sin display.
"""
import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
