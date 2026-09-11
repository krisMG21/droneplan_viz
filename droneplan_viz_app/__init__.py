"""Paquete droneplan_viz_app: aplicación interactiva del proyecto.

Wrapper pygame + pygame_gui sobre la librería droneplan_viz. Construye una
ventana redimensionable, ejecuta un Plan con PlanRunner, y proporciona al
usuario controles de reproducción (play/pause, paso adelante/atrás, control
de velocidad) sobre la animación que produce el paquete render.

Este paquete es HERMANO de droneplan_viz, no un subpaquete: quien solo use
la librería (validación, render headless) no paga el coste de pygame_gui
ni de la maquinaria de UI. El extras `app` de pyproject.toml gestiona la
dependencia: `pip install -e ".[app]"` la activa.

Responsabilidades de cada módulo (definidas, ver memoria_sesion_e.md):

- app_state    : dataclass mutable acotada con el estado de la UI.
- scenarios    : factorías de (World, Plan) para los escenarios docentes.
- layout       : cálculo puro de regiones (rects) de la ventana.
- controller   : mapeo puro evento → mutación de AppState.
- hud          : widgets de pygame_gui y su pintado.
- app          : loop principal y glue code.
- main         : entry point CLI (`droneplan-viz` tras pip install).

Filosofía heredada de las sesiones A-D:
- Inmutabilidad estructural en el dominio (intacto).
- Mutabilidad ACOTADA en AppState (única fuente de mutación de la UI).
- Funciones puras donde sea posible (controller, layout, scenarios).
- Composición sobre herencia: dataclasses + funciones, cero jerarquías.
- Cero modificaciones a domain/commands/history/runtime (la propiedad
  añadida a render/Timeline.snapshot_times en Paso E.3 es la única
  excepción, justificada en la memoria como adición no destructiva).
"""

__all__: list[str] = []
