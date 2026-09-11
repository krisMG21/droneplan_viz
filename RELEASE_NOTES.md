# droneplan_viz v1.0.0

Primera versión estable de **droneplan_viz**, una biblioteca de Python para la
visualización y validación lógica de planes de planificación automática (PDDL)
en un dominio logístico de drones.

## Funcionalidades

### Biblioteca
- **Reproducción de planes secuenciales (STRIPS) y temporales (PDDL 2.1)** con
  un único motor de ejecución basado en eventos, que admite acciones durativas
  y concurrentes.
- **Validación lógica integrada**: un validador central y una máquina de
  estados por dron comprueban cada acción antes de aplicarla (disponibilidad de
  brazos, capacidad de los transportadores y que los objetos estén en el mismo
  lugar). Un plan inconsistente lleva el dron a un estado de ERROR.
- **API de fachada** con constructores encadenables para montar el mundo y el
  plan, con métodos en español e inglés e identificadores deterministas.
- **Historial de simulación** que permite navegar hacia delante y hacia atrás
  sin recalcular.
- **Renderizado** con sprites en pixel art y respaldo automático a primitivas
  geométricas cuando faltan los recursos.

### Aplicación interactiva
- Reproducción con controles de tiempo, velocidades ajustables y pausa.
- Cámara con zoom y desplazamiento.
- Avance fotograma a fotograma.
- Panel de métricas, lista de acciones y panel de fallos.
- Capturas de pantalla (ventana completa o solo el visor).
- Panel de ayuda integrado y alternancia de etiquetas.

## Requisitos
- Python 3.12 o 3.13.

## Instalación
```bash
pip install -e ".[app]"
```

## Pruebas
Cerca de 1920 pruebas automatizadas cubren el dominio, el motor de ejecución,
el renderizado y la aplicación.
```bash
pytest
```
