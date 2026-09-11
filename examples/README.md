# Ejemplos de la fachada `DronePlanViz`

Scripts que muestran cómo traducir, **a mano**, la salida de un planificador
(PDDL/JSHOP) a llamadas de la fachada pública `DronePlanViz`, en complejidad
creciente. Donde hay un plan, cada acción lleva al lado, en comentario, la
línea PDDL que traduce.

## Las tres partes del curso

| Ejemplo | Caso | Modo de tiempos |
|---|---|---|
| `parte1_ff_secuencial.py` | Un dron: recoge, vuela, entrega. | Secuencial (plan FF sin tiempos) |
| `parte2_transportador.py` | Un dron con transportador (carrier): carga, arrastra, descarga y entrega. | Secuencial |
| `parte3_temporal_concurrente.py` | Dos drones entregando en paralelo. | Temporal (plan OPTIC con `inicio=`) |

## Ejemplos transversales (configuraciones y escala)

| Ejemplo | Qué añade |
|---|---|
| `api_referencias_y_alias.py` | Identificar por **referencia** (no solo por id), **alias** en inglés (move/grab/deliver…), `at_screen=`, dron **monobrazo** y dron **explorador** (`arms=[]`). |
| `grafo_multisalto.py` | Grafo con cuello de botella, **costes asimétricos** (`simetrico=False`) y plan **multisalto** (deposito→cruce→hospital). |
| `diagnostico_fallos_y_errores.py` | Contrasta **errores de construcción** (excepción inmediata, try/except) con **fallos de ejecución** (reportados en `RunResult.failures`, sin excepción). Herramienta de consola. |
| `diagnostico_visual_de_fallo.py` | Abre la **aplicación** con un escenario de fallo: un dron entra en estado ERROR (X roja, panel de fallos, marca en la barra de progreso) mientras otro completa su plan en paralelo. Complementa a `diagnostico_fallos_y_errores.py`, que muestra lo mismo en consola. |
| `escala_reparto_masivo.py` | **Escala**: 1 depósito + N casas y D drones repartiendo en paralelo, plan generado por bucle. Simula y mide. Probado hasta 120 casas / 12 drones (≈470 acciones) sin fallos. |

## Ejecutar

La mayoría abre la ventana interactiva por defecto y acepta `--check` para
solo simular e imprimir el resultado (sin ventana):

```bash
python examples/parte1_ff_secuencial.py            # ventana
python examples/parte1_ff_secuencial.py --check    # solo simula
```

Dos excepciones:

```bash
# Solo consola (no abre ventana):
python examples/diagnostico_fallos_y_errores.py

# Escala: por defecto SOLO simula y mide; --run abre además la ventana.
python examples/escala_reparto_masivo.py
python examples/escala_reparto_masivo.py --casas 60 --drones 10
python examples/escala_reparto_masivo.py --run
```

## Lo que conviene fijarse

- **Identificar por id o por referencia**: puedes pasar `"dron1"` (el id) o
  el objeto que te devolvió `viz.agents.drone(...)`. Son equivalentes.
- **Simetría de brazos**: `recoger` y `sacar_de` ocupan un brazo concreto y
  por eso llevan `brazo=`; `entregar` y `poner_en` lo liberan y no lo llevan.
- **Costes = aristas**: declarar un coste entre dos localizaciones con
  `viz.world.costes(...)` es lo que las hace transitables. Sin esa arista,
  un `mover` hacia allí falla en la ejecución ("la arista no es transitable").
- **Regla de tiempos "todo o nada"**: o todas las acciones llevan `inicio=`
  (plan temporal) o ninguna (plan secuencial, la fachada encadena los
  tiempos sola). Mezclar las dos cosas es un error.
- **Dos niveles de error**: lo mal *declarado* salta como excepción al
  construir; lo físicamente *imposible* aparece en `RunResult.failures` al
  simular. El ejemplo `diagnostico_fallos_y_errores.py` enseña a leer ambos.
- **Feedback visual en la ventana**: cada persona usa un sprite mientras
  ESPERA y cambia al recibir su entrega (el dron se acerca a ella y no queda
  caja en el suelo). Las cajas llevan un **outline del color de su tipo de
  contenido**; la paleta (reproducible) la da `viz.colores_contenido()` y se
  inyecta sola al hacer `viz.run()`. Para tu propia paleta:
  `viz.run(content_colors=viz.colores_contenido(seed=N))`.

## Subcarpeta `internal/`

`internal/` contiene material de desarrollo, no ejemplos de uso de la
biblioteca. Se conserva como referencia del trabajo de fases previas y
no representa la forma prevista de usar la herramienta:

- Sondas del motor de render, headless, que generan PNG para verificar
  el dibujo de forma aislada (`demo_render.py`, `demo_camera.py`,
  `_anim_probe.py`, `_stack_probe.py`).
- Visores de reproducción anteriores al desarrollo de la aplicación
  interactiva (`demo_sim.py`, `demo_sim_carrier.py`,
  `demo_sim_concurrent.py`, `demo_sim_failure.py`): abren una ventana con
  la animación del render, pero sin el HUD ni los controles de la
  aplicación. Los casos que cubren (secuencial, transportador,
  concurrente y fallo) se muestran con la aplicación real en los
  ejemplos `parteN` y en `diagnostico_fallos_y_errores.py`.
- Un generador de capturas de la aplicación, headless
  (`generar_capturas_app.py`), usado para producir material gráfico.
