"""Ejemplo 3 — Plan TEMPORAL concurrente (OPTIC) traducido a la fachada.

Un planificador temporal (OPTIC/LPG-TD) produce un plan DURATIVO con
tiempos de inicio y duraciones explícitos, y permite que acciones de
agentes distintos se SOLAPEN en el tiempo. Aquí dos drones parten del
depósito y entregan a la vez, cada uno a su casa, sobre recursos disjuntos
(drones y cajas distintos). El makespan refleja el PARALELISMO (máximo de
las dos pistas), no la suma.

Como TODAS las acciones llevan `inicio=`, la fachada entra en modo
TEMPORAL y respeta esos tiempos absolutos tal cual (regla "todo o nada":
o todas con inicio, o ninguna).

Plan OPTIC de referencia (durativo; "t: (acción) [dur]"):

    0.0:  (coger dron1 caja-med brazo-izq deposito) [5.0]
    0.0:  (coger dron2 caja-com brazo-izq deposito) [5.0]
    5.0:  (volar dron1 deposito casa1)              [8.0]
    5.0:  (volar dron2 deposito casa2)              [8.0]
    13.0: (entregar dron1 caja-med persona1 casa1)  [5.0]
    13.0: (entregar dron2 caja-com persona2 casa2)  [5.0]

Las dos pistas arrancan a la vez (t=0) y terminan a la vez (t=18): makespan
18, no 36.

Ejecutar:
    python examples/parte3_temporal_concurrente.py            # abre la ventana
    python examples/parte3_temporal_concurrente.py --check    # solo simula y sale
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from droneplan_viz import DronePlanViz  # noqa: E402


def construir() -> DronePlanViz:
    viz = DronePlanViz()

    # --- Mundo ---
    viz.world.location("deposito")
    viz.world.location("casa1")
    viz.world.location("casa2")
    # Aristas transitables desde el depósito a cada casa.
    viz.world.costes(
        {("deposito", "casa1"): 8, ("deposito", "casa2"): 12},
        simetrico=True,
    )

    viz.world.content("medicina")
    viz.world.content("comida")
    viz.world.person("persona1", at="casa1", necesita=["medicina"])
    viz.world.person("persona2", at="casa2", necesita=["comida"])
    viz.world.package("caja_med", contiene="medicina", at="deposito")
    viz.world.package("caja_com", contiene="comida", at="deposito")

    viz.agents.drone("dron1", at="deposito")
    viz.agents.drone("dron2", at="deposito")

    # --- Plan temporal (todas con inicio= -> modo temporal) ---
    # Pista de dron1 y pista de dron2 arrancan concurrentes en t=0.
    viz.recoger("dron1", caja="caja_med", brazo="izq", inicio=0.0, duracion=5)
    #   0.0: (coger dron1 caja-med brazo-izq deposito) [5.0]
    viz.recoger("dron2", caja="caja_com", brazo="izq", inicio=0.0, duracion=5)
    #   0.0: (coger dron2 caja-com brazo-izq deposito) [5.0]

    viz.mover("dron1", a="casa1", inicio=5.0, duracion=8)
    #   5.0: (volar dron1 deposito casa1) [8.0]
    viz.mover("dron2", a="casa2", inicio=5.0, duracion=8)
    #   5.0: (volar dron2 deposito casa2) [8.0]

    viz.entregar("dron1", caja="caja_med", a="persona1", inicio=13.0, duracion=5)
    #   13.0: (entregar dron1 caja-med persona1 casa1) [5.0]
    viz.entregar("dron2", caja="caja_com", a="persona2", inicio=13.0, duracion=5)
    #   13.0: (entregar dron2 caja-com persona2 casa2) [5.0]

    return viz


def main() -> None:
    viz = construir()
    res = viz.simular()
    print("Parte 3 — temporal concurrente (OPTIC)")
    print(f"  éxito:    {res.succeeded}")
    print(f"  makespan: {res.makespan}  (paralelo: 18, no 36)")
    print(f"  fallos:   {len(res.failures)}")
    if "--check" not in sys.argv:
        viz.run()


if __name__ == "__main__":
    main()
