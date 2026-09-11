"""Ejemplo B — Grafo de localizaciones multisalto y costes ASIMÉTRICOS.

Hasta ahora los grafos eran pequeños y simétricos. Aquí el mapa es una red
con un cuello de botella (un "cruce" por el que hay que pasar) y aristas
con coste DISTINTO según la dirección (subir cuesta más que bajar). Esto
ejercita:

  - `costes(..., simetrico=False)`: aristas dirigidas. Declaramos cada
    sentido con su propio coste. Recuerda: la arista declarada es lo que
    hace transitable ese sentido; si solo declaras ida, la vuelta NO es
    transitable.

  - Un plan MULTISALTO: el dron no va directo, encadena varios `mover`
    para recorrer deposito → cruce → hospital, porque no existe arista
    directa deposito → hospital.

Mapa (coste por sentido):

        deposito --10--> cruce --7--> hospital
        deposito <--6--- cruce <--7--- hospital

    (no existe arista directa deposito <-> hospital: hay que pasar por cruce)

El dron sube cargado (deposito→cruce→hospital, coste 10+7) y, ya entregado,
podría bajar más barato (hospital→cruce→deposito, 7+6); aquí solo modelamos
la ida.

Ejecutar:
    python examples/grafo_multisalto.py            # abre la ventana
    python examples/grafo_multisalto.py --check    # solo simula y sale
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
    viz.world.location("cruce")
    viz.world.location("hospital")

    # Costes ASIMÉTRICOS, declarados sentido a sentido (simetrico=False).
    # Subir (hacia el hospital) cuesta más que bajar.
    viz.world.costes(
        {
            ("deposito", "cruce"): 10,   # subida
            ("cruce", "deposito"): 6,    # bajada (más barata)
            ("cruce", "hospital"): 7,
            ("hospital", "cruce"): 7,
        },
        simetrico=False,
    )
    # NO declaramos deposito<->hospital: no hay arista directa, hay que
    # pasar por el cruce.

    viz.world.content("medicina")
    viz.world.person("paciente", at="hospital", necesita=["medicina"])
    viz.world.package("caja_med", contiene="medicina", at="deposito")

    viz.agents.drone("dron1", at="deposito")

    # --- Plan multisalto (secuencial) ---
    viz.recoger("dron1", caja="caja_med", brazo="izq")
    viz.mover("dron1", a="cruce")       # 1er salto: deposito -> cruce
    viz.mover("dron1", a="hospital")    # 2º salto:  cruce -> hospital
    viz.entregar("dron1", caja="caja_med", a="paciente")

    return viz


def main() -> None:
    viz = construir()
    res = viz.simular()
    print("Ejemplo B — grafo multisalto, costes asimétricos")
    print(f"  éxito:    {res.succeeded}")
    print(f"  makespan: {res.makespan}")
    print(f"  fallos:   {len(res.failures)}")
    if "--check" not in sys.argv:
        viz.run()


if __name__ == "__main__":
    main()
