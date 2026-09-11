"""Ejemplo — Un transportador porta DOS cajas en un solo viaje.

Variante de la parte 1 pensada para ilustrar la carga de un transportador:
el dron carga dos cajas en un carrier en el depósito, arrastra el carrier
a casa1 y allí descarga y entrega. Sirve para recrear la parte derecha de
la figura 4.15 (un transportador con el número exacto de cajas que porta).

Mismo mundo de dos localizaciones que dos_cajas_un_dron.py, para que las
dos situaciones (dron con dos cajas frente a transportador con dos cajas)
se puedan capturar en escenas equivalentes.

Como todo lo hace el mismo dron sobre el mismo carrier en secuencia, no se
usa `inicio=` (modo secuencial).

Simetría de brazos en la traducción:
    - recoger / sacar_de OCUPAN un brazo  -> llevan `brazo=`.
    - poner_en / entregar LIBERAN el brazo -> no lo llevan.

Plan de referencia (secuencial):

    (coger     dron1 caja-comida brazo-izq deposito)
    (cargar    dron1 caja-comida carrier   deposito)
    (coger     dron1 caja-agua   brazo-izq deposito)
    (cargar    dron1 caja-agua   carrier   deposito)
    (volar-con dron1 carrier     deposito  casa1)
    (descargar dron1 caja-comida carrier   brazo-izq casa1)
    (entregar  dron1 caja-comida persona1  casa1)
    (descargar dron1 caja-agua   carrier   brazo-izq casa1)
    (entregar  dron1 caja-agua   persona2  casa1)

Ejecutar:
    python examples/dos_cajas_transportador.py            # abre la ventana
    python examples/dos_cajas_transportador.py --check    # solo simula y sale
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from droneplan_viz import DronePlanViz  # noqa: E402


def construir() -> DronePlanViz:
    viz = DronePlanViz()

    # --- Mundo (dos localizaciones, como en la parte 1) ---
    viz.world.location("deposito")
    viz.world.location("casa1")
    viz.world.costes({("deposito", "casa1"): 8}, simetrico=True)

    viz.world.content("comida")
    viz.world.content("agua")
    viz.world.person("persona1", at="casa1", necesita=["comida"])
    viz.world.person("persona2", at="casa1", necesita=["agua"])
    viz.world.package("caja_comida", contiene="comida", at="deposito")
    viz.world.package("caja_agua", contiene="agua", at="deposito")

    viz.agents.drone("dron1", at="deposito")
    # Capacidad 2: justo las dos cajas que va a portar en el viaje.
    viz.agents.transporter("carrier", capacidad=2, at="deposito")

    # --- Plan: cargar dos cajas en el carrier y llevarlas de una pasada ---
    viz.recoger("dron1", caja="caja_comida", brazo="izq")
    #   (coger dron1 caja-comida brazo-izq deposito)
    viz.poner_en("dron1", caja="caja_comida", transportador="carrier")
    #   (cargar dron1 caja-comida carrier deposito)   [libera el brazo]
    viz.recoger("dron1", caja="caja_agua", brazo="izq")
    #   (coger dron1 caja-agua brazo-izq deposito)
    viz.poner_en("dron1", caja="caja_agua", transportador="carrier")
    #   (cargar dron1 caja-agua carrier deposito)      [el carrier lleva 2]

    viz.mover("dron1", a="casa1", con="carrier")
    #   (volar-con dron1 carrier deposito casa1)

    viz.sacar_de("dron1", caja="caja_comida", transportador="carrier", brazo="izq")
    #   (descargar dron1 caja-comida carrier brazo-izq casa1)  [ocupa el brazo]
    viz.entregar("dron1", caja="caja_comida", a="persona1")
    #   (entregar dron1 caja-comida persona1 casa1)
    viz.sacar_de("dron1", caja="caja_agua", transportador="carrier", brazo="izq")
    #   (descargar dron1 caja-agua carrier brazo-izq casa1)
    viz.entregar("dron1", caja="caja_agua", a="persona2")
    #   (entregar dron1 caja-agua persona2 casa1)

    return viz


def main() -> None:
    viz = construir()
    res = viz.simular()
    print("Dos cajas, un transportador")
    print(f"  éxito:    {res.succeeded}")
    print(f"  makespan: {res.makespan}")
    print(f"  fallos:   {len(res.failures)}")
    if "--check" not in sys.argv:
        viz.run()


if __name__ == "__main__":
    main()
