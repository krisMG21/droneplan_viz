"""Ejemplo — Un dron lleva DOS cajas en un solo viaje (una por brazo).

Variante de la parte 1 pensada para ilustrar la carga de un dron con los
dos brazos ocupados: el dron recoge una caja con el brazo izquierdo y otra
con el derecho en el depósito, vuela con ambas a casa1 y las entrega a dos
personas. Sirve para recrear la parte central de la figura 4.15 (un dron
con dos cajas, una por brazo).

Como todo lo hace el mismo dron en secuencia, no se usa `inicio=` (modo
secuencial: la fachada encadena los tiempos automáticamente).

Simetría de brazos en la traducción:
    - recoger OCUPA un brazo  -> lleva `brazo=`.
    - entregar LIBERA el brazo -> no lo lleva.

Plan de referencia (secuencial):

    (coger    dron1 caja-comida brazo-izq deposito)
    (coger    dron1 caja-agua   brazo-der deposito)
    (volar    dron1 deposito    casa1)
    (entregar dron1 caja-comida persona1  casa1)
    (entregar dron1 caja-agua   persona2  casa1)

Ejecutar:
    python examples/dos_cajas_un_dron.py            # abre la ventana
    python examples/dos_cajas_un_dron.py --check    # solo simula y sale
"""
from __future__ import annotations

import sys
from pathlib import Path

# Asegurar la raíz del repo en sys.path para poder importar el paquete
# sin instalarlo.
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

    # dron1 con los dos brazos por defecto (izq, der).
    viz.agents.drone("dron1", at="deposito")

    # --- Plan: dos cajas, una por brazo, en un solo viaje ---
    viz.recoger("dron1", caja="caja_comida", brazo="izq")
    #   (coger dron1 caja-comida brazo-izq deposito)
    viz.recoger("dron1", caja="caja_agua", brazo="der")
    #   (coger dron1 caja-agua brazo-der deposito)  [el otro brazo]
    viz.mover("dron1", a="casa1")
    #   (volar dron1 deposito casa1)                 [vuela con las dos]
    viz.entregar("dron1", caja="caja_comida", a="persona1")
    #   (entregar dron1 caja-comida persona1 casa1)  [libera brazo izq]
    viz.entregar("dron1", caja="caja_agua", a="persona2")
    #   (entregar dron1 caja-agua persona2 casa1)    [libera brazo der]

    return viz


def main() -> None:
    viz = construir()
    res = viz.simular()
    print("Dos cajas, un dron (una por brazo)")
    print(f"  éxito:    {res.succeeded}")
    print(f"  makespan: {res.makespan}")
    print(f"  fallos:   {len(res.failures)}")
    if "--check" not in sys.argv:
        viz.run()


if __name__ == "__main__":
    main()
