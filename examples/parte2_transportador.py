"""Ejemplo 2 — Plan con TRANSPORTADOR (carrier) traducido a la fachada.

Un único dron usa un transportador para llevar dos cajas del depósito al
hospital de una sola pasada: las RECOGE y las PONE EN el carrier, ARRASTRA
el carrier hasta el hospital, y allí las SACA DE él y las ENTREGA. Como todo
lo hace el mismo dron sobre el mismo carrier, las acciones van en secuencia
(no podrían solaparse sin chocar por recursos), así que de nuevo NO usamos
`inicio=` (modo secuencial).

Observa la simetría de brazos en la traducción:
    - recoger / sacar_de OCUPAN un brazo  -> llevan `brazo=`.
    - poner_en / entregar LIBERAN el brazo -> no lo llevan.

Plan de referencia (secuencial):

    (coger      dron1 caja-med  brazo-izq deposito)
    (cargar     dron1 caja-med  carrier   deposito)
    (coger      dron1 caja-com  brazo-izq deposito)
    (cargar     dron1 caja-com  carrier   deposito)
    (volar-con  dron1 carrier   deposito  hospital)
    (descargar  dron1 caja-med  carrier   brazo-izq hospital)
    (entregar   dron1 caja-med  paciente1 hospital)
    (descargar  dron1 caja-com  carrier   brazo-izq hospital)
    (entregar   dron1 caja-com  paciente2 hospital)

Ejecutar:
    python examples/parte2_transportador.py            # abre la ventana
    python examples/parte2_transportador.py --check    # solo simula y sale
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
    viz.world.location("hospital")
    viz.world.costes({("deposito", "hospital"): 6}, simetrico=True)

    viz.world.content("medicina")
    viz.world.content("comida")
    viz.world.person("paciente1", at="hospital", necesita=["medicina"])
    viz.world.person("paciente2", at="hospital", necesita=["comida"])
    viz.world.package("caja_med", contiene="medicina", at="deposito")
    viz.world.package("caja_com", contiene="comida", at="deposito")

    viz.agents.drone("dron1", at="deposito")
    viz.agents.transporter("carrier", capacidad=4, at="deposito")

    # --- Plan con carrier (secuencial) ---
    viz.recoger("dron1", caja="caja_med", brazo="izq")
    #   (coger dron1 caja-med brazo-izq deposito)
    viz.poner_en("dron1", caja="caja_med", transportador="carrier")
    #   (cargar dron1 caja-med carrier deposito)        [libera el brazo]
    viz.recoger("dron1", caja="caja_com", brazo="izq")
    #   (coger dron1 caja-com brazo-izq deposito)
    viz.poner_en("dron1", caja="caja_com", transportador="carrier")
    #   (cargar dron1 caja-com carrier deposito)

    viz.mover("dron1", a="hospital", con="carrier")
    #   (volar-con dron1 carrier deposito hospital)

    viz.sacar_de("dron1", caja="caja_med", transportador="carrier", brazo="izq")
    #   (descargar dron1 caja-med carrier brazo-izq hospital) [ocupa el brazo]
    viz.entregar("dron1", caja="caja_med", a="paciente1")
    #   (entregar dron1 caja-med paciente1 hospital)
    viz.sacar_de("dron1", caja="caja_com", transportador="carrier", brazo="izq")
    #   (descargar dron1 caja-com carrier brazo-izq hospital)
    viz.entregar("dron1", caja="caja_com", a="paciente2")
    #   (entregar dron1 caja-com paciente2 hospital)

    return viz


def main() -> None:
    viz = construir()
    res = viz.simular()
    print("Parte 2 — transportador (carrier)")
    print(f"  éxito:    {res.succeeded}")
    print(f"  makespan: {res.makespan}")
    print(f"  fallos:   {len(res.failures)}")
    if "--check" not in sys.argv:
        viz.run()


if __name__ == "__main__":
    main()
