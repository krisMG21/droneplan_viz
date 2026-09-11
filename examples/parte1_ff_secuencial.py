"""Ejemplo 1 — Plan FF (secuencial, atemporal) traducido a la fachada.

Caso más simple de la asignatura: un planificador clásico (FF) resuelve el
problema "lleva la comida a persona1" y produce un plan SIN tiempos, solo
una secuencia de acciones. El alumno traduce ese plan, acción por acción, a
llamadas de DronePlanViz. Como no hay `inicio=`, la fachada entra en modo
SECUENCIAL y encadena los tiempos automáticamente para la animación.

Plan FF de referencia (salida del planificador, formato simplificado):

    0: (coger-caja dron1 caja-comida brazo-izq deposito)
    1: (volar dron1 deposito casa1)
    2: (entregar-caja dron1 caja-comida persona1 casa1)

A la derecha de cada acción de abajo aparece la línea PDDL que traduce.

Ejecutar:
    python examples/parte1_ff_secuencial.py            # abre la ventana
    python examples/parte1_ff_secuencial.py --check    # solo simula y sale
"""
from __future__ import annotations

import sys
from pathlib import Path

# Asegurar la raíz del repo en sys.path para poder
# importar el paquete sin instalarlo.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from droneplan_viz import DronePlanViz  # noqa: E402


def construir() -> DronePlanViz:
    viz = DronePlanViz()

    # --- Mundo (sección :objects + :init del problema PDDL) ---
    viz.world.location("deposito")
    viz.world.location("casa1")
    # La arista de coste ES lo que hace transitable el vuelo deposito<->casa1.
    viz.world.costes({("deposito", "casa1"): 8}, simetrico=True)

    viz.world.content("comida")
    viz.world.person("persona1", at="casa1", necesita=["comida"])
    viz.world.package("caja_comida", contiene="comida", at="deposito")

    # dron1 con los dos brazos por defecto (izq, der).
    viz.agents.drone("dron1", at="deposito")

    # --- Plan FF (sin tiempos -> modo secuencial) ---
    viz.recoger("dron1", caja="caja_comida", brazo="izq")
    #   (coger-caja dron1 caja-comida brazo-izq deposito)
    viz.mover("dron1", a="casa1")
    #   (volar dron1 deposito casa1)
    viz.entregar("dron1", caja="caja_comida", a="persona1")
    #   (entregar-caja dron1 caja-comida persona1 casa1)

    return viz


def main() -> None:
    viz = construir()
    res = viz.simular()
    print("Parte 1 — FF secuencial")
    print(f"  éxito:    {res.succeeded}")
    print(f"  makespan: {res.makespan}")
    print(f"  fallos:   {len(res.failures)}")
    if "--check" not in sys.argv:
        viz.run()


if __name__ == "__main__":
    main()
