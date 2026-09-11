"""Ejemplo C — Diagnóstico: errores de CONSTRUCCIÓN vs fallos de EJECUCIÓN.

La fachada distingue dos niveles de problema, y conviene que el alumno
sepa dónde aparece cada uno:

  · ERRORES DE CONSTRUCCIÓN: cosas mal declaradas (referenciar algo que no
    existe, cruzar tipos, brazo inexistente, mezclar tiempos, ids
    repetidos). Saltan como EXCEPCIÓN en la propia llamada, antes de
    simular, señalando la línea culpable. Se atrapan con try/except.

  · FALLOS DE EJECUCIÓN: el plan está bien escrito pero es físicamente
    imposible (volar por una arista que no existe, recoger algo que no está
    donde el dron, usar un brazo ya ocupado). NO lanzan excepción: la
    fachada construye el plan y el PlanRunner los reporta en
    RunResult.failures al simular. Así el alumno ve la animación de lo que
    sí funcionó y, a la vez, qué falló y por qué.

Este script provoca varios de cada tipo y los imprime. No abre ventana:
es una herramienta de consola. (Acepta --check por uniformidad con el resto,
pero su comportamiento es el mismo con o sin él.)

Ejecutar:
    python examples/diagnostico_fallos_y_errores.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from droneplan_viz import DronePlanViz, FacadeError  # noqa: E402


def _mundo_basico() -> DronePlanViz:
    """deposito + casa (con arista), una caja en deposito, un dron 2 brazos."""
    viz = DronePlanViz()
    viz.world.location("deposito")
    viz.world.location("casa")
    viz.world.costes({("deposito", "casa"): 8}, simetrico=True)
    viz.world.content("medicina")
    viz.world.person("paciente", at="casa", necesita=["medicina"])
    viz.world.package("caja", contiene="medicina", at="deposito")
    viz.agents.drone("dron1", at="deposito")
    return viz


# ===========================================================================
# 1) Errores de construcción (excepción inmediata)
# ===========================================================================


def errores_de_construccion() -> None:
    print("=" * 64)
    print("ERRORES DE CONSTRUCCIÓN  (saltan como excepción en la llamada)")
    print("=" * 64)

    casos = []

    # a) Referenciar una localización no declarada.
    def caso_a():
        viz = _mundo_basico()
        viz.world.person("otro", at="casa_inexistente")
    casos.append(("persona anclada a localización no declarada", caso_a))

    # b) Cruzar tipos: mover espera una localización en `a`, le damos persona.
    def caso_b():
        viz = _mundo_basico()
        viz.mover("dron1", a="paciente")  # 'paciente' es persona, no location
    casos.append(("mover hacia una persona (se esperaba localización)", caso_b))

    # c) Brazo inexistente en el dron.
    def caso_c():
        viz = _mundo_basico()
        viz.recoger("dron1", caja="caja", brazo="tercer_brazo")
    casos.append(("recoger con un brazo que el dron no tiene", caso_c))

    # d) Mezclar tiempos: una acción con inicio= y otra sin.
    def caso_d():
        viz = _mundo_basico()
        viz.recoger("dron1", caja="caja", brazo="izq", inicio=0.0)
        viz.mover("dron1", a="casa")  # sin inicio -> mezcla prohibida
        viz.build()  # la regla "todo o nada" se comprueba al construir
    casos.append(("mezclar acciones con y sin inicio=", caso_d))

    # e) Dos acciones con el mismo id=.
    def caso_e():
        viz = _mundo_basico()
        viz.recoger("dron1", caja="caja", brazo="izq", id="paso_0")
        viz.mover("dron1", a="casa", id="paso_0")  # id repetido
    casos.append(("reutilizar el mismo id= en dos acciones", caso_e))

    for titulo, fn in casos:
        try:
            fn()
            print(f"  [!] {titulo}: no lanzó (inesperado)")
        except FacadeError as e:
            print(f"  · {titulo}")
            print(f"      -> {type(e).__name__}: {e}")
    print()


# ===========================================================================
# 2) Fallos de ejecución (en RunResult, sin excepción)
# ===========================================================================


def fallos_de_ejecucion() -> None:
    print("=" * 64)
    print("FALLOS DE EJECUCIÓN  (el plan se construye; los reporta simular())")
    print("=" * 64)

    # a) Volar por una arista no declarada (localización válida, sin coste).
    def plan_arista():
        viz = DronePlanViz()
        viz.world.location("deposito")
        viz.world.location("isla")  # declarada, pero SIN arista de coste
        viz.agents.drone("dron1", at="deposito")
        viz.mover("dron1", a="isla")
        return viz

    # b) Recoger una caja que no está donde el dron (no co-localizados).
    def plan_colocalizacion():
        viz = DronePlanViz()
        viz.world.location("deposito")
        viz.world.location("casa")
        viz.world.costes({("deposito", "casa"): 8}, simetrico=True)
        viz.world.content("medicina")
        viz.world.package("caja", contiene="medicina", at="casa")  # en casa
        viz.agents.drone("dron1", at="deposito")  # dron en deposito
        viz.recoger("dron1", caja="caja", brazo="izq")
        return viz

    # c) Ocupar dos veces el mismo brazo sin liberarlo.
    def plan_brazo_ocupado():
        viz = DronePlanViz()
        viz.world.location("deposito")
        viz.world.content("medicina")
        viz.world.content("comida")
        viz.world.package("caja1", contiene="medicina", at="deposito")
        viz.world.package("caja2", contiene="comida", at="deposito")
        viz.agents.drone("dron1", at="deposito")
        viz.recoger("dron1", caja="caja1", brazo="izq")  # ocupa izq (ÉXITO)
        viz.recoger("dron1", caja="caja2", brazo="izq")  # izq ocupado (FALLO)
        return viz

    casos = [
        ("volar por una arista no declarada", plan_arista),
        ("recoger una caja no co-localizada", plan_colocalizacion),
        ("reusar un brazo ya ocupado", plan_brazo_ocupado),
    ]
    for titulo, fn in casos:
        viz = fn()
        res = viz.simular()  # NO lanza: el fallo viaja en el RunResult
        print(f"  · {titulo}")
        print(f"      éxito={res.succeeded}  fallos={len(res.failures)}")
        for f in res.failures:
            print(f"      -> [{f.kind}] {f.reason}")
    print()


def main() -> None:
    errores_de_construccion()
    fallos_de_ejecucion()
    print("Resumen: los errores de construcción se atrapan con try/except;")
    print("los fallos de ejecución se leen en RunResult.failures.")


if __name__ == "__main__":
    main()
