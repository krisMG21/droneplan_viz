"""Ejemplo A — Configuraciones de la API de la fachada.

No introduce un caso PDDL nuevo: reúne en un escenario pequeño varias
FORMAS de usar la fachada que los ejemplos `parteN` no muestran, para que
se vean juntas:

  1. Identificar por REFERENCIA en vez de por id: las acciones aceptan tanto
     el string del id ("dron1") como el objeto que devolvió el builder. Aquí
     guardamos los handles (deposito, casa, repartidor, caja...) y los
     pasamos directamente.

  2. ALIAS en inglés: cada acción tiene su alias (move/grab/deliver/
     load_into/unload_from). Mezclamos castellano e inglés a propósito para
     enseñar que son el mismo método.

  3. `at_screen=`: fijar la posición en pantalla de una localización en vez
     de dejar el layout automático. Es solo una pista visual; no cambia la
     lógica (el espacio sigue siendo un grafo).

  4. Dron MONOBRAZO (`arms=["pinza"]`) frente al dron de dos brazos por
     defecto, y dron EXPLORADOR (`arms=[]`) que solo puede moverse.

El plan: un repartidor monobrazo lleva una caja del depósito a la casa,
mientras (en la línea secuencial) un explorador patrulla depósito → mirador
→ casa. El explorador no recoge ni entrega: arms=[] solo permite volar.

Ejecutar:
    python examples/api_referencias_y_alias.py            # abre la ventana
    python examples/api_referencias_y_alias.py --check    # solo simula y sale
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from droneplan_viz import DronePlanViz  # noqa: E402


def construir() -> DronePlanViz:
    viz = DronePlanViz()

    # --- Mundo, capturando los handles para usarlos POR REFERENCIA ---
    # at_screen fija la posición en pantalla (pista visual, no lógica).
    deposito = viz.world.location("deposito", at_screen=(200, 300))
    mirador = viz.world.location("mirador", at_screen=(450, 150))
    casa = viz.world.location("casa", at_screen=(700, 300))

    # Grafo: deposito-casa (ruta de reparto) y el triángulo de patrulla.
    viz.world.costes(
        {
            ("deposito", "casa"): 8,
            ("deposito", "mirador"): 5,
            ("mirador", "casa"): 6,
        },
        simetrico=True,
    )

    suministro = viz.world.content("suministro")
    vecino = viz.world.person("vecino", at=casa, necesita=[suministro])
    caja = viz.world.package("caja", contiene=suministro, at=deposito)

    # Dron MONOBRAZO (una sola pinza) y dron EXPLORADOR (sin brazos).
    repartidor = viz.agents.drone("repartidor", at=deposito, arms=["pinza"])
    explorador = viz.agents.drone("explorador", at=deposito, arms=[])

    # --- Plan (secuencial) ---
    # Reparto, identificando todo POR REFERENCIA y con ALIAS en inglés:
    viz.grab(repartidor, caja=caja, brazo="pinza")     # grab == recoger
    viz.move(repartidor, a=casa)                        # move == mover
    viz.deliver(repartidor, caja=caja, a=vecino)        # deliver == entregar

    # Patrulla del explorador: solo vuela (arms=[] no permite recoger).
    viz.mover(explorador, a=mirador)
    viz.mover(explorador, a=casa)
    viz.mover(explorador, a=deposito)

    return viz


def main() -> None:
    viz = construir()
    res = viz.simular()
    print("Ejemplo A — características de la API")
    print(f"  éxito:    {res.succeeded}")
    print(f"  makespan: {res.makespan}")
    print(f"  fallos:   {len(res.failures)}")
    if "--check" not in sys.argv:
        viz.run()


if __name__ == "__main__":
    main()
