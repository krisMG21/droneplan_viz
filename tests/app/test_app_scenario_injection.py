"""Tests del camino de entrada `scenario=` de DroneplanVizApp.

Sesión F añadió a la app un parámetro keyword-only `scenario=(world, plan)`
para que la fachada DronePlanViz pueda abrir la app sobre un escenario
construido al vuelo, sin pasar por el registro SCENARIOS. Estos tests
verifican, en headless (SDL dummy):

  - la app construye sobre un (world, plan) inyectado y lo ejecuta;
  - lo construido por la fachada y lo ejecutado por la app coinciden
    (mismo makespan / mismo veredicto) -> la integración fachada<->app es
    coherente;
  - `scenario=` tiene precedencia sobre `scenario_name`;
  - la guarda de scenario_name desconocido SOLO actúa cuando no se inyecta
    escenario (no rompe el camino antiguo).

El camino por `scenario_name` (CLI y tests de la sesión E) queda idéntico;
eso lo cubre test_app.py y aquí solo reconfirmamos la guarda.
"""
from __future__ import annotations

import pygame
import pytest

from droneplan_viz import DronePlanViz
from droneplan_viz_app.app import DroneplanVizApp


@pytest.fixture(autouse=True)
def cleanup_pygame():
    yield
    pygame.quit()


def _viz_con_plan_correcto() -> DronePlanViz:
    """Fachada con un plan que SÍ ejecuta sin fallos (incluye la arista de
    coste por la que vuela el dron: sin ella el Move fallaría)."""
    viz = DronePlanViz()
    viz.world.location("deposito")
    viz.world.location("casa1")
    viz.world.costes({("deposito", "casa1"): 8}, simetrico=True)
    viz.world.content("comida")
    viz.world.person("p1", at="casa1", necesita=["comida"])
    viz.world.package("caja1", contiene="comida", at="deposito")
    viz.agents.drone("dron1", at="deposito")
    viz.recoger("dron1", caja="caja1", brazo="izq", inicio=0.0, duracion=5)
    viz.mover("dron1", a="casa1", inicio=5.0, duracion=8)
    viz.entregar("dron1", caja="caja1", a="p1", inicio=13.0, duracion=5)
    return viz


class TestScenarioInjection:
    def test_app_construye_sobre_escenario_inyectado(self):
        viz = _viz_con_plan_correcto()
        world, plan = viz.build()
        app = DroneplanVizApp(scenario=(world, plan), window_size=(900, 600))
        assert app.timeline.duration > 0
        assert app._result.succeeded is True

    def test_resultado_de_la_app_coincide_con_la_fachada(self):
        """La app ejecuta exactamente el plan que la fachada construyó."""
        viz = _viz_con_plan_correcto()
        world, plan = viz.build()
        esperado = viz.simular()
        app = DroneplanVizApp(scenario=(world, plan), window_size=(900, 600))
        assert app._result.makespan == esperado.makespan
        assert app._result.succeeded == esperado.succeeded

    def test_scenario_tiene_precedencia_sobre_scenario_name(self):
        """Con scenario= presente, un scenario_name basura se ignora (no se
        consulta el registro), en vez de lanzar."""
        viz = _viz_con_plan_correcto()
        world, plan = viz.build()
        app = DroneplanVizApp(
            scenario=(world, plan), scenario_name="no_existe_xyz"
        )
        assert app.timeline.duration > 0

    def test_guarda_solo_actua_sin_escenario(self):
        """Sin scenario y con scenario_name desconocido -> ValueError; la
        guarda no se debilita por el nuevo parámetro."""
        with pytest.raises(ValueError, match="desconocido"):
            DroneplanVizApp(scenario_name="no_existe_xyz")

    def test_camino_demo_por_nombre_intacto(self):
        app = DroneplanVizApp(scenario_name="demo")
        assert app._result.makespan == 18.0
