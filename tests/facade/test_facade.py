"""Tests de droneplan_viz.facade.facade (DronePlanViz).

Cubren la superficie pública de la fachada:

  - cableado de builders (viz.world, viz.agents) sobre estado compartido;
  - encolado de acciones y construcción de los Commands correctos;
  - resolución string/referencia en las acciones;
  - equivalencia de los alias castellano/inglés;
  - guarda cruzada del parámetro `a` (mover<->localización, entregar<->persona);
  - validación de brazo (existencia, incluido dron explorador);
  - asignación de command_id determinista y `id=` explícito + duplicados;
  - regla de tiempos "todo o nada" y encadenado secuencial sin solape;
  - build() en headless y simular() delegando la física al PlanRunner.

Sin pygame: todo es build()/simular(), nunca run().
"""
from __future__ import annotations

import pytest

from droneplan_viz import DronePlanViz
from droneplan_viz.commands import (
    Deliver,
    LoadIntoTransporter,
    Move,
    MoveWithTransporter,
    PickUp,
    UnloadFromTransporter,
)
from droneplan_viz.facade.agent_builder import AgentBuilder
from droneplan_viz.facade.errors import (
    DuplicateIdError,
    FacadeError,
    MixedTimingError,
    UnknownArmError,
    UnknownDroneError,
    UnknownPackageError,
    UnknownTransporterError,
    WrongReferenceKindError,
)
from droneplan_viz.facade.world_builder import WorldBuilder
from droneplan_viz.runtime import Plan, RunResult


def _viz_basico() -> DronePlanViz:
    """Fachada con un mundo mínimo: deposito+casa1, una caja, una persona,
    un dron de dos brazos y un transportador. Suficiente para casi todas
    las acciones sin gimnasia previa."""
    viz = DronePlanViz()
    viz.world.location("deposito")
    viz.world.location("casa1")
    viz.world.costes({("deposito", "casa1"): 8}, simetrico=True)
    viz.world.content("comida")
    viz.world.person("p1", at="casa1", necesita=["comida"])
    viz.world.package("caja1", contiene="comida", at="deposito")
    viz.agents.drone("dron1", at="deposito")
    viz.agents.transporter("t1", capacidad=4, at="deposito")
    return viz


# ===========================================================================
# Cableado
# ===========================================================================


class TestCableado:
    def test_world_y_agents_son_builders(self):
        viz = DronePlanViz()
        assert isinstance(viz.world, WorldBuilder)
        assert isinstance(viz.agents, AgentBuilder)

    def test_builders_comparten_el_mismo_state(self):
        """world y agents mutan el mismo _FacadeState."""
        viz = DronePlanViz()
        viz.world.location("base")
        d = viz.agents.drone("dron1", at="base")  # ve la location de world
        assert d.position == "base"

    def test_importable_desde_la_raiz(self):
        # El propio import al principio del módulo ya lo prueba; lo
        # reafirmamos como contrato de superficie pública.
        from droneplan_viz import DronePlanViz as D

        assert D is DronePlanViz


# ===========================================================================
# Acciones: construyen el Command correcto
# ===========================================================================


class TestAccionesCommands:
    def test_mover_construye_move(self):
        viz = _viz_basico()
        viz.mover("dron1", a="casa1")
        _w, plan = viz.build()
        (sc,) = plan.scheduled
        assert isinstance(sc.command, Move)
        assert sc.command.drone_id == "dron1"
        assert sc.command.destination_id == "casa1"

    def test_mover_con_transportador_construye_move_with_transporter(self):
        viz = _viz_basico()
        viz.mover("dron1", a="casa1", con="t1")
        _w, plan = viz.build()
        (sc,) = plan.scheduled
        assert isinstance(sc.command, MoveWithTransporter)
        assert sc.command.transporter_id == "t1"
        assert sc.command.destination_id == "casa1"

    def test_recoger_construye_pickup_con_brazo(self):
        viz = _viz_basico()
        viz.recoger("dron1", caja="caja1", brazo="izq")
        _w, plan = viz.build()
        (sc,) = plan.scheduled
        assert isinstance(sc.command, PickUp)
        assert sc.command.arm_id == "izq"
        assert sc.command.package_id == "caja1"

    def test_entregar_construye_deliver_sin_brazo(self):
        viz = _viz_basico()
        viz.entregar("dron1", caja="caja1", a="p1")
        _w, plan = viz.build()
        (sc,) = plan.scheduled
        assert isinstance(sc.command, Deliver)
        assert sc.command.package_id == "caja1"
        assert sc.command.person_id == "p1"

    def test_poner_en_construye_load(self):
        viz = _viz_basico()
        viz.poner_en("dron1", caja="caja1", transportador="t1")
        _w, plan = viz.build()
        (sc,) = plan.scheduled
        assert isinstance(sc.command, LoadIntoTransporter)
        assert sc.command.transporter_id == "t1"

    def test_sacar_de_construye_unload_con_brazo(self):
        viz = _viz_basico()
        viz.sacar_de("dron1", caja="caja1", transportador="t1", brazo="der")
        _w, plan = viz.build()
        (sc,) = plan.scheduled
        assert isinstance(sc.command, UnloadFromTransporter)
        assert sc.command.arm_id == "der"
        assert sc.command.transporter_id == "t1"

    def test_orden_de_encolado_se_preserva(self):
        viz = _viz_basico()
        viz.recoger("dron1", caja="caja1", brazo="izq")
        viz.mover("dron1", a="casa1")
        viz.entregar("dron1", caja="caja1", a="p1")
        _w, plan = viz.build()
        tipos = [type(sc.command) for sc in plan.scheduled]
        assert tipos == [PickUp, Move, Deliver]


# ===========================================================================
# Resolución string vs referencia
# ===========================================================================


class TestResolucionReferencia:
    def test_str_y_referencia_producen_el_mismo_plan(self):
        # Por string.
        v1 = _viz_basico()
        v1.mover("dron1", a="casa1")
        # Por referencia: usamos los handles devueltos por los builders.
        v2 = DronePlanViz()
        v2.world.location("deposito")
        casa1 = v2.world.location("casa1")
        dron1 = v2.agents.drone("dron1", at="deposito")
        v2.mover(dron1, a=casa1)
        c1 = v1.build()[1].scheduled[0].command
        c2 = v2.build()[1].scheduled[0].command
        assert (c1.drone_id, c1.destination_id) == (
            c2.drone_id,
            c2.destination_id,
        )


# ===========================================================================
# Alias castellano/inglés
# ===========================================================================


class TestAlias:
    @pytest.mark.parametrize(
        "es, en",
        [
            ("mover", "move"),
            ("recoger", "grab"),
            ("entregar", "deliver"),
            ("poner_en", "load_into"),
            ("sacar_de", "unload_from"),
        ],
    )
    def test_alias_es_el_mismo_callable(self, es, en):
        assert getattr(DronePlanViz, es) is getattr(DronePlanViz, en)

    def test_simulate_es_simular(self):
        assert DronePlanViz.simulate is DronePlanViz.simular

    def test_grab_produce_lo_mismo_que_recoger(self):
        v1 = _viz_basico()
        v1.recoger("dron1", caja="caja1", brazo="izq")
        v2 = _viz_basico()
        v2.grab("dron1", caja="caja1", brazo="izq")
        c1 = v1.build()[1].scheduled[0].command
        c2 = v2.build()[1].scheduled[0].command
        assert type(c1) is type(c2)
        assert (c1.drone_id, c1.arm_id, c1.package_id) == (
            c2.drone_id,
            c2.arm_id,
            c2.package_id,
        )


# ===========================================================================
# Guarda cruzada del parámetro `a`
# ===========================================================================


class TestGuardaCruzada:
    def test_mover_hacia_persona_lanza_wrong_kind(self):
        viz = _viz_basico()  # p1 es persona, no localización
        with pytest.raises(WrongReferenceKindError) as exc:
            viz.mover("dron1", a="p1")
        assert exc.value.metodo == "mover"
        assert exc.value.recibido == "la persona"
        assert "espera una localización" in str(exc.value)

    def test_entregar_a_localizacion_lanza_wrong_kind(self):
        viz = _viz_basico()  # casa1 es localización, no persona
        with pytest.raises(WrongReferenceKindError) as exc:
            viz.entregar("dron1", caja="caja1", a="casa1")
        assert exc.value.metodo == "entregar"
        assert exc.value.recibido == "la localización"

    def test_mover_destino_inexistente_lanza_unknown_location(self):
        from droneplan_viz.facade.errors import UnknownLocationError

        viz = _viz_basico()
        with pytest.raises(UnknownLocationError):
            viz.mover("dron1", a="casa9")

    def test_entregar_persona_inexistente_lanza_unknown_person(self):
        from droneplan_viz.facade.errors import UnknownPersonError

        viz = _viz_basico()
        with pytest.raises(UnknownPersonError):
            viz.entregar("dron1", caja="caja1", a="pZ")


# ===========================================================================
# Validación de brazo
# ===========================================================================


class TestBrazo:
    def test_brazo_inexistente_lanza_unknown_arm(self):
        viz = _viz_basico()  # dron1 tiene izq, der
        with pytest.raises(UnknownArmError) as exc:
            viz.recoger("dron1", caja="caja1", brazo="central")
        assert exc.value.brazos_disponibles == ("izq", "der")

    def test_dron_explorador_no_puede_recoger(self):
        viz = DronePlanViz()
        viz.world.location("deposito")
        viz.world.content("comida")
        viz.world.package("caja1", contiene="comida", at="deposito")
        viz.agents.drone("explorador", at="deposito", arms=[])
        with pytest.raises(UnknownArmError) as exc:
            viz.recoger("explorador", caja="caja1", brazo="izq")
        assert exc.value.brazos_disponibles == ()
        assert "explorador" in str(exc.value)

    def test_sacar_de_tambien_valida_brazo(self):
        viz = _viz_basico()
        with pytest.raises(UnknownArmError):
            viz.sacar_de("dron1", caja="caja1", transportador="t1", brazo="x")


# ===========================================================================
# Referencias inexistentes en acciones
# ===========================================================================


class TestReferenciasAccion:
    def test_dron_inexistente(self):
        viz = _viz_basico()
        with pytest.raises(UnknownDroneError):
            viz.mover("dronZ", a="casa1")

    def test_caja_inexistente(self):
        viz = _viz_basico()
        with pytest.raises(UnknownPackageError):
            viz.recoger("dron1", caja="cajaZ", brazo="izq")

    def test_transportador_inexistente(self):
        viz = _viz_basico()
        with pytest.raises(UnknownTransporterError):
            viz.poner_en("dron1", caja="caja1", transportador="tZ")


# ===========================================================================
# command_id: deterministas y explícitos
# ===========================================================================


class TestCommandIds:
    def test_ids_deterministas_por_verbo_y_posicion(self):
        viz = _viz_basico()
        viz.recoger("dron1", caja="caja1", brazo="izq")
        viz.mover("dron1", a="casa1")
        viz.entregar("dron1", caja="caja1", a="p1")
        _w, plan = viz.build()
        ids = [sc.command.command_id for sc in plan.scheduled]
        assert ids == ["recoger_1", "mover_2", "entregar_3"]

    def test_ids_estables_entre_dos_construcciones(self):
        """Dos escenarios idénticos -> mismos ids (pureza, robustez)."""
        v1 = _viz_basico()
        v1.mover("dron1", a="casa1")
        v2 = _viz_basico()
        v2.mover("dron1", a="casa1")
        id1 = v1.build()[1].scheduled[0].command.command_id
        id2 = v2.build()[1].scheduled[0].command.command_id
        assert id1 == id2 == "mover_1"

    def test_id_explicito_se_usa_verbatim(self):
        viz = _viz_basico()
        viz.recoger("dron1", caja="caja1", brazo="izq", id="coger_caja_brazo1_0")
        _w, plan = viz.build()
        assert plan.scheduled[0].command.command_id == "coger_caja_brazo1_0"

    def test_id_explicito_duplicado_lanza_en_la_llamada(self):
        viz = _viz_basico()
        viz.recoger("dron1", caja="caja1", brazo="izq", id="paso0")
        with pytest.raises(DuplicateIdError):
            viz.mover("dron1", a="casa1", id="paso0")

    def test_id_explicito_choca_con_automatico_en_build(self):
        """id='mover_2' en la acción 1 colisiona con el automático de la 2."""
        viz = _viz_basico()
        viz.mover("dron1", a="casa1", id="mover_2")  # acción 1
        viz.mover("dron1", a="deposito")  # acción 2 -> auto "mover_2"
        with pytest.raises(DuplicateIdError):
            viz.build()


# ===========================================================================
# Tiempos: regla "todo o nada"
# ===========================================================================


class TestTiempos:
    def test_secuencial_encadena_sin_solape(self):
        viz = _viz_basico()
        viz.recoger("dron1", caja="caja1", brazo="izq", duracion=5)
        viz.mover("dron1", a="casa1", duracion=8)
        viz.entregar("dron1", caja="caja1", a="p1", duracion=5)
        _w, plan = viz.build()
        starts = [sc.start_time for sc in plan.scheduled]
        assert starts == [0.0, 5.0, 13.0]

    def test_sin_duracion_usa_default_animable_y_tiempos_0_1_2(self):
        """Sin `duracion=` (plan FF) cada acción recibe una duración por
        defecto > 0 para que SIEMPRE se anime (vuelo + coreografía), y los
        tiempos de inicio quedan encadenados 0, 1, 2, … sin solape."""
        viz = _viz_basico()
        viz.recoger("dron1", caja="caja1", brazo="izq")
        viz.mover("dron1", a="casa1")
        viz.entregar("dron1", caja="caja1", a="p1")
        _w, plan = viz.build()
        starts = [sc.start_time for sc in plan.scheduled]
        assert starts == [0.0, 1.0, 2.0]
        # Lo esencial: ninguna acción es instantánea (duration > 0), porque
        # una duración 0 impediría la animación de movimiento.
        for sc in plan.scheduled:
            assert sc.command.duration > 0.0

    def test_duracion_sin_inicio_se_respeta_en_el_command(self):
        """duracion sin inicio es válida: alimenta la animación."""
        viz = _viz_basico()
        viz.mover("dron1", a="casa1", duracion=7)
        _w, plan = viz.build()
        assert plan.scheduled[0].command.duration == 7.0

    def test_temporal_usa_tiempos_absolutos(self):
        viz = _viz_basico()
        viz.recoger("dron1", caja="caja1", brazo="izq", inicio=0.0, duracion=5)
        viz.mover("dron1", a="casa1", inicio=5.0, duracion=8)
        _w, plan = viz.build()
        starts = [sc.start_time for sc in plan.scheduled]
        assert starts == [0.0, 5.0]

    def test_mixto_falta_inicio_lanza(self):
        viz = _viz_basico()
        viz.mover("dron1", a="casa1", inicio=0.0)  # con inicio
        viz.mover("dron1", a="deposito")  # sin inicio
        with pytest.raises(MixedTimingError) as exc:
            viz.build()
        assert exc.value.falta_inicio is True
        assert "todas" in str(exc.value) and "ninguna" in str(exc.value)

    def test_mixto_sobra_inicio_lanza(self):
        viz = _viz_basico()
        viz.mover("dron1", a="casa1")  # sin inicio
        viz.mover("dron1", a="deposito", inicio=5.0)  # con inicio
        with pytest.raises(MixedTimingError) as exc:
            viz.build()
        assert exc.value.falta_inicio is False

    def test_inicio_negativo_lanza(self):
        viz = _viz_basico()
        with pytest.raises(FacadeError):
            viz.mover("dron1", a="casa1", inicio=-1.0)

    def test_duracion_negativa_lanza(self):
        viz = _viz_basico()
        with pytest.raises(FacadeError):
            viz.mover("dron1", a="casa1", duracion=-2.0)


# ===========================================================================
# build / simular
# ===========================================================================


class TestBuild:
    def test_build_devuelve_world_y_plan(self):
        viz = _viz_basico()
        viz.mover("dron1", a="casa1")
        world, plan = viz.build()
        assert isinstance(plan, Plan)
        assert set(world.locations) == {"deposito", "casa1"}
        assert set(world.drones) == {"dron1"}

    def test_build_plan_vacio_es_valido(self):
        viz = _viz_basico()  # sin acciones
        _w, plan = viz.build()
        assert len(plan.scheduled) == 0

    def test_world_construido_tiene_costes_simetricos(self):
        viz = _viz_basico()
        world, _plan = viz.build()
        assert world.costs[("deposito", "casa1")] == 8.0
        assert world.costs[("casa1", "deposito")] == 8.0


class TestSimular:
    def test_simular_devuelve_runresult(self):
        viz = _viz_basico()
        viz.recoger("dron1", caja="caja1", brazo="izq", duracion=5)
        viz.mover("dron1", a="casa1", duracion=8)
        viz.entregar("dron1", caja="caja1", a="p1", duracion=5)
        res = viz.simular()
        assert isinstance(res, RunResult)
        assert res.succeeded is True
        assert res.failures == ()

    def test_simular_delega_la_fisica_al_runner(self):
        """Un plan físicamente imposible NO lanza excepción en la fachada:
        el fallo aparece en RunResult.failures, vía el PlanRunner."""
        viz = _viz_basico()  # dron1 y caja1 en deposito; p1 en casa1
        # Entregar sin haber recogido: el dron no sostiene la caja -> fallo
        # PDDL del runner, no excepción de construcción.
        viz.entregar("dron1", caja="caja1", a="p1")
        res = viz.simular()
        assert res.succeeded is False
        assert len(res.failures) == 1

    def test_simular_equivale_a_build_mas_runner(self):
        from droneplan_viz.runtime import PlanRunner

        viz = _viz_basico()
        viz.mover("dron1", a="casa1", duracion=8)
        world, plan = viz.build()
        esperado = PlanRunner(world).execute(plan)
        obtenido = viz.simular()
        assert obtenido.succeeded == esperado.succeeded
        assert obtenido.makespan == esperado.makespan


# ---------------------------------------------------------------------------
# colores_contenido: paleta de color por tipo de contenido (semilla fija →
# reproducible). La app la inyecta en el theme para el outline de las cajas.
# ---------------------------------------------------------------------------
class TestColoresContenido:
    def _viz(self, contenidos):
        viz = DronePlanViz()
        for c in contenidos:
            viz.world.content(c)
        return viz

    def test_una_color_por_contenido(self):
        pal = self._viz(["agua", "comida", "medicina"]).colores_contenido()
        assert set(pal) == {"agua", "comida", "medicina"}
        for rgb in pal.values():
            assert len(rgb) == 3 and all(0 <= c <= 255 for c in rgb)

    def test_claves_en_minuscula(self):
        pal = self._viz(["Agua", "COMIDA"]).colores_contenido()
        assert set(pal) == {"agua", "comida"}

    def test_semilla_fija_es_reproducible(self):
        a = self._viz(["agua", "comida"]).colores_contenido()
        b = self._viz(["agua", "comida"]).colores_contenido()
        assert a == b  # mismo seed por defecto → mismo color en cada run

    def test_no_depende_del_orden_de_declaracion(self):
        a = self._viz(["agua", "comida"]).colores_contenido()
        b = self._viz(["comida", "agua"]).colores_contenido()
        assert a == b  # se recorre por id ordenado

    def test_semilla_distinta_cambia_la_paleta(self):
        viz = self._viz(["agua", "comida", "medicina"])
        assert viz.colores_contenido(seed=1) != viz.colores_contenido(seed=2)

    def test_alias_ingles(self):
        viz = self._viz(["agua"])
        assert viz.content_colors() == viz.colores_contenido()
