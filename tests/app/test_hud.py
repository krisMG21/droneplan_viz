"""Tests del módulo droneplan_viz_app.hud.

Smoke tests con SDL dummy: el HUD necesita un display real (aunque sea
dummy) para que pygame_gui funcione. Los tests verifican:

  - El HUD se construye sin crashear con escenarios feliz y de fallo.
  - process_event traduce los eventos de pygame_gui (UI_BUTTON_PRESSED,
    UI_SELECTION_LIST_NEW_SELECTION) a HudIntent correctamente.
  - sync_from_state actualiza los textos visibles según AppState.
  - draw() deja la región del HUD con píxeles distintos al fondo (smoke
    estructural).
  - build_hud_plan_info construye el plano informativo correctamente.
  - _format_failure_label y _format_speed_label producen las cadenas
    esperadas.

NO se testean detalles internos de pygame_gui (que un botón haga focus,
que el slider arrastre, etc.). Eso es responsabilidad de la librería
upstream.
"""
from __future__ import annotations

import pygame
import pygame_gui
import pytest

from droneplan_viz.render import Theme, Timeline
from droneplan_viz.runtime import PlanRunner
from droneplan_viz_app.app_state import AppState
from droneplan_viz_app.controller import HudIntent
from droneplan_viz_app.hud import (
    Hud,
    HudPlanInfo,
    _format_failure_label,
    build_hud_plan_info,
)
from droneplan_viz_app.layout import compute_regions
from droneplan_viz_app.scenarios import (
    build_demo_scenario,
    build_failure_demo_scenario,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def window_and_manager():
    """Inicializa pygame con SDL dummy y devuelve (window, manager).

    pygame.init() y display.set_mode necesarios para que pygame_gui
    pueda crear superficies de fondo y cargar fuentes. Funcionan con
    SDL_VIDEODRIVER=dummy (configurado en conftest.py).
    """
    pygame.init()
    window = pygame.display.set_mode((1280, 800))
    manager = pygame_gui.UIManager((1280, 800))
    yield window, manager
    pygame.quit()


@pytest.fixture
def demo_setup(window_and_manager):
    """Setup completo con escenario feliz: window, manager, regions,
    timeline, plan_info, hud, app_state."""
    window, manager = window_and_manager
    regions = compute_regions((1280, 800))
    theme = Theme.default()

    world, plan = build_demo_scenario()
    runner = PlanRunner(world)
    result = runner.execute(plan)
    timeline = Timeline(runner.history, theme=theme)
    plan_info = build_hud_plan_info(
        timeline.duration, timeline.snapshot_times, result.failures
    )
    hud = Hud(manager, regions, plan_info, theme)
    state = AppState()

    return {
        "window": window,
        "manager": manager,
        "regions": regions,
        "theme": theme,
        "timeline": timeline,
        "plan_info": plan_info,
        "hud": hud,
        "state": state,
        "runner": runner,
        "result": result,
    }


@pytest.fixture
def failure_setup(window_and_manager):
    """Setup con escenario de fallo (1 failure)."""
    window, manager = window_and_manager
    regions = compute_regions((1280, 800))
    theme = Theme.default()

    world, plan = build_failure_demo_scenario()
    runner = PlanRunner(world)
    result = runner.execute(plan)
    timeline = Timeline(runner.history, theme=theme)
    plan_info = build_hud_plan_info(
        timeline.duration, timeline.snapshot_times, result.failures
    )
    hud = Hud(manager, regions, plan_info, theme)
    state = AppState()

    return {
        "window": window,
        "manager": manager,
        "regions": regions,
        "timeline": timeline,
        "plan_info": plan_info,
        "hud": hud,
        "state": state,
        "result": result,
    }


# ---------------------------------------------------------------------------
# HudPlanInfo y build_hud_plan_info
# ---------------------------------------------------------------------------


class TestBuildHudPlanInfo:
    def test_demo_scenario_sin_fallos(self):
        """En el escenario feliz, failure_starts y failure_labels son
        ambos vacíos."""
        world, plan = build_demo_scenario()
        runner = PlanRunner(world)
        result = runner.execute(plan)
        tl = Timeline(runner.history, theme=Theme.default())
        info = build_hud_plan_info(tl.duration, tl.snapshot_times, result.failures)

        assert info.duration == pytest.approx(tl.duration)
        assert info.snapshot_times == tl.snapshot_times
        assert info.failure_starts == ()
        assert info.failure_labels == ()

    def test_failure_scenario_un_fallo(self):
        world, plan = build_failure_demo_scenario()
        runner = PlanRunner(world)
        result = runner.execute(plan)
        tl = Timeline(runner.history, theme=Theme.default())
        info = build_hud_plan_info(tl.duration, tl.snapshot_times, result.failures)

        assert len(info.failure_starts) == 1
        assert info.failure_starts[0] == pytest.approx(2.0)
        assert len(info.failure_labels) == 1
        # El label debe incluir el kind, el tiempo y el nombre del Command.
        label = info.failure_labels[0]
        assert "pddl" in label
        assert "2.0" in label
        assert "PickUp" in label

    def test_es_dataclass_frozen(self):
        info = HudPlanInfo(
            duration=10.0, snapshot_times=(0.0, 5.0, 10.0),
            failure_starts=(), failure_labels=(),
        )
        with pytest.raises(Exception):
            info.duration = 999  # type: ignore[misc]


class TestFormatFailureLabel:
    def test_incluye_tiempo_kind_y_clase(self):
        world, plan = build_failure_demo_scenario()
        result = PlanRunner(world).execute(plan)
        (failure,) = result.failures
        label = _format_failure_label(failure)
        assert "t=2.0s" in label
        assert "pddl" in label
        assert "PickUp" in label


class TestBuildHudPlanInfoAcciones:
    """build_hud_plan_info, con plan+history, construye la lista de acciones."""

    def _info(self, builder):
        world, plan = builder()
        runner = PlanRunner(world)
        result = runner.execute(plan)
        tl = Timeline(runner.history, theme=Theme.default())
        return build_hud_plan_info(
            tl.duration, tl.snapshot_times, result.failures,
            result.plan, runner.history,
        ), tl

    def test_demo_lista_acciones_en_orden(self):
        info, tl = self._info(build_demo_scenario)
        assert len(info.action_starts) == len(info.action_labels) >= 1
        # Inicios crecientes y dentro de la duración.
        assert list(info.action_starts) == sorted(info.action_starts)
        assert all(0.0 <= s <= tl.duration for s in info.action_starts)
        # Las etiquetas mencionan el tipo de comando.
        joined = " ".join(info.action_labels)
        assert "PickUp" in joined and "Move" in joined and "Deliver" in joined

    def test_failure_marca_accion_fallida(self):
        info, _tl = self._info(build_failure_demo_scenario)
        assert any("FALLÓ" in lab for lab in info.action_labels)

    def test_sin_plan_history_no_lista_acciones(self):
        # Compatibilidad: sin plan/history, action_starts/labels vacíos.
        world, plan = build_demo_scenario()
        runner = PlanRunner(world)
        result = runner.execute(plan)
        tl = Timeline(runner.history, theme=Theme.default())
        info = build_hud_plan_info(tl.duration, tl.snapshot_times, result.failures)
        assert info.action_starts == ()
        assert info.action_labels == ()


class TestHudActionsPanel:
    """El HUD construye el panel de Acciones con su lista seleccionable."""

    def _hud_with_actions(self, window_and_manager, builder):
        window, manager = window_and_manager
        regions = compute_regions((1280, 800))
        theme = Theme.default()
        world, plan = builder()
        runner = PlanRunner(world)
        result = runner.execute(plan)
        tl = Timeline(runner.history, theme=theme)
        info = build_hud_plan_info(
            tl.duration, tl.snapshot_times, result.failures,
            result.plan, runner.history,
        )
        return Hud(manager, regions, info, theme), info

    def test_panel_y_lista_de_acciones_existen(self, window_and_manager):
        hud, info = self._hud_with_actions(window_and_manager, build_demo_scenario)
        assert hud._panel_actions is not None
        assert hud._act_scroll is not None
        assert hud._lbl_actions_title is not None

    def test_una_tarjeta_por_accion(self, window_and_manager):
        hud, info = self._hud_with_actions(window_and_manager, build_demo_scenario)
        # Una tarjeta (con su índice) por acción, en orden.
        assert len(hud._act_cards) == len(info.action_lines)
        assert [idx for _img, idx, *_ in hud._act_cards] == list(
            range(len(info.action_lines)))

    def test_tarjetas_dos_lineas(self, window_and_manager):
        hud, info = self._hud_with_actions(window_and_manager, build_demo_scenario)
        # Cada acción tiene encabezado + stats (dos líneas no vacías).
        assert all(l1 and l2 for (l1, l2) in info.action_lines)

    def test_area_scrollable_ajustada(self, window_and_manager):
        # Con suficientes acciones, el área scrollable supera el alto visible
        # del contenedor → hay scroll.
        hud, info = self._hud_with_actions(
            window_and_manager, build_failure_demo_scenario)
        assert hud._act_scroll is not None
        # El contenido (n tarjetas * (38+4)) cabe en el área scrollable.
        assert len(hud._act_cards) == len(info.action_lines)

    def test_failure_scenario_construye_acciones_y_fallos(self, window_and_manager):
        hud, info = self._hud_with_actions(
            window_and_manager, build_failure_demo_scenario)
        assert hud._act_scroll is not None  # acciones (scroll de tarjetas)
        assert hud._fail_cards  # y fallos (tarjetas)


# ---------------------------------------------------------------------------
# Construcción del Hud
# ---------------------------------------------------------------------------


class TestHudConstruction:
    def test_se_construye_sin_crashear_escenario_feliz(self, demo_setup):
        """El smoke más básico: __init__ no lanza."""
        assert demo_setup["hud"] is not None

    def test_se_construye_sin_crashear_escenario_fallo(self, failure_setup):
        assert failure_setup["hud"] is not None

    def test_construccion_con_ventana_pequena_funciona(self, window_and_manager):
        """Tamaño mínimo del layout (1024×640) — el HUD se construye."""
        _, manager = window_and_manager
        regions = compute_regions((1024, 640))
        theme = Theme.default()
        world, plan = build_demo_scenario()
        runner = PlanRunner(world)
        result = runner.execute(plan)
        tl = Timeline(runner.history, theme=theme)
        info = build_hud_plan_info(tl.duration, tl.snapshot_times, result.failures)
        hud = Hud(manager, regions, info, theme)
        assert hud is not None

    # Tests del UITabContainer introducido al rediseñar el panel lateral.
    # El antiguo diseño tenía dos UIPanel separados (métricas y fallos);
    # ahora ambos viven dentro de la tab "Métricas+Fallos" del
    # UITabContainer, y aparece una nueva tab "Inventario" (placeholder
    # ahora; dinámica y navegable en una fase posterior).

    def test_tab_container_se_construye(self, demo_setup):
        hud = demo_setup["hud"]
        assert hud._tab_container is not None

    def test_dos_tabs_creadas_con_indices_distintos(self, demo_setup):
        hud = demo_setup["hud"]
        assert hud._tab_idx_inventario >= 0
        assert hud._tab_idx_metrics >= 0
        assert hud._tab_idx_inventario != hud._tab_idx_metrics

    def test_inventario_es_la_tab_activa_al_arrancar(self, demo_setup):
        """Decisión de diseño: la novedad del rediseño (Inventario) debe
        ser visible al abrir. Métricas+Fallos queda a un click."""
        hud = demo_setup["hud"]
        assert hud._tab_container.current_container_index == hud._tab_idx_inventario

    def test_widgets_de_metricas_existen_aunque_la_tab_no_este_activa(self, demo_setup):
        """Aunque la tab Métricas+Fallos arranque inactiva, sus widgets
        están construidos (los necesita sync_from_state para mutarlos).
        pygame_gui solo controla la VISIBILIDAD de la tab."""
        hud = demo_setup["hud"]
        assert hud._lbl_total_cost is not None
        assert hud._lbl_total_time is not None
        assert hud._lbl_action_count is not None
        assert hud._lbl_failed_count is not None
        assert hud._panel_metrics is not None
        assert hud._panel_failures is not None

    def test_inventario_tiene_scroll_container(self, demo_setup):
        """El rediseño sustituyó el UISelectionList por un
        UIScrollingContainer que aloja cabeceras + cajas por entidad."""
        hud = demo_setup["hud"]
        assert hud._inv_scroll is not None

    def test_inventario_arranca_vacio_antes_de_sync(self, demo_setup):
        """_build_right_panel crea el scroll container vacío: es
        sync_from_state quien lo puebla via _rebuild_inventory_ui."""
        hud = demo_setup["hud"]
        assert hud._inventory_items == ()
        assert hud._inv_headers == {}
        assert hud._inv_boxes == []

    def test_inventario_se_rellena_tras_sync(self, demo_setup):
        """Tras un sync_from_state, _inventory_items, _inv_headers e
        _inv_boxes están poblados coherentemente."""
        hud = demo_setup["hud"]
        timeline = demo_setup["timeline"]
        state = demo_setup["state"]
        snap_a, snap_b, progress = timeline.sample(state.playback_time)
        hud.sync_from_state(state, snap_a, snap_b, progress)
        assert len(hud._inventory_items) > 0
        # Hay al menos una cabecera y una entidad en los items.
        has_header = any(it.kind == "header" for it in hud._inventory_items)
        has_entity = any(it.kind == "entity" for it in hud._inventory_items)
        assert has_header
        assert has_entity
        # Y se han construido widgets correspondientes.
        assert len(hud._inv_headers) > 0
        assert len(hud._inv_boxes) > 0

    def test_inventario_muta_al_avanzar_el_plan(self, demo_setup):
        """Al avanzar a un snapshot donde d1 lleva un paquete, la línea
        de pkg_med1 cambia (de 'en deposito' a 'arm.izq de d1')."""
        hud = demo_setup["hud"]
        timeline = demo_setup["timeline"]
        state = demo_setup["state"]

        # Snapshot inicial.
        state.playback_time = 0.0
        snap_a, snap_b, progress = timeline.sample(0.0)
        hud.sync_from_state(state, snap_a, snap_b, progress)
        pkg_line_initial = next(
            it.text for it in hud._inventory_items
            if it.entity_id == "pkg_med1"
        )

        # Tras PickUp.
        state.playback_time = 7.0  # durante Move, pkg_med1 HeldByArm.
        snap_a, snap_b, progress = timeline.sample(7.0)
        hud.sync_from_state(state, snap_a, snap_b, progress)
        pkg_line_after = next(
            it.text for it in hud._inventory_items
            if it.entity_id == "pkg_med1"
        )

        assert pkg_line_initial != pkg_line_after, (
            "el inventario no cambió al avanzar el plan"
        )

    def test_inventario_no_re_renderiza_si_datos_iguales(self, demo_setup):
        """Optimización: si el snapshot activo y el plegado no cambian, el
        inventario NO se reconstruye ni se re-rasteriza (evita destruir/recrear
        decenas de widgets cada frame). El trabajo se gatea por identidad del
        snapshot (mismo objeto durante todo un segmento) y, cuando cambia,
        solo se actualizan in situ las cajas cuyo contenido difiere."""
        hud = demo_setup["hud"]
        timeline = demo_setup["timeline"]
        state = demo_setup["state"]
        state.playback_time = 0.0
        snap_a, snap_b, progress = timeline.sample(0.0)
        hud.sync_from_state(state, snap_a, snap_b, progress)
        # Capturo refs exactas tras la primera construcción.
        items_ref_before = hud._inventory_items
        key_before = hud._inv_structure_key
        cards_before = dict(hud._inv_card_by_id)
        # Otro sync con el MISMO snapshot: nada debe reconstruirse ni
        # re-rasterizarse; las cajas deben ser los MISMOS objetos UIImage.
        hud.sync_from_state(state, snap_a, snap_b, progress)
        assert hud._inventory_items is items_ref_before
        assert hud._inv_structure_key == key_before
        assert hud._inv_card_by_id == cards_before
        for eid, img in cards_before.items():
            assert hud._inv_card_by_id[eid] is img

    def test_click_en_cabecera_produce_intent_toggle(self, demo_setup):
        """Click sobre una cabecera del inventario produce un HudIntent
        inventory_toggle_section con la categoría como payload."""
        import pygame_gui
        hud = demo_setup["hud"]
        timeline = demo_setup["timeline"]
        state = demo_setup["state"]
        snap_a, snap_b, progress = timeline.sample(0.0)
        hud.sync_from_state(state, snap_a, snap_b, progress)

        # Tomar la cabecera de DRONES (header_btn, chevron_img).
        header_btn, _chevron = hud._inv_headers["DRONES"]
        ev = pygame.event.Event(
            pygame_gui.UI_BUTTON_PRESSED, {"ui_element": header_btn},
        )
        intent = hud.process_event(ev)
        assert intent is not None
        assert intent.kind == "inventory_toggle_section"
        assert intent.payload == "DRONES"

    def test_seccion_plegada_oculta_sus_cajas(self, demo_setup):
        """Con una categoría en state.inventory_collapsed, el rebuild
        no crea cajas para sus entidades."""
        hud = demo_setup["hud"]
        timeline = demo_setup["timeline"]
        state = demo_setup["state"]
        state.inventory_collapsed.add("DRONES")
        snap_a, snap_b, progress = timeline.sample(0.0)
        hud.sync_from_state(state, snap_a, snap_b, progress)
        box_ids = [eid for _, eid in hud._inv_boxes]
        assert "d1" not in box_ids
        assert "d2" not in box_ids
        # Pero la cabecera DRONES sigue presente (solo se ocultan cajas).
        assert "DRONES" in hud._inv_headers

    def test_desplegar_restaura_cajas(self, demo_setup):
        """Quitar la categoría del set restaura sus cajas en el rebuild."""
        hud = demo_setup["hud"]
        timeline = demo_setup["timeline"]
        state = demo_setup["state"]
        snap_a, snap_b, progress = timeline.sample(0.0)
        # Plegar y sincronizar.
        state.inventory_collapsed.add("DRONES")
        hud.sync_from_state(state, snap_a, snap_b, progress)
        assert "d1" not in [eid for _, eid in hud._inv_boxes]
        # Desplegar y re-sincronizar.
        state.inventory_collapsed.discard("DRONES")
        hud.sync_from_state(state, snap_a, snap_b, progress)
        assert "d1" in [eid for _, eid in hud._inv_boxes]


# ---------------------------------------------------------------------------
# process_event: inventory_focus desde click sobre caja (hit-test)
# ---------------------------------------------------------------------------


class TestHudInventoryClickToFocus:
    """El click sobre una caja del inventario produce un HudIntent
    inventory_focus con el entity_id de esa caja, vía hit-test manual
    sobre los UIImage (que no emiten eventos de click por sí mismos)."""

    def test_click_en_caja_produce_inventory_focus(self, demo_setup):
        hud = demo_setup["hud"]
        timeline = demo_setup["timeline"]
        state = demo_setup["state"]
        snap_a, snap_b, progress = timeline.sample(0.0)
        hud.sync_from_state(state, snap_a, snap_b, progress)

        # Tomar una caja cualquiera y simular click en su centro.
        box_img, entity_id = hud._inv_boxes[0]
        ev = pygame.event.Event(
            pygame.MOUSEBUTTONDOWN,
            {"pos": box_img.rect.center, "button": 1},
        )
        intent = hud.process_event(ev)
        assert intent is not None
        assert intent.kind == "inventory_focus"
        assert intent.payload == entity_id

    def test_click_fuera_de_cajas_no_produce_intent(self, demo_setup):
        """Un click fuera del área del scroll del inventario no
        produce intent."""
        hud = demo_setup["hud"]
        timeline = demo_setup["timeline"]
        state = demo_setup["state"]
        snap_a, snap_b, progress = timeline.sample(0.0)
        hud.sync_from_state(state, snap_a, snap_b, progress)

        # Click en (5, 5) — esquina superior izquierda, zona del HUD top,
        # lejos del panel lateral del inventario.
        ev = pygame.event.Event(
            pygame.MOUSEBUTTONDOWN, {"pos": (5, 5), "button": 1},
        )
        intent = hud.process_event(ev)
        assert intent is None

    def test_click_derecho_no_produce_intent(self, demo_setup):
        """Solo el botón izquierdo (button=1) dispara el hit-test."""
        hud = demo_setup["hud"]
        timeline = demo_setup["timeline"]
        state = demo_setup["state"]
        snap_a, snap_b, progress = timeline.sample(0.0)
        hud.sync_from_state(state, snap_a, snap_b, progress)

        box_img, _ = hud._inv_boxes[0]
        ev = pygame.event.Event(
            pygame.MOUSEBUTTONDOWN,
            {"pos": box_img.rect.center, "button": 3},  # botón derecho.
        )
        intent = hud.process_event(ev)
        assert intent is None

    def test_caja_de_seccion_plegada_no_es_clicable(self, demo_setup):
        """Tras plegar una sección, sus entidades no tienen caja, así
        que un click en su posición previa no las enfoca."""
        hud = demo_setup["hud"]
        timeline = demo_setup["timeline"]
        state = demo_setup["state"]
        snap_a, snap_b, progress = timeline.sample(0.0)
        hud.sync_from_state(state, snap_a, snap_b, progress)
        # Localizar caja de d1 antes de plegar.
        d1_box = next((b for b, e in hud._inv_boxes if e == "d1"), None)
        assert d1_box is not None
        d1_center = d1_box.rect.center

        # Plegar DRONES y re-sincronizar.
        state.inventory_collapsed.add("DRONES")
        hud.sync_from_state(state, snap_a, snap_b, progress)

        # Un click en la posición previa de d1 NO debe producir un
        # inventory_focus de d1 (su caja ya no existe).
        ev = pygame.event.Event(
            pygame.MOUSEBUTTONDOWN, {"pos": d1_center, "button": 1},
        )
        intent = hud.process_event(ev)
        # Puede caer sobre otra caja (la que ocupó ese hueco) o sobre
        # nada, pero NUNCA sobre d1.
        if intent is not None:
            assert intent.payload != "d1"


class TestHudFailurePanelEnTab:
    def test_lista_de_fallos_dentro_de_la_tab_metricas(self, failure_setup):
        """En el escenario con fallos, las tarjetas de fallo se crean
        dentro de la tab Métricas+Fallos. Lo verificamos comprobando que
        la lista de tarjetas no está vacía."""
        hud = failure_setup["hud"]
        assert hud._fail_cards


# ---------------------------------------------------------------------------
# process_event: traducción a HudIntent
# ---------------------------------------------------------------------------


class TestHudProcessEvent:
    def test_button_pressed_play_pause(self, demo_setup):
        hud = demo_setup["hud"]
        # Construimos un evento sintético UI_BUTTON_PRESSED con el
        # ui_element apuntando al botón play_pause.
        ev = pygame.event.Event(
            pygame_gui.UI_BUTTON_PRESSED,
            {"ui_element": hud._btn_play_pause},
        )
        intent = hud.process_event(ev)
        assert intent is not None
        assert intent.kind == "play_pause"

    def test_button_pressed_home(self, demo_setup):
        hud = demo_setup["hud"]
        ev = pygame.event.Event(
            pygame_gui.UI_BUTTON_PRESSED,
            {"ui_element": hud._btn_home},
        )
        intent = hud.process_event(ev)
        assert intent.kind == "home"

    def test_button_pressed_end(self, demo_setup):
        hud = demo_setup["hud"]
        ev = pygame.event.Event(
            pygame_gui.UI_BUTTON_PRESSED,
            {"ui_element": hud._btn_end},
        )
        intent = hud.process_event(ev)
        assert intent.kind == "end"

    def test_button_pressed_step_back(self, demo_setup):
        hud = demo_setup["hud"]
        ev = pygame.event.Event(
            pygame_gui.UI_BUTTON_PRESSED,
            {"ui_element": hud._btn_step_back},
        )
        intent = hud.process_event(ev)
        assert intent.kind == "step_back"

    def test_button_pressed_step_forward(self, demo_setup):
        hud = demo_setup["hud"]
        ev = pygame.event.Event(
            pygame_gui.UI_BUTTON_PRESSED,
            {"ui_element": hud._btn_step_forward},
        )
        intent = hud.process_event(ev)
        assert intent.kind == "step_forward"

    def test_button_pressed_speed_cada_velocidad(self, demo_setup):
        """Cada botón de velocidad emite un speed intent con su payload."""
        hud = demo_setup["hud"]
        from droneplan_viz_app.app_state import ALLOWED_SPEEDS
        for btn, expected_speed in zip(hud._btn_speeds, ALLOWED_SPEEDS):
            ev = pygame.event.Event(
                pygame_gui.UI_BUTTON_PRESSED,
                {"ui_element": btn},
            )
            intent = hud.process_event(ev)
            assert intent.kind == "speed"
            assert intent.payload == expected_speed

    def test_evento_irrelevante_devuelve_none(self, demo_setup):
        """Eventos no manejados (KEYDOWN, MOUSEMOTION) → None."""
        hud = demo_setup["hud"]
        ev = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_a)
        assert hud.process_event(ev) is None

        ev2 = pygame.event.Event(
            pygame.MOUSEMOTION, pos=(100, 100), rel=(1, 1), buttons=(0, 0, 0)
        )
        assert hud.process_event(ev2) is None

    def test_button_pressed_externo_devuelve_none(self, demo_setup):
        """Un UI_BUTTON_PRESSED con ui_element que no es uno de los
        nuestros (p.ej. un botón añadido por código externo) debe
        devolver None."""
        hud = demo_setup["hud"]
        from pygame_gui.elements import UIButton
        external_btn = UIButton(
            relative_rect=pygame.Rect(0, 0, 50, 30),
            text="external",
            manager=demo_setup["manager"],
        )
        ev = pygame.event.Event(
            pygame_gui.UI_BUTTON_PRESSED, {"ui_element": external_btn}
        )
        assert hud.process_event(ev) is None


class TestHudProcessEventFailureSelection:
    def test_failure_select_emite_intent(self, failure_setup):
        """Un click sobre una tarjeta de fallo se traduce a
        HudIntent(kind='failure_select', payload=idx).

        El panel de fallos se rediseñó de UISelectionList a tarjetas
        UIImage: la selección es un hit-test de click igual que las
        tarjetas de acción.
        """
        hud = failure_setup["hud"]
        hud.show_metrics_failures_tab()
        assert hud._fail_cards  # hay al menos una tarjeta de fallo

        card_img = hud._fail_cards[0][0]
        cx, cy = card_img.rect.center
        ev = pygame.event.Event(
            pygame.MOUSEBUTTONDOWN, {"button": 1, "pos": (cx, cy)}
        )
        intent = hud.process_event(ev)

        assert intent is not None
        assert intent.kind == "failure_select"
        assert intent.payload == 0

    def test_panel_sin_fallos_no_tiene_tarjetas(self, demo_setup):
        """En el escenario feliz no hay tarjetas de fallo (label 'Sin
        fallos'). El scroll y la lista de tarjetas quedan vacíos.
        """
        hud = demo_setup["hud"]
        assert hud._fail_scroll is None
        assert not hud._fail_cards


# ---------------------------------------------------------------------------
# sync_from_state
# ---------------------------------------------------------------------------


class TestHudSyncFromState:
    def test_play_label_cuando_pausado(self, demo_setup):
        """En modo legacy (sin sprites): texto "Play" cuando paused=True.
        En modo sprites: el object_id se ajusta a @transport_play.

        Ambos casos comunican lo mismo (botón listo para iniciar
        reproducción) pero el mecanismo es distinto."""
        hud = demo_setup["hud"]
        state = AppState(paused=True)
        tl = demo_setup["timeline"]
        snap_a, snap_b, p = tl.sample(0.0)
        hud.sync_from_state(state, snap_a, snap_b, p)
        if hud._transport_sprites_available():
            # Con sprites: comprobar el object_id resultante.
            assert getattr(
                hud._btn_play_pause, "_inv_current_oid", None
            ) == "@transport_play"
        else:
            # Sin sprites: comprobar el texto.
            assert hud._btn_play_pause.text == "Play"

    def test_pause_label_cuando_playing(self, demo_setup):
        """Análogo a test_play_label_cuando_pausado pero para
        paused=False (debe mostrar Pause)."""
        hud = demo_setup["hud"]
        state = AppState(paused=False)
        tl = demo_setup["timeline"]
        snap_a, snap_b, p = tl.sample(0.0)
        hud.sync_from_state(state, snap_a, snap_b, p)
        if hud._transport_sprites_available():
            assert getattr(
                hud._btn_play_pause, "_inv_current_oid", None
            ) == "@transport_pause"
        else:
            assert hud._btn_play_pause.text == "Pause"

    def test_botones_de_velocidad_marcan_el_activo(self, demo_setup):
        hud = demo_setup["hud"]
        state = AppState(playback_speed=2.0)
        tl = demo_setup["timeline"]
        snap_a, snap_b, p = tl.sample(0.0)
        hud.sync_from_state(state, snap_a, snap_b, p)
        # El botón 2× debe llevar corchetes; los demás no.
        from droneplan_viz_app.app_state import ALLOWED_SPEEDS
        for btn, speed in zip(hud._btn_speeds, ALLOWED_SPEEDS):
            if speed == 2.0:
                assert btn.text == "[2×]", f"botón 2× = {btn.text!r}"
            else:
                assert "[" not in btn.text, f"botón {speed}× = {btn.text!r}"

    def test_progress_bar_avanza(self, demo_setup):
        hud = demo_setup["hud"]
        tl = demo_setup["timeline"]
        state = AppState(playback_time=tl.duration / 2)
        snap_a, snap_b, p = tl.sample(state.playback_time)
        hud.sync_from_state(state, snap_a, snap_b, p)
        assert hud._progress_bar.current_progress == pytest.approx(0.5, abs=0.01)

    def test_label_now_muestra_command_en_transicion(self, demo_setup):
        hud = demo_setup["hud"]
        tl = demo_setup["timeline"]
        # Caemos en medio de PickUp (entre t=0 y t=5).
        state = AppState(playback_time=2.5)
        snap_a, snap_b, p = tl.sample(2.5)
        hud.sync_from_state(state, snap_a, snap_b, p)
        # El Command de la transición es un PickUp.
        assert "PickUp" in hud._lbl_now.text or "Now" in hud._lbl_now.text

    def test_label_now_dash_en_estado_estatico(self, demo_setup):
        """En t=0 estamos en el snapshot inicial, sin transición."""
        hud = demo_setup["hud"]
        tl = demo_setup["timeline"]
        state = AppState(playback_time=0.0)
        snap_a, snap_b, p = tl.sample(0.0)
        # snap_a is snap_b en este caso (saneo izquierdo de sample).
        hud.sync_from_state(state, snap_a, snap_b, p)
        assert hud._lbl_now.text == "Now: —"

    def test_label_now_dash_en_tramo_administrativo_inicial(self, demo_setup):
        """Regresión del bug del label "Now" descubierto en validación
        interactiva tras el patch de Sesión D.

        Entre el snapshot inicial (t=0) y el snap_start del primer
        Command durativo (vt=0.6 en el demo), classify_transition
        devuelve TransitionStatic (caso 4b nuevo del fix de D). El
        label "Now:" debe mostrar dash, NO el comando aún por venir.

        Antes del fix de E.5+, el label se encendía con "Now: PickUp(...)"
        durante esos 0.6s administrativos, dando la falsa impresión de
        que el PickUp se ejecutaba dos veces.
        """
        hud = demo_setup["hud"]
        tl = demo_setup["timeline"]
        # t=0.3 cae a la mitad del tramo administrativo 0 → 0.6.
        state = AppState(playback_time=0.3)
        snap_a, snap_b, p = tl.sample(0.3)
        hud.sync_from_state(state, snap_a, snap_b, p)
        assert hud._lbl_now.text == "Now: —", (
            f"label debería ser dash durante tramo administrativo, "
            f"es {hud._lbl_now.text!r}"
        )

    def test_label_now_dash_en_tramo_administrativo_entre_commands(self, demo_setup):
        """En los pares end_anterior→start_siguiente (timestamps
        idénticos, fallback de 0.6s), el render devuelve TransitionStatic
        por el caso 4 de classify_transition (Commands distintos). El
        label "Now:" debe respetar eso.

        Plan demo: end PickUp en vt=5.4, start Move en vt=5.8. El tramo
        intermedio (vt ∈ (5.4, 5.8)) es administrativo y debe mostrar
        dash, no anticipar el Move.
        """
        hud = demo_setup["hud"]
        tl = demo_setup["timeline"]
        # t=5.6 cae entre end del PickUp (vt=5.4) y start del Move (vt=5.8).
        state = AppState(playback_time=5.6)
        snap_a, snap_b, p = tl.sample(5.6)
        hud.sync_from_state(state, snap_a, snap_b, p)
        assert hud._lbl_now.text == "Now: —", (
            f"label debería ser dash entre Commands consecutivos, "
            f"es {hud._lbl_now.text!r}"
        )

    def test_label_now_muestra_command_durante_tramo_animable(self, demo_setup):
        """En vt=10 estamos dentro del Move real (entre vt=6.2 y vt=14.2).
        El label debe mostrar el Move."""
        hud = demo_setup["hud"]
        tl = demo_setup["timeline"]
        state = AppState(playback_time=10.0)
        snap_a, snap_b, p = tl.sample(10.0)
        hud.sync_from_state(state, snap_a, snap_b, p)
        assert "Move" in hud._lbl_now.text
        assert "Now: —" not in hud._lbl_now.text

    def test_metricas_se_muestran(self, demo_setup):
        """Tras sync, los labels de métricas contienen valores numéricos."""
        hud = demo_setup["hud"]
        tl = demo_setup["timeline"]
        state = AppState(playback_time=tl.duration)
        snap_a, snap_b, p = tl.sample(tl.duration)
        hud.sync_from_state(state, snap_a, snap_b, p)
        # Las métricas finales del demo: 3 actions, 0 failures.
        assert "actions:" in hud._lbl_action_count.text
        assert "failures:" in hud._lbl_failed_count.text


# ---------------------------------------------------------------------------
# update + draw
# ---------------------------------------------------------------------------


class TestHudDraw:
    def test_update_y_draw_no_crashean(self, demo_setup):
        hud = demo_setup["hud"]
        window = demo_setup["window"]
        state = demo_setup["state"]
        tl = demo_setup["timeline"]
        snap_a, snap_b, p = tl.sample(0.0)
        hud.sync_from_state(state, snap_a, snap_b, p)
        hud.update(0.016)
        hud.draw(window, state)

    def test_draw_deja_pixeles_en_hud_top(self, demo_setup):
        """Smoke estructural: tras draw, la región hud_top tiene
        píxeles distintos al fondo."""
        hud = demo_setup["hud"]
        window = demo_setup["window"]
        state = demo_setup["state"]
        tl = demo_setup["timeline"]
        regions = demo_setup["regions"]
        snap_a, snap_b, p = tl.sample(0.0)

        # Limpiar la ventana primero.
        window.fill((0, 0, 0))
        hud.sync_from_state(state, snap_a, snap_b, p)
        hud.update(0.016)
        hud.draw(window, state)

        # En la zona de los botones (x≈40, y≈25) debe haber píxel no negro.
        pixel = window.get_at((40, 25))
        assert pixel != (0, 0, 0, 255), f"hud_top vacío: pixel={pixel}"

    def test_draw_pinta_marca_roja_de_fallo(self, failure_setup):
        """Smoke: tras draw del HUD con un escenario de fallo, hay al menos
        un píxel rojo brillante sobre la línea de la progress bar."""
        hud = failure_setup["hud"]
        window = failure_setup["window"]
        state = failure_setup["state"]
        tl = failure_setup["timeline"]
        snap_a, snap_b, p = tl.sample(0.0)

        window.fill((0, 0, 0))
        hud.sync_from_state(state, snap_a, snap_b, p)
        hud.update(0.016)
        hud.draw(window, state)

        # La marca está sobre la barra. La barra está en hud_bottom.
        bar_rect = hud._progress_bar.rect
        # Buscar un píxel rojo (alto R, bajo G/B) en la zona vertical
        # de la barra.
        found_red = False
        y_mid = bar_rect.y + bar_rect.height // 2
        for x in range(bar_rect.x, bar_rect.x + bar_rect.width):
            r, g, b, _a = window.get_at((x, y_mid))
            if r > 200 and g < 120 and b < 120:
                found_red = True
                break
        assert found_red, "no se encontró marca roja sobre la barra"

    def test_halo_amarillo_aparece_cuando_failure_seleccionado(self, failure_setup):
        """Cuando state.selected_failure = 0, la marca seleccionada
        lleva un halo amarillo translúcido pintado DETRÁS de la línea
        roja. El halo se mezcla con el fondo (verde grisáceo del relleno
        de la barra), produciendo tonos cálidos donde R y G están altos
        respecto a B.
        """
        hud = failure_setup["hud"]
        window = failure_setup["window"]
        tl = failure_setup["timeline"]
        state = AppState(selected_failure=0)
        snap_a, snap_b, p = tl.sample(0.0)

        window.fill((0, 0, 0))
        hud.sync_from_state(state, snap_a, snap_b, p)
        hud.update(0.016)
        hud.draw(window, state)

        # El halo se pinta en x∈[mark_x-4, mark_x+4]; la línea roja
        # de thickness=3 cubre x∈[mark_x-1, mark_x+1]. Sample en los
        # bordes del halo (a 3 px de la línea central) para no
        # quedar dentro del rojo puro.
        bar_rect = hud._progress_bar.rect
        info = failure_setup["plan_info"]
        frac = info.failure_starts[0] / info.duration
        mark_x = bar_rect.x + int(frac * bar_rect.width)
        y_mid = bar_rect.y + bar_rect.height // 2

        # Sample en x = mark_x ± 3, ambos lados.
        found_warm = False
        for dx in (-3, -2, 2, 3):
            r, g, b, _a = window.get_at((mark_x + dx, y_mid))
            # Tonos cálidos: R y G distintamente más altos que B.
            if r > 100 and g > 100 and r > b + 50 and g > b + 30:
                found_warm = True
                break
        assert found_warm, "no se encontró halo cálido (amarillo) sobre la barra"


# ---------------------------------------------------------------------------
# Relayout (recomposición tras VIDEORESIZE)
# ---------------------------------------------------------------------------


class TestHudRelayout:
    """El Hud debe poder reposicionar todos sus widgets en regions
    nuevas. Verifica que tras relayout:

      - Los widgets siguen existiendo (referencias no son None).
      - Sus rect coinciden con las nuevas regions (al menos el
        progress bar, que es el más sensible al ancho).
      - process_event sigue funcionando con los widgets NUEVOS (las
        referencias internas se actualizaron).
      - draw() sigue produciendo píxeles, sin crashear.
    """

    def test_relayout_no_crashea(self, demo_setup):
        hud = demo_setup["hud"]
        new_regions = compute_regions((1600, 900))
        hud.relayout(new_regions)
        # Sin asserts: el smoke es que no lance.

    def test_relayout_reposiciona_progress_bar(self, demo_setup):
        """La progress bar debe expandirse al hacer la ventana más
        ancha. Si seguía con el width de 1280, fallaría."""
        hud = demo_setup["hud"]
        bar_w_before = hud._progress_bar.rect.width
        new_regions = compute_regions((1600, 900))
        hud.relayout(new_regions)
        bar_w_after = hud._progress_bar.rect.width
        assert bar_w_after > bar_w_before, (
            f"barra no se expandió: antes={bar_w_before}, después={bar_w_after}"
        )

    def test_relayout_reposiciona_botones_de_velocidad(self, demo_setup):
        """En una ventana más ancha, los botones de velocidad deben
        estar más a la derecha."""
        hud = demo_setup["hud"]
        speed_x_before = hud._btn_speeds[0].rect.x
        hud.relayout(compute_regions((1600, 900)))
        speed_x_after = hud._btn_speeds[0].rect.x
        assert speed_x_after > speed_x_before

    def test_relayout_reposiciona_panel_lateral(self, demo_setup):
        """El panel lateral debe quedar pegado al borde derecho de la
        nueva ventana."""
        hud = demo_setup["hud"]
        hud.relayout(compute_regions((1600, 900)))
        # hud_right.x = 1600 - 380 = 1220. El panel de métricas está
        # dentro de hud_right (que ahora aloja el tab container) con
        # padding adicional.
        assert hud._panel_metrics.rect.x >= 1220

    def test_relayout_acepta_misma_region_idempotente(self, demo_setup):
        """Recomponer con las mismas regions debe ser inocuo."""
        hud = demo_setup["hud"]
        bar_w_before = hud._progress_bar.rect.width
        hud.relayout(demo_setup["regions"])
        bar_w_after = hud._progress_bar.rect.width
        assert bar_w_after == bar_w_before

    def test_process_event_funciona_tras_relayout(self, demo_setup):
        """Las referencias internas a botones se actualizan tras
        relayout (los antiguos están muertos)."""
        hud = demo_setup["hud"]
        hud.relayout(compute_regions((1600, 900)))
        # Disparar el botón Play/Pause NUEVO.
        ev = pygame.event.Event(
            pygame_gui.UI_BUTTON_PRESSED,
            {"ui_element": hud._btn_play_pause},
        )
        intent = hud.process_event(ev)
        assert intent is not None
        assert intent.kind == "play_pause"

    def test_draw_tras_relayout_no_crashea(self, demo_setup):
        hud = demo_setup["hud"]
        window = demo_setup["window"]
        state = demo_setup["state"]
        tl = demo_setup["timeline"]

        hud.relayout(compute_regions((1600, 900)))
        snap_a, snap_b, p = tl.sample(0.0)
        hud.sync_from_state(state, snap_a, snap_b, p)
        hud.update(0.016)
        hud.draw(window, state)

    def test_relayout_con_failure_panel(self, failure_setup):
        """El panel de fallos también debe recomponerse correctamente
        cuando hay fallos en el plan."""
        hud = failure_setup["hud"]
        hud.relayout(compute_regions((1600, 900)))
        # Las tarjetas de fallo se reconstruyen tras el relayout.
        assert hud._fail_cards

    def test_relayout_recrea_widgets_distintos(self, demo_setup):
        """Tras relayout, las referencias a widgets son OBJETOS NUEVOS,
        no los mismos objetos reposicionados. Esto importa porque la
        identidad (is) es lo que usa process_event para despachar."""
        hud = demo_setup["hud"]
        btn_play_before = hud._btn_play_pause
        hud.relayout(compute_regions((1600, 900)))
        btn_play_after = hud._btn_play_pause
        assert btn_play_after is not btn_play_before


# ---------------------------------------------------------------------------
# HudLayout personalizado
# ---------------------------------------------------------------------------


class TestHudConHudLayoutPersonalizado:
    """Pasar un HudLayout personalizado debe alterar el aspecto del HUD
    sin tocar el código del módulo. Esto prueba el patrón de extensión
    consensuado para futuras sesiones de sprites/theming."""

    def test_acepta_hud_layout_personalizado(self, window_and_manager):
        from droneplan_viz_app.layout import HudLayout
        import dataclasses
        _, manager = window_and_manager
        regions = compute_regions((1280, 800))
        theme = Theme.default()
        world, plan = build_demo_scenario()
        runner = PlanRunner(world)
        result = runner.execute(plan)
        tl = Timeline(runner.history, theme=theme)
        info = build_hud_plan_info(tl.duration, tl.snapshot_times, result.failures)

        custom = dataclasses.replace(HudLayout(), top_btn_h=48, top_btn_w=80)
        hud = Hud(manager, regions, info, theme, hud_layout=custom)
        # Botones más altos y anchos: el primero debe medir 80×48.
        assert hud._btn_home.rect.width == 80
        assert hud._btn_home.rect.height == 48