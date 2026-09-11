"""Tests del módulo droneplan_viz_app.inventory.

Tests puros: el módulo inventory NO consume pygame, así que no necesita
SDL dummy ni inicialización de display. Esto los hace muy rápidos.
"""
from __future__ import annotations

import pytest

from droneplan_viz_app.inventory import (
    InventoryItem,
    build_inventory,
    format_drone_line,
    format_location_line,
    format_package_line,
    format_person_line,
    format_transporter_line,
)
from droneplan_viz_app.scenarios import (
    _build_demo_world,
    build_demo_scenario,
    build_failure_demo_scenario,
)
from droneplan_viz.runtime import PlanRunner


# ---------------------------------------------------------------------------
# build_inventory: estructura y orden
# ---------------------------------------------------------------------------


class TestBuildInventoryEstructura:
    def test_produce_tupla_de_inventory_items(self):
        world = _build_demo_world()
        items = build_inventory(world)
        assert isinstance(items, tuple)
        assert all(isinstance(it, InventoryItem) for it in items)

    def test_no_vacio_para_world_demo(self):
        """El demo tiene drones, locations, personas, paquetes y
        transporter: el inventario debe tener items de las 5 categorías."""
        world = _build_demo_world()
        items = build_inventory(world)
        kinds_in_text = {
            "DRONES": False,
            "LOCATIONS": False,
            "PERSONAS": False,
            "PAQUETES": False,
            "TRANSPORTERS": False,
        }
        for it in items:
            if it.kind != "header":
                continue
            for category in kinds_in_text:
                if category in it.text:
                    kinds_in_text[category] = True
        assert all(kinds_in_text.values()), (
            f"categorías faltantes: {[c for c, v in kinds_in_text.items() if not v]}"
        )

    def test_entidades_aparecen_tras_su_cabecera(self):
        """En el orden de items, una entidad debe aparecer DESPUÉS de
        la cabecera de su categoría, no antes."""
        world = _build_demo_world()
        items = build_inventory(world)
        # Localizar la cabecera DRONES.
        for i, it in enumerate(items):
            if it.kind == "header" and "DRONES" in it.text:
                # Los siguientes items hasta otra header deben ser drones.
                j = i + 1
                while j < len(items) and items[j].kind == "entity":
                    assert items[j].entity_id in world.drones, (
                        f"entity {items[j].entity_id!r} bajo DRONES no es un drone"
                    )
                    j += 1
                break

    def test_orden_alfabetico_dentro_de_categoria(self):
        """Items de cada categoría aparecen ordenados por id (lista
        estable; no salta al refrescar)."""
        world = _build_demo_world()
        items = build_inventory(world)
        # Agrupar entidades por la cabecera previa que les corresponda.
        current_section = None
        section_items: dict[str, list[str]] = {}
        for it in items:
            if it.kind == "header":
                current_section = it.text
                section_items[current_section] = []
            elif current_section is not None and it.entity_id is not None:
                section_items[current_section].append(it.entity_id)
        # Cada sección debe estar ordenada.
        for section, ids in section_items.items():
            assert ids == sorted(ids), (
                f"sección {section!r} no está ordenada: {ids}"
            )

    def test_cabeceras_no_tienen_entity_id(self):
        world = _build_demo_world()
        for it in build_inventory(world):
            if it.kind == "header":
                assert it.entity_id is None

    def test_entities_tienen_entity_id(self):
        world = _build_demo_world()
        for it in build_inventory(world):
            if it.kind == "entity":
                assert it.entity_id is not None
                assert isinstance(it.entity_id, str)

    def test_omite_categoria_vacia(self):
        """Si una categoría está vacía, NO debe aparecer su cabecera
        (evita ruido del tipo `· PAQUETES ·` seguido de nada)."""
        # Construyo un world mínimo sin paquetes ni transporters ni personas.
        from dataclasses import replace
        world = _build_demo_world()
        world = replace(world, packages={}, transporters={}, persons={})
        items = build_inventory(world)
        for it in items:
            if it.kind == "header":
                assert "PAQUETES" not in it.text
                assert "TRANSPORTERS" not in it.text
                assert "PERSONAS" not in it.text


# ---------------------------------------------------------------------------
# Formatters individuales: texto correcto según estado de la entidad
# ---------------------------------------------------------------------------


class TestFormatDroneLine:
    def test_drone_idle_sin_carga(self):
        world = _build_demo_world()
        line = format_drone_line(world, "d1")
        # d1 está IDLE en deposito con 2 brazos vacíos.
        assert "d1" in line
        assert "IDLE" in line
        assert "deposito" in line
        assert "brazos 2" in line

    def test_drone_idle_un_solo_brazo(self):
        world = _build_demo_world()
        line = format_drone_line(world, "d2")
        # d2 tiene 1 brazo.
        assert "brazos 1" in line

    def test_drone_con_carga_muestra_paquete(self):
        """Tras un PickUp, el drone lleva carga: el formatter debe
        listarla en la línea en lugar del recuento de brazos."""
        world, plan = build_demo_scenario()
        runner = PlanRunner(world)
        runner.execute(plan)
        # Snapshot 3 = snap_start de Move: d1 lleva pkg_med1.
        snap = runner.history.at(3)
        line = format_drone_line(snap.world, "d1")
        assert "pkg_med1" in line
        assert "izq" in line  # arm.izq lleva la carga
        assert "brazos" not in line  # ya no muestra el recuento.


class TestFormatLocationLine:
    def test_deposito_con_paquetes_y_transporter(self):
        world = _build_demo_world()
        line = format_location_line(world, "deposito")
        # deposito tiene 3 pkg + t1 + 0 personas.
        assert "deposito" in line
        assert "3 pkg" in line
        assert "1 transp" in line
        # 0 personas se omite (decisión: no mostrar categorías a cero).
        assert "0 personas" not in line

    def test_casa1_solo_una_persona(self):
        world = _build_demo_world()
        line = format_location_line(world, "casa1")
        assert "casa1" in line
        assert "1 persona" in line
        # Singular sin "s".
        assert "1 personas" not in line

    def test_hospital_vacio(self):
        world = _build_demo_world()
        line = format_location_line(world, "hospital")
        assert "hospital" in line
        assert "vacía" in line


class TestFormatPersonLine:
    def test_persona_con_dos_necesidades(self):
        world = _build_demo_world()
        line = format_person_line(world, "p1")
        # p1 en casa1, pide medicina y comida.
        assert "p1" in line
        assert "casa1" in line
        assert "medicina" in line
        assert "comida" in line

    def test_persona_con_una_necesidad(self):
        world = _build_demo_world()
        line = format_person_line(world, "p2")
        assert "p2" in line
        assert "casa2" in line
        assert "agua" in line


class TestFormatPackageLine:
    def test_paquete_en_location(self):
        world = _build_demo_world()
        line = format_package_line(world, "pkg_med1")
        # pkg_med1 (medicina) en deposito.
        assert "pkg_med1" in line
        assert "medicina" in line
        assert "deposito" in line

    def test_paquete_held_by_arm(self):
        """Cuando un drone lleva el paquete, la línea informa de quién."""
        world, plan = build_demo_scenario()
        runner = PlanRunner(world)
        runner.execute(plan)
        snap = runner.history.at(3)  # d1 lleva pkg_med1.
        line = format_package_line(snap.world, "pkg_med1")
        assert "pkg_med1" in line
        assert "d1" in line
        assert "izq" in line


class TestFormatTransporterLine:
    def test_transporter_vacio(self):
        world = _build_demo_world()
        line = format_transporter_line(world, "t1")
        assert "t1" in line
        assert "deposito" in line
        assert "0/4" in line


# ---------------------------------------------------------------------------
# Reactivividad: el inventario cambia coherentemente al avanzar el plan
# ---------------------------------------------------------------------------


class TestBuildInventoryReactivo:
    def test_paquete_cambia_de_location_a_held(self):
        """Tras un PickUp, la línea del paquete pasa de 'en location' a
        'arm.X de drone'. El inventario refleja el cambio."""
        world, plan = build_demo_scenario()
        runner = PlanRunner(world)
        runner.execute(plan)

        # Snapshot inicial: pkg_med1 está AtLocation.
        items_initial = build_inventory(world)
        pkg_line_initial = next(
            it.text for it in items_initial if it.entity_id == "pkg_med1"
        )
        assert "deposito" in pkg_line_initial
        assert "arm." not in pkg_line_initial

        # Tras PickUp: pkg_med1 HeldByArm.
        snap_after = runner.history.at(3)
        items_after = build_inventory(snap_after.world)
        pkg_line_after = next(
            it.text for it in items_after if it.entity_id == "pkg_med1"
        )
        assert "arm." in pkg_line_after
        assert "d1" in pkg_line_after

    def test_conteo_de_paquetes_en_location_baja(self):
        """Al recoger pkg_med1, el conteo de paquetes en deposito baja."""
        world, plan = build_demo_scenario()
        runner = PlanRunner(world)
        runner.execute(plan)

        line_initial = format_location_line(world, "deposito")
        assert "3 pkg" in line_initial

        snap_after = runner.history.at(3)
        line_after = format_location_line(snap_after.world, "deposito")
        assert "2 pkg" in line_after


# ---------------------------------------------------------------------------
# Robustez contra casos límite
# ---------------------------------------------------------------------------


class TestBuildInventoryRobustez:
    def test_world_failure_demo_funciona(self):
        """El escenario failure_demo (con d1 + d2 + paquetes y un fallo)
        también produce inventario sin reventar."""
        world, _ = build_failure_demo_scenario()
        items = build_inventory(world)
        assert len(items) > 0

    def test_items_son_inmutables(self):
        """InventoryItem es frozen: no se puede mutar tras crearse."""
        world = _build_demo_world()
        items = build_inventory(world)
        with pytest.raises(Exception):  # FrozenInstanceError o similar.
            items[0].text = "modificado"  # type: ignore[misc]
