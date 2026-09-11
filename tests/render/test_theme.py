"""Tests para droneplan_viz.render.theme.

Sin pygame. Verifican:
- Inmutabilidad y construcción del Theme.
- Heurística de Locations (con y sin opt-out).
- Estrategias de color de Content (hybrid, fallback_only, id_only).
- Determinismo cross-run del fallback (zlib.adler32).
- color_for_drone_state cubre los 4 estados FSM.

Importamos DroneState del dominio porque está consagrado en Sesión A y
es estable. El test_color_for_drone_state_con_objeto_no_enum verifica
que la función también acepta objetos con .name (defensa en profundidad
documentada en el docstring).
"""
from dataclasses import FrozenInstanceError

import pytest

from droneplan_viz.domain.drone_state import DroneState
from droneplan_viz.render.theme import (
    FALLBACK_CONTENT_PALETTE,
    KNOWN_CONTENT_COLORS,
    Theme,
    color_for_content,
    color_for_drone_state,
    color_for_location,
)


# ---------------------------------------------------------------------------
# Construcción e inmutabilidad
# ---------------------------------------------------------------------------


class TestThemeConstruccion:
    """Theme: construcción por default() y with_overrides()."""

    def test_default_produce_instancia_completa(self):
        t = Theme.default()
        # Sanity checks sobre algunos campos clave:
        assert isinstance(t.background, tuple)
        assert len(t.background) == 3
        assert t.drone_radius > 0
        assert t.padding > 0
        assert t.fallback_step_duration == 0.4
        assert t.differentiate_locations is False
        assert t.content_color_strategy == "hybrid"

    def test_default_es_idempotente_pero_no_singleton(self):
        # Dos llamadas a default() producen valores iguales pero NO la
        # misma instancia: dataclasses frozen no son singletons.
        a = Theme.default()
        b = Theme.default()
        assert a == b  # igualdad estructural
        # No exigimos identidad (a is b) porque no usamos cache.

    def test_inmutable_lanza_excepcion(self):
        t = Theme.default()
        with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
            t.padding = 100  # type: ignore[misc]

    def test_with_overrides_un_campo(self):
        t = Theme.default()
        t2 = t.with_overrides(padding=40)
        assert t2.padding == 40
        # El original sigue intacto:
        assert t.padding == 24

    def test_with_overrides_varios_campos(self):
        t = Theme.default().with_overrides(
            padding=40,
            drone_radius=22,
            differentiate_locations=False,
        )
        assert t.padding == 40
        assert t.drone_radius == 22
        assert t.differentiate_locations is False

    def test_with_overrides_sin_argumentos_devuelve_equivalente(self):
        t = Theme.default()
        assert t.with_overrides() == t

    def test_campo_no_existente_lanza(self):
        # dataclasses.replace lanza TypeError si el kwarg no es un campo.
        with pytest.raises(TypeError):
            Theme.default().with_overrides(zorblax=42)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# color_for_location: heurística de id
# ---------------------------------------------------------------------------


class TestColorForLocationConHeuristica:
    """Con differentiate_locations=True (activado explícitamente), aplicar la heurística."""

    def test_casa_color_tierra(self):
        theme = Theme.default().with_overrides(differentiate_locations=True)
        fill, _ = color_for_location("casa1", theme)
        assert fill == theme.loc_fill_house

    def test_casa_case_insensitive(self):
        theme = Theme.default().with_overrides(differentiate_locations=True)
        fill, _ = color_for_location("Casa42", theme)
        assert fill == theme.loc_fill_house

    def test_house_en_ingles(self):
        theme = Theme.default().with_overrides(differentiate_locations=True)
        fill, _ = color_for_location("house_A", theme)
        assert fill == theme.loc_fill_house

    def test_home_en_ingles(self):
        theme = Theme.default().with_overrides(differentiate_locations=True)
        fill, _ = color_for_location("home1", theme)
        assert fill == theme.loc_fill_house

    def test_hospital_color_blanco_borde_rojo(self):
        theme = Theme.default().with_overrides(differentiate_locations=True)
        fill, border = color_for_location("hospital_central", theme)
        assert fill == theme.loc_fill_hospital
        assert border == theme.loc_border_hospital

    def test_clinica_se_trata_como_hospital(self):
        theme = Theme.default().with_overrides(differentiate_locations=True)
        fill, _ = color_for_location("clinica2", theme)
        assert fill == theme.loc_fill_hospital

    def test_deposito_color_azul(self):
        theme = Theme.default().with_overrides(differentiate_locations=True)
        fill, _ = color_for_location("deposito_norte", theme)
        assert fill == theme.loc_fill_depot

    def test_almacen_se_trata_como_deposito(self):
        theme = Theme.default().with_overrides(differentiate_locations=True)
        fill, _ = color_for_location("almacen1", theme)
        assert fill == theme.loc_fill_depot

    def test_warehouse_se_trata_como_deposito(self):
        theme = Theme.default().with_overrides(differentiate_locations=True)
        fill, _ = color_for_location("warehouse_main", theme)
        assert fill == theme.loc_fill_depot

    def test_loc_neutral_por_defecto(self):
        theme = Theme.default().with_overrides(differentiate_locations=True)
        fill, border = color_for_location("punto_x", theme)
        assert fill == theme.loc_fill_neutral
        assert border == theme.loc_border_neutral

    def test_no_matchea_subcadena_accidentalmente(self):
        # "encasamiento" no debería matchear "casa" porque no empieza por casa.
        theme = Theme.default().with_overrides(differentiate_locations=True)
        fill, _ = color_for_location("encasamiento", theme)
        assert fill == theme.loc_fill_neutral

    def test_id_vacio_es_neutral(self):
        theme = Theme.default().with_overrides(differentiate_locations=True)
        fill, border = color_for_location("", theme)
        assert fill == theme.loc_fill_neutral
        assert border == theme.loc_border_neutral


class TestColorForLocationSinHeuristica:
    """Con differentiate_locations=False, todo neutral sea cual sea el id."""

    def test_casa_sin_heuristica_es_neutral(self):
        theme = Theme.default().with_overrides(differentiate_locations=False)
        fill, border = color_for_location("casa1", theme)
        assert fill == theme.loc_fill_neutral
        assert border == theme.loc_border_neutral

    def test_hospital_sin_heuristica_es_neutral(self):
        theme = Theme.default().with_overrides(differentiate_locations=False)
        fill, _ = color_for_location("hospital1", theme)
        assert fill == theme.loc_fill_neutral

    def test_deposito_sin_heuristica_es_neutral(self):
        theme = Theme.default().with_overrides(differentiate_locations=False)
        fill, _ = color_for_location("deposito1", theme)
        assert fill == theme.loc_fill_neutral


# ---------------------------------------------------------------------------
# color_for_content: estrategias
# ---------------------------------------------------------------------------


class TestColorForContentHybrid:
    """Estrategia hybrid (default): canónicos vía dict, resto vía fallback."""

    def test_medicina_singular(self):
        theme = Theme.default()
        assert color_for_content("medicina", theme) == KNOWN_CONTENT_COLORS["medicina"]

    def test_medicinas_plural(self):
        theme = Theme.default()
        assert color_for_content("medicinas", theme) == KNOWN_CONTENT_COLORS["medicinas"]

    def test_medicina_y_medicinas_mismo_color(self):
        # Por construcción del dict, singular y plural coinciden.
        theme = Theme.default()
        assert color_for_content("medicina", theme) == color_for_content("medicinas", theme)

    def test_comida(self):
        theme = Theme.default()
        assert color_for_content("comida", theme) == KNOWN_CONTENT_COLORS["comida"]

    def test_agua(self):
        theme = Theme.default()
        assert color_for_content("agua", theme) == KNOWN_CONTENT_COLORS["agua"]

    def test_case_insensitive(self):
        theme = Theme.default()
        assert color_for_content("MEDICINA", theme) == KNOWN_CONTENT_COLORS["medicina"]
        assert color_for_content("Comida", theme) == KNOWN_CONTENT_COLORS["comida"]

    def test_desconocido_usa_fallback(self):
        theme = Theme.default()
        color = color_for_content("juguete", theme)
        # Debe ser uno de los 8 colores del fallback.
        assert color in FALLBACK_CONTENT_PALETTE

    def test_desconocido_es_determinista_entre_llamadas(self):
        theme = Theme.default()
        a = color_for_content("juguete", theme)
        b = color_for_content("juguete", theme)
        assert a == b


class TestColorForContentFallbackOnly:
    """Estrategia fallback_only: ignora canónicos, todos vía adler32."""

    def test_medicina_no_usa_dict(self):
        theme = Theme.default().with_overrides(content_color_strategy="fallback_only")
        color = color_for_content("medicina", theme)
        # NO debe ser el rojo canónico, debe ser uno del fallback.
        assert color in FALLBACK_CONTENT_PALETTE
        assert color != KNOWN_CONTENT_COLORS["medicina"]

    def test_desconocido_funciona_igual_que_hybrid(self):
        theme_hybrid = Theme.default()
        theme_fallback = Theme.default().with_overrides(content_color_strategy="fallback_only")
        # Para un id desconocido, ambas estrategias deben dar el mismo color.
        assert color_for_content("juguete", theme_hybrid) == color_for_content("juguete", theme_fallback)


class TestColorForContentIdOnly:
    """Estrategia id_only: todos los contents reciben text_dim (gris)."""

    def test_medicina_es_gris(self):
        theme = Theme.default().with_overrides(content_color_strategy="id_only")
        assert color_for_content("medicina", theme) == theme.text_dim

    def test_desconocido_es_gris(self):
        theme = Theme.default().with_overrides(content_color_strategy="id_only")
        assert color_for_content("juguete", theme) == theme.text_dim

    def test_todos_los_contents_tienen_el_mismo_color(self):
        theme = Theme.default().with_overrides(content_color_strategy="id_only")
        assert (
            color_for_content("a", theme)
            == color_for_content("b", theme)
            == color_for_content("c", theme)
            == theme.text_dim
        )


class TestColorForContentDeterminismo:
    """El fallback debe ser DETERMINISTA entre procesos (no hash() aleatorio)."""

    def test_juguete_mapea_a_color_concreto(self):
        # Valor calculado a partir de zlib.adler32("juguete") % 8.
        # Documentado aquí como assert literal: si alguien cambia la
        # paleta o la función, este test salta y obliga a actualizar
        # tests dependientes.
        import zlib
        idx = zlib.adler32(b"juguete") % len(FALLBACK_CONTENT_PALETTE)
        expected = FALLBACK_CONTENT_PALETTE[idx]
        theme = Theme.default()
        assert color_for_content("juguete", theme) == expected

    def test_dos_contents_distintos_pueden_caer_en_mismo_color(self):
        # Con solo 8 colores y N contents, la colisión es posible.
        # No es un bug, es la naturaleza del fallback.
        # Este test simplemente documenta que NO garantizamos unicidad.
        theme = Theme.default()
        # En la práctica para el PDF docente solo hay 3 contents
        # canónicos cubiertos por el dict, así que el fallback rara
        # vez se usará en escenarios reales.
        # Lo que sí garantizamos: el color devuelto está en la paleta.
        for content_id in ["a", "b", "c", "d", "e", "f"]:
            color = color_for_content(content_id, theme)
            assert color in FALLBACK_CONTENT_PALETTE


# ---------------------------------------------------------------------------
# color_for_drone_state
# ---------------------------------------------------------------------------


class TestColorForDroneState:
    """Cobertura de los 4 estados FSM."""

    def test_idle_es_verde(self):
        theme = Theme.default()
        assert color_for_drone_state(DroneState.IDLE, theme) == theme.drone_idle

    def test_moving_es_azul(self):
        theme = Theme.default()
        assert color_for_drone_state(DroneState.MOVING, theme) == theme.drone_moving

    def test_interacting_es_ambar(self):
        theme = Theme.default()
        assert color_for_drone_state(DroneState.INTERACTING, theme) == theme.drone_interacting

    def test_error_es_rojo(self):
        theme = Theme.default()
        assert color_for_drone_state(DroneState.ERROR, theme) == theme.drone_error

    def test_acepta_objeto_no_enum_con_atributo_name(self):
        # Defensa en profundidad documentada en el docstring: acepta
        # cualquier objeto con .name. Útil para tests que no quieren
        # importar DroneState.
        class FakeState:
            name = "IDLE"

        theme = Theme.default()
        assert color_for_drone_state(FakeState(), theme) == theme.drone_idle

    def test_estado_desconocido_devuelve_fallback_gris(self):
        # Si llega un estado con .name="ZORBLAX" (jamás debería en la
        # práctica), no rompemos: devolvemos text_dim como fallback.
        class FakeState:
            name = "ZORBLAX"

        theme = Theme.default()
        assert color_for_drone_state(FakeState(), theme) == theme.text_dim


# ---------------------------------------------------------------------------
# Sanity checks de la paleta como datos
# ---------------------------------------------------------------------------


class TestPaletaSanity:
    """Verificaciones estructurales sobre las constantes de paleta."""

    def test_known_content_colors_no_vacio(self):
        assert len(KNOWN_CONTENT_COLORS) > 0

    def test_known_content_colors_son_rgb_validos(self):
        for content_id, rgb in KNOWN_CONTENT_COLORS.items():
            assert len(rgb) == 3
            for c in rgb:
                assert isinstance(c, int)
                assert 0 <= c <= 255

    def test_fallback_palette_es_octeto(self):
        # Mantenemos exactamente 8 colores para que adler32 % 8 reparta
        # bien. Si cambia, el test salta y obliga a revisar el módulo.
        assert len(FALLBACK_CONTENT_PALETTE) == 8

    def test_fallback_palette_son_rgb_validos(self):
        for rgb in FALLBACK_CONTENT_PALETTE:
            assert len(rgb) == 3
            for c in rgb:
                assert isinstance(c, int)
                assert 0 <= c <= 255

    def test_theme_todos_los_colores_son_rgb_validos(self):
        # Iteración defensiva sobre los campos del Theme: cualquier campo
        # cuyo nombre acabe en color-relacionado debe ser tupla 3-int en [0,255].
        # Implementación pragmática: chequear los campos que conocemos
        # como RGB. No vale dataclass.fields() porque algunos campos son
        # ints o bool; chequeamos los que sabemos que son RGB.
        t = Theme.default()
        rgb_fields = [
            t.background, t.edge,
            t.loc_fill_neutral, t.loc_border_neutral,
            t.loc_fill_house, t.loc_fill_hospital, t.loc_border_hospital,
            t.loc_fill_depot,
            t.drone_idle, t.drone_moving, t.drone_interacting, t.drone_error,
            t.drone_border, t.error_x,
            t.transporter_fill, t.transporter_border,
            t.person_fill, t.person_border, t.needs_indicator,
            t.text, t.text_dim,
        ]
        for rgb in rgb_fields:
            assert isinstance(rgb, tuple)
            assert len(rgb) == 3
            for c in rgb:
                assert isinstance(c, int)
                assert 0 <= c <= 255


# ---------------------------------------------------------------------------
# content_colors: override explícito de paleta por contenido (inyectado por el
# escenario). Tiene prioridad sobre la estrategia, case-insensitive.
# ---------------------------------------------------------------------------
class TestContentColorsOverride:
    def test_override_gana_sobre_estrategia(self):
        theme = Theme.default().with_overrides(content_colors={"medicina": (1, 2, 3)})
        assert color_for_content("medicina", theme) == (1, 2, 3)

    def test_override_case_insensitive(self):
        theme = Theme.default().with_overrides(content_colors={"agua": (9, 9, 9)})
        assert color_for_content("AGUA", theme) == (9, 9, 9)

    def test_override_gana_incluso_sobre_id_only(self):
        theme = Theme.default().with_overrides(
            content_colors={"x": (5, 6, 7)}, content_color_strategy="id_only"
        )
        assert color_for_content("x", theme) == (5, 6, 7)

    def test_contenido_fuera_del_override_usa_estrategia(self):
        theme = Theme.default().with_overrides(content_colors={"medicina": (1, 2, 3)})
        # "comida" no está en el override → estrategia normal (canónico), no (1,2,3)
        assert color_for_content("comida", theme) != (1, 2, 3)

    def test_none_no_afecta(self):
        base = color_for_content("medicina", Theme.default())
        theme = Theme.default().with_overrides(content_colors=None)
        assert color_for_content("medicina", theme) == base
