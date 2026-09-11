"""Tests para droneplan_viz.render.sprites.

Tests con pygame.Surface en memoria. El conftest.py del paquete activa
SDL_VIDEODRIVER=dummy ANTES de cualquier import de pygame, así que estos
tests funcionan en cualquier entorno headless (CI, sandbox, docker).

Estrategia de testing (decisión de diseño consensuada en Sesión D):

1. **Smoke**: tras dibujar X, la Surface ya no es uniforme de fondo.
2. **Color presente**: en una zona pequeña alrededor de la posición
   esperada, hay al menos un píxel del color esperado (o de un color
   compatible: por ejemplo, en el centro de un drone IDLE esperamos
   ver el verde IDLE; en el borde, el theme.drone_border).
3. **Cambio entre estados**: dibujar el mismo elemento en dos estados
   distintos produce surfaces distintas (al menos un píxel cambia).

NO comprobamos píxel-perfect en posiciones exactas; eso sería frágil
ante el antialiasing de pygame.draw.circle, que difumina los bordes.
Sí podemos comprobar: "en el centro exacto, el color es el de relleno",
porque ahí el antialiasing no actúa.
"""
import pygame
import pytest

from droneplan_viz.domain.drone_state import DroneState
from droneplan_viz.render.geometry import Point
from droneplan_viz.render.sprites import (
    draw_background,
    draw_drone,
    draw_drone_arms,
    draw_edge,
    draw_error_x,
    draw_location,
    draw_package,
    draw_person,
    draw_transporter,
    position_in_transporter,
    position_of_arm,
    _draw_ground_shadow,
    _get_font,
    _render_text,
    _FONT_CACHE,
)
from droneplan_viz.render.theme import (
    Theme,
    color_for_content,
    color_for_drone_state,
    color_for_location,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_surface(w: int = 200, h: int = 200) -> pygame.Surface:
    """Crea una Surface limpia (negra) para los tests.

    Los tests deciden si rellenarla con background antes de dibujar
    (smoke vs comprobación de cambio).
    """
    surf = pygame.Surface((w, h))
    surf.fill((0, 0, 0))
    return surf


def color_at(surf: pygame.Surface, x: int, y: int) -> tuple[int, int, int]:
    """Devuelve el color RGB en (x, y), descartando alpha."""
    r, g, b, _ = surf.get_at((x, y))
    return (r, g, b)


def surface_is_uniform(surf: pygame.Surface) -> bool:
    """True si toda la Surface tiene el mismo color (= nada se dibujó encima)."""
    ref = color_at(surf, 0, 0)
    w, h = surf.get_size()
    # Sampling de 9 puntos: las 4 esquinas, los 4 puntos medios y el centro.
    samples = [
        (0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1),
        (w // 2, 0), (w // 2, h - 1), (0, h // 2), (w - 1, h // 2),
        (w // 2, h // 2),
    ]
    for x, y in samples:
        if color_at(surf, x, y) != ref:
            return False
    return True


def find_color_near(
    surf: pygame.Surface,
    target_rgb: tuple[int, int, int],
    center: tuple[int, int],
    half_box: int = 10,
    tol: int = 0,
) -> bool:
    """¿Aparece `target_rgb` (con tolerancia `tol` por canal) en una caja
    centrada en `center` de lado 2*half_box+1?

    Útil para asserts del tipo "el color de relleno del drone aparece
    cerca del centro esperado", tolerando pequeñas variaciones del
    antialiasing de pygame.
    """
    cx, cy = center
    w, h = surf.get_size()
    for dx in range(-half_box, half_box + 1):
        for dy in range(-half_box, half_box + 1):
            x, y = cx + dx, cy + dy
            if 0 <= x < w and 0 <= y < h:
                px = color_at(surf, x, y)
                if all(abs(px[i] - target_rgb[i]) <= tol for i in range(3)):
                    return True
    return False


# ---------------------------------------------------------------------------
# Background
# ---------------------------------------------------------------------------


class TestDrawBackground:
    def test_rellena_toda_la_surface_con_color_background(self):
        surf = make_surface()
        theme = Theme.default()
        draw_background(surf, theme)
        # Toda la Surface debe ser background ahora.
        assert color_at(surf, 0, 0) == theme.background
        assert color_at(surf, 100, 100) == theme.background
        assert color_at(surf, 199, 199) == theme.background

    def test_surface_es_uniforme_tras_background(self):
        surf = make_surface()
        theme = Theme.default()
        draw_background(surf, theme)
        assert surface_is_uniform(surf)


# ---------------------------------------------------------------------------
# Edges
# ---------------------------------------------------------------------------


class TestDrawEdge:
    def test_pinta_linea_visible_entre_dos_puntos(self):
        surf = make_surface()
        theme = Theme.default()
        draw_background(surf, theme)
        draw_edge(surf, Point(20.0, 100.0), Point(180.0, 100.0), theme)
        # En el punto medio (100, 100) debe haber un píxel del color de edge.
        assert find_color_near(surf, theme.edge, (100, 100), half_box=2)

    def test_no_pinta_fuera_de_la_linea(self):
        surf = make_surface()
        theme = Theme.default()
        draw_background(surf, theme)
        draw_edge(surf, Point(20.0, 100.0), Point(180.0, 100.0), theme)
        # Una zona alejada de la línea sigue siendo background.
        assert color_at(surf, 100, 30) == theme.background


# ---------------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------------


class TestDrawLocation:
    def test_smoke_dibuja_algo(self):
        surf = make_surface()
        theme = Theme.default()
        draw_background(surf, theme)
        ref = color_at(surf, 0, 0)
        draw_location(surf, Point(100.0, 100.0), "casa1", theme)
        # El centro NO debe seguir siendo background.
        assert color_at(surf, 100, 100) != ref

    def test_casa_tiene_color_de_casa_en_el_centro(self):
        surf = make_surface()
        theme = Theme.default()
        draw_background(surf, theme)
        # Etiqueta deshabilitada para que el texto no pise el píxel central.
        draw_location(surf, Point(100.0, 100.0), "casa1", theme, label=False)
        fill, _ = color_for_location("casa1", theme)
        assert color_at(surf, 100, 100) == fill

    def test_hospital_tiene_color_de_hospital(self):
        surf = make_surface()
        theme = Theme.default()
        draw_background(surf, theme)
        draw_location(surf, Point(100.0, 100.0), "hospital", theme, label=False)
        fill, _ = color_for_location("hospital", theme)
        assert color_at(surf, 100, 100) == fill

    def test_deposito_tiene_color_de_deposito(self):
        surf = make_surface()
        theme = Theme.default()
        draw_background(surf, theme)
        draw_location(surf, Point(100.0, 100.0), "deposito1", theme, label=False)
        fill, _ = color_for_location("deposito1", theme)
        assert color_at(surf, 100, 100) == fill

    def test_loc_neutral_tiene_color_neutral(self):
        surf = make_surface()
        theme = Theme.default()
        draw_background(surf, theme)
        draw_location(surf, Point(100.0, 100.0), "punto_x", theme, label=False)
        assert color_at(surf, 100, 100) == theme.loc_fill_neutral

    def test_borde_aparece(self):
        # Surface y centro escalados para acomodar el location_radius=120
        # actual sin recortar el círculo. El test verifica el COMPORTAMIENTO
        # (que hay borde a distancia=radius del centro), no un valor
        # concreto del radio.
        theme = Theme.default()
        side = (theme.location_radius + 30) * 2
        surf = make_surface(side, side)
        cx = cy = side // 2
        draw_background(surf, theme)
        draw_location(surf, Point(float(cx), float(cy)), "casa1", theme, label=False)
        _, border = color_for_location("casa1", theme)
        ell_half_w = int(
            theme.location_sprite_native_size[0]
            * theme.location_sprite_scale
            * theme.location_fallback_scale
        ) // 2
        assert find_color_near(
            surf, border, (cx + ell_half_w, cy), half_box=3, tol=10
        )

    def test_label_se_dibuja_si_label_true(self):
        # Surface y centro escalados al sprite isométrico actual. El test
        # verifica que el label aparece en su zona: debajo del lienzo del
        # sprite.
        theme = Theme.default()
        # Tamaño del lienzo en pantalla (no es 2*location_radius con el
        # sprite isométrico).
        screen_w = theme.location_sprite_native_size[0] * theme.location_sprite_scale
        screen_h = theme.location_sprite_native_size[1] * theme.location_sprite_scale
        side = max(screen_w, screen_h) + 80
        surf_with = make_surface(side, side)
        surf_without = make_surface(side, side)
        cx = cy = side // 2
        draw_background(surf_with, theme)
        draw_background(surf_without, theme)
        draw_location(surf_with, Point(float(cx), float(cy)), "casa1", theme, label=True)
        draw_location(surf_without, Point(float(cx), float(cy)), "casa1", theme, label=False)
        # El label va a y = cy + screen_h//2 + 8 (debajo del borde inferior
        # del lienzo del sprite, según draw_location).
        label_y = cy + (int(screen_h * theme.location_fallback_scale) // 2) + 8
        differ = False
        for y in range(label_y - 10, label_y + 12):
            for x in range(cx - 20, cx + 21):
                if color_at(surf_with, x, y) != color_at(surf_without, x, y):
                    differ = True
                    break
            if differ:
                break
        assert differ

    def test_locs_sin_diferenciacion_son_iguales(self):
        # Con differentiate_locations=False, una casa y un hospital se
        # ven idénticos.
        theme = Theme.default().with_overrides(differentiate_locations=False)
        surf_casa = make_surface()
        surf_hosp = make_surface()
        draw_background(surf_casa, theme)
        draw_background(surf_hosp, theme)
        draw_location(surf_casa, Point(100.0, 100.0), "casa1", theme, label=False)
        draw_location(surf_hosp, Point(100.0, 100.0), "hospital1", theme, label=False)
        # Píxel central debe ser el mismo:
        assert color_at(surf_casa, 100, 100) == color_at(surf_hosp, 100, 100)


# ---------------------------------------------------------------------------
# Drones
# ---------------------------------------------------------------------------


class TestDrawDrone:
    def test_drone_idle_tiene_color_idle_en_el_centro(self):
        surf = make_surface()
        theme = Theme.default()
        draw_background(surf, theme)
        draw_drone(surf, Point(100.0, 100.0), "d1", DroneState.IDLE, theme, label=False)
        assert color_at(surf, 100, 100) == theme.drone_idle

    def test_drone_moving_tiene_color_moving(self):
        surf = make_surface()
        theme = Theme.default()
        draw_background(surf, theme)
        draw_drone(surf, Point(100.0, 100.0), "d1", DroneState.MOVING, theme, label=False)
        assert color_at(surf, 100, 100) == theme.drone_moving

    def test_drone_interacting_tiene_color_interacting(self):
        surf = make_surface()
        theme = Theme.default()
        draw_background(surf, theme)
        draw_drone(surf, Point(100.0, 100.0), "d1", DroneState.INTERACTING, theme, label=False)
        assert color_at(surf, 100, 100) == theme.drone_interacting

    def test_drone_error_tiene_color_rojo_y_X(self):
        # Mismo test, dos asserts: rojo en el centro y X blanca cerca.
        surf = make_surface()
        theme = Theme.default()
        draw_background(surf, theme)
        draw_drone(surf, Point(100.0, 100.0), "d1", DroneState.ERROR, theme, label=False)
        # La X pasa por el centro, así que el centro puede ser blanco
        # (línea de la X) o rojo (relleno bajo la X). Aceptamos cualquiera
        # de los dos por la naturaleza del pixel exacto.
        center = color_at(surf, 100, 100)
        assert center in (theme.drone_error, theme.error_x)
        # Pero el color rojo TIENE que aparecer en algún sitio cerca del
        # centro (las zonas fuera de la X):
        assert find_color_near(surf, theme.drone_error, (100, 100), half_box=8)
        # Y la X blanca también:
        assert find_color_near(surf, theme.error_x, (100, 100), half_box=8)

    def test_estados_distintos_producen_surfaces_distintas(self):
        theme = Theme.default()
        s_idle = make_surface()
        s_error = make_surface()
        draw_background(s_idle, theme)
        draw_background(s_error, theme)
        # En el fallback, IDLE/MOVING/INTERACTING comparten el cuerpo naranja
        # (el estado lo marca la CARA del sprite, no el color). Solo ERROR
        # cambia el color del cuerpo (rojo) — comprobamos esa distinción.
        draw_drone(s_idle, Point(100.0, 100.0), "d", DroneState.IDLE, theme, label=False)
        draw_drone(s_error, Point(100.0, 100.0), "d", DroneState.ERROR, theme, label=False)
        assert color_at(s_idle, 100, 100) != color_at(s_error, 100, 100)


class TestDrawErrorX:
    def test_pinta_pixeles_blancos_en_diagonal(self):
        surf = make_surface()
        theme = Theme.default()
        draw_background(surf, theme)
        draw_error_x(surf, Point(100.0, 100.0), theme)
        # El centro es el cruce de las dos diagonales → blanco.
        assert color_at(surf, 100, 100) == theme.error_x
        # Y puntos arriba-izquierda, arriba-derecha, etc., debe haber
        # píxeles de la X cerca:
        offset = int(theme.drone_radius * 0.5)
        assert find_color_near(
            surf, theme.error_x, (100 + offset, 100 + offset), half_box=3
        )
        assert find_color_near(
            surf, theme.error_x, (100 - offset, 100 - offset), half_box=3
        )


class TestDrawDroneArms:
    def test_sin_brazos_no_dibuja_nada(self):
        # Caso edge: drone explorador sin brazos.
        surf = make_surface()
        theme = Theme.default()
        draw_background(surf, theme)
        before = pygame.surfarray.array3d(surf).copy()
        draw_drone_arms(surf, Point(100.0, 100.0), [], [], theme)
        after = pygame.surfarray.array3d(surf)
        # Surface inalterada.
        assert (before == after).all()

    def test_brazo_libre_dibuja_solo_contorno(self):
        surf = make_surface()
        theme = Theme.default()
        draw_background(surf, theme)
        draw_drone_arms(
            surf,
            drone_pos=Point(100.0, 100.0),
            arm_ids=["izq"],
            holding_ids=[None],
            theme=theme,
        )
        # El brazo está arriba (single arm → angle = -π/2).
        # Posición esperada: (100, 100 - drone_radius).
        arm_y = 100 - theme.drone_radius
        # Algún píxel cerca debe ser el borde del drone (contorno).
        assert find_color_near(surf, theme.drone_border, (100, arm_y), half_box=4)

    def test_brazo_ocupado_dibuja_color_del_content(self):
        surf = make_surface()
        theme = Theme.default()
        draw_background(surf, theme)
        draw_drone_arms(
            surf,
            drone_pos=Point(100.0, 100.0),
            arm_ids=["izq"],
            holding_ids=["medicina"],
            theme=theme,
        )
        arm_y = 100 - theme.drone_radius
        expected_color = color_for_content("medicina", theme)
        assert find_color_near(surf, expected_color, (100, arm_y), half_box=4)


class TestPositionOfArm:
    def test_un_solo_brazo_va_arriba(self):
        theme = Theme.default()
        p = position_of_arm(Point(0.0, 0.0), 0, 1, theme)
        # Arriba: (0, -drone_radius).
        assert p.x == pytest.approx(0.0, abs=1e-6)
        assert p.y == pytest.approx(-float(theme.drone_radius), abs=1e-6)

    def test_dos_brazos_opuestos(self):
        theme = Theme.default()
        a = position_of_arm(Point(0.0, 0.0), 0, 2, theme)
        b = position_of_arm(Point(0.0, 0.0), 1, 2, theme)
        # Suma de vectores: (0, 0).
        assert a.x + b.x == pytest.approx(0.0, abs=1e-6)
        assert a.y + b.y == pytest.approx(0.0, abs=1e-6)

    def test_total_zero_devuelve_drone_pos(self):
        theme = Theme.default()
        center = Point(50.0, 50.0)
        # Defensivo: si el painter pide brazo de un drone sin brazos.
        p = position_of_arm(center, 0, 0, theme)
        assert p == center


# ---------------------------------------------------------------------------
# Paquetes
# ---------------------------------------------------------------------------


class TestDrawPackage:
    def test_dibuja_cuadrado_con_color_del_content(self):
        surf = make_surface()
        theme = Theme.default()
        draw_background(surf, theme)
        draw_package(surf, Point(100.0, 100.0), "medicina", theme)
        expected = color_for_content("medicina", theme)
        assert color_at(surf, 100, 100) == expected

    def test_contenidos_distintos_colores_distintos(self):
        theme = Theme.default()
        s_med = make_surface()
        s_food = make_surface()
        draw_background(s_med, theme)
        draw_background(s_food, theme)
        draw_package(s_med, Point(100.0, 100.0), "medicina", theme)
        draw_package(s_food, Point(100.0, 100.0), "comida", theme)
        assert color_at(s_med, 100, 100) != color_at(s_food, 100, 100)

    def test_content_desconocido_usa_fallback(self):
        # No crashea con un content_id no canónico.
        surf = make_surface()
        theme = Theme.default()
        draw_background(surf, theme)
        draw_package(surf, Point(100.0, 100.0), "juguete", theme)
        # Surface ya no es background:
        assert color_at(surf, 100, 100) != theme.background

    def test_label_opcional_default_false(self):
        # Por defecto draw_package no pinta etiqueta. Compruebo que con
        # label=False NO añade nada debajo del cuadrado.
        surf = make_surface()
        theme = Theme.default()
        draw_background(surf, theme)
        draw_package(surf, Point(100.0, 100.0), "medicina", theme, label=False)
        # Píxel claramente debajo del package (a 15px) debe seguir siendo background.
        # 15 > package_size/2 (14/2 = 7) más 7 de margen → fuera del cuadrado.
        assert color_at(surf, 100, 115) == theme.background


# ---------------------------------------------------------------------------
# Transportadores
# ---------------------------------------------------------------------------


class TestDrawTransporter:
    def test_dibuja_rectangulo_con_color_de_transp(self):
        surf = make_surface()
        theme = Theme.default()
        draw_background(surf, theme)
        draw_transporter(
            surf,
            position=Point(100.0, 100.0),
            transporter_id="t1",
            contents_count=0,
            capacity=4,
            theme=theme,
            label=False,
        )
        # El centro del rect debe ser transporter_fill.
        # Pero ojo: si la indicador-bar pasa por el centro, podría
        # superponerse. Con contents_count=0 NO se dibuja barra.
        assert color_at(surf, 100, 100) == theme.transporter_fill

    def test_capacidad_llena_no_dibuja_barra(self):
        # La barra de ocupación se eliminó: el nivel de llenado lo transmite
        # el sprite carrier_N. Con el carrier lleno NO debe aparecer la
        # franja verde que antes dibujaba la barra.
        surf = make_surface()
        theme = Theme.default()
        draw_background(surf, theme)
        draw_transporter(
            surf,
            position=Point(100.0, 100.0),
            transporter_id="t1",
            contents_count=4,
            capacity=4,
            theme=theme,
            label=False,
        )
        assert not find_color_near(surf, theme.drone_idle, (100, 110), half_box=3)

    def test_capacidad_parcial_no_dibuja_barra(self):
        surf = make_surface()
        theme = Theme.default()
        draw_background(surf, theme)
        draw_transporter(
            surf,
            position=Point(100.0, 100.0),
            transporter_id="t1",
            contents_count=2,
            capacity=4,
            theme=theme,
            label=False,
        )
        assert not find_color_near(surf, theme.drone_interacting, (100, 110), half_box=3)

    def test_capacidad_cero_no_dibuja_barra(self):
        # Caso edge: transp con capacity=0 (degenerado pero válido).
        surf = make_surface()
        theme = Theme.default()
        draw_background(surf, theme)
        draw_transporter(
            surf,
            position=Point(100.0, 100.0),
            transporter_id="t1",
            contents_count=0,
            capacity=0,
            theme=theme,
            label=False,
        )
        # En la zona de la barra no debe haber verde ni ámbar:
        assert not find_color_near(
            surf, theme.drone_idle, (100, 110), half_box=3
        )


class TestPositionInTransporter:
    def test_un_solo_paquete_al_centro(self):
        theme = Theme.default()
        center = Point(100.0, 100.0)
        p = position_in_transporter(center, 0, 1, theme)
        assert p == center

    def test_dos_paquetes_se_separan_horizontalmente(self):
        theme = Theme.default()
        center = Point(100.0, 100.0)
        a = position_in_transporter(center, 0, 2, theme)
        b = position_in_transporter(center, 1, 2, theme)
        assert a.x < b.x
        # Misma y:
        assert a.y == pytest.approx(b.y)

    def test_total_cero_devuelve_centro_defensivamente(self):
        theme = Theme.default()
        p = position_in_transporter(Point(50.0, 50.0), 0, 0, theme)
        assert p == Point(50.0, 50.0)


# ---------------------------------------------------------------------------
# Personas
# ---------------------------------------------------------------------------


class TestDrawPerson:
    def test_persona_centro_es_color_persona(self):
        surf = make_surface()
        theme = Theme.default()
        draw_background(surf, theme)
        draw_person(surf, Point(100.0, 100.0), "p1", theme=theme, label=False)
        assert color_at(surf, 100, 100) == theme.person_fill

    def test_no_dibuja_anillo_alrededor(self):
        # Ya no se dibuja ningún anillo/indicador alrededor de la persona:
        # el estado (espera vs. entrega) lo transmite el sprite.
        surf = make_surface()
        theme = Theme.default()
        draw_background(surf, theme)
        draw_person(surf, Point(100.0, 100.0), "p1", theme=theme, label=False)
        ring_r = theme.person_radius + 3
        assert not find_color_near(
            surf, theme.needs_indicator, (100 + ring_r, 100), half_box=4
        )


# ---------------------------------------------------------------------------
# Integración: dibujar todo en una sola Surface
# ---------------------------------------------------------------------------


class TestIntegracionMixta:
    """Tests de que varias funciones pueden coexistir sin pisarse mal."""

    def test_drone_sobre_location_drone_gana(self):
        # Pintar location en (100, 100), luego drone también en (100, 100).
        # El drone debe ser visible en el centro (último dibujado, gana).
        surf = make_surface()
        theme = Theme.default()
        draw_background(surf, theme)
        draw_location(surf, Point(100.0, 100.0), "casa1", theme, label=False)
        draw_drone(surf, Point(100.0, 100.0), "d1", DroneState.IDLE, theme, label=False)
        # El centro ahora es color de drone IDLE, no de casa.
        assert color_at(surf, 100, 100) == theme.drone_idle

    def test_multiples_drones_no_crashean(self):
        surf = make_surface(400, 400)
        theme = Theme.default()
        draw_background(surf, theme)
        for i, state in enumerate([DroneState.IDLE, DroneState.MOVING, DroneState.INTERACTING, DroneState.ERROR]):
            draw_drone(
                surf,
                Point(100.0 + i * 80, 200.0),
                f"d{i}",
                state,
                theme,
            )
        # Smoke: no crashea y los 4 colores de estado están todos
        # presentes en la Surface. surface_is_uniform no aplica aquí
        # porque solo muestrea 9 puntos en los bordes y esquinas, y los
        # drones están en y=200 (zona central).
        assert find_color_near(surf, theme.drone_idle, (100, 200), half_box=4)
        assert find_color_near(surf, theme.drone_moving, (180, 200), half_box=4)
        assert find_color_near(surf, theme.drone_interacting, (260, 200), half_box=4)
        assert find_color_near(surf, theme.drone_error, (340, 200), half_box=4)


# ---------------------------------------------------------------------------
# Selección de cara direccional y por estado (modelo de capas)
# ---------------------------------------------------------------------------


class TestDroneFaceSelection:
    """_face_key_for_angle y _drone_face_key: caras por dirección y estado."""

    def test_angulo_norte(self):
        from droneplan_viz.render.sprites import _face_key_for_angle
        assert _face_key_for_angle(90.0) == "face_N"

    def test_angulo_sur(self):
        from droneplan_viz.render.sprites import _face_key_for_angle
        assert _face_key_for_angle(270.0) == "face_S"

    def test_angulo_este_cae_en_diagonal(self):
        # E (0°) no tiene cara propia: cae en NE (sector [0,60)).
        from droneplan_viz.render.sprites import _face_key_for_angle
        assert _face_key_for_angle(0.0) == "face_NE"

    def test_angulo_oeste_cae_en_diagonal(self):
        # O (180°) cae en SW (sector [180,240)).
        from droneplan_viz.render.sprites import _face_key_for_angle
        assert _face_key_for_angle(180.0) == "face_SW"

    def test_diagonales(self):
        from droneplan_viz.render.sprites import _face_key_for_angle
        assert _face_key_for_angle(135.0) == "face_NW"
        assert _face_key_for_angle(45.0) == "face_NE"
        assert _face_key_for_angle(225.0) == "face_SW"
        assert _face_key_for_angle(315.0) == "face_SE"

    def test_angulo_se_normaliza(self):
        # Ángulos fuera de [0,360) se normalizan.
        from droneplan_viz.render.sprites import _face_key_for_angle
        assert _face_key_for_angle(90.0 + 360.0) == "face_N"
        assert _face_key_for_angle(-90.0) == "face_S"  # -90 ≡ 270

    def test_error_tiene_prioridad(self):
        from droneplan_viz.render.sprites import _drone_face_key
        # Aunque haya dirección, ERROR manda.
        key = _drone_face_key(
            DroneState.ERROR, direction_angle=90.0, finished=False,
        )
        assert key == "face_error1"

    def test_movimiento_usa_direccion(self):
        from droneplan_viz.render.sprites import _drone_face_key
        key = _drone_face_key(
            DroneState.MOVING, direction_angle=90.0, finished=False,
        )
        assert key == "face_N"

    def test_terminado_usa_idle(self):
        from droneplan_viz.render.sprites import _drone_face_key
        key = _drone_face_key(
            DroneState.IDLE, direction_angle=None, finished=True,
        )
        assert key == "face_idle"

    def test_espera_usa_face_neutra(self):
        from droneplan_viz.render.sprites import _drone_face_key
        key = _drone_face_key(
            DroneState.IDLE, direction_angle=None, finished=False,
        )
        assert key == "face"


# ---------------------------------------------------------------------------
# Sombras de objetos (sombra ovalada de contacto bajo transporters y cajas)
# ---------------------------------------------------------------------------


def _luma(rgb: tuple[int, int, int]) -> int:
    """Brillo aproximado (suma de canales) para comparar oscurecimiento."""
    return rgb[0] + rgb[1] + rgb[2]


class TestGroundShadow:
    """_draw_ground_shadow: elipse plana translúcida anclada a la base."""

    def test_oscurece_bajo_la_base_del_objeto(self):
        surf = make_surface()
        theme = Theme.default()
        draw_background(surf, theme)
        center = Point(100.0, 100.0)
        obj_w, obj_h = 48, 30
        _draw_ground_shadow(surf, center, (obj_w, obj_h), theme)
        # Justo bajo la base (cy + h/2 + offset) la sombra está rellena:
        sy = 100 + obj_h // 2 + theme.shadow_offset_y
        assert _luma(color_at(surf, 100, sy)) < _luma(theme.background)

    def test_no_pinta_sobre_el_objeto_ni_lejos(self):
        surf = make_surface()
        theme = Theme.default()
        draw_background(surf, theme)
        center = Point(100.0, 100.0)
        _draw_ground_shadow(surf, center, (48, 30), theme)
        # El centro del objeto (muy por encima de la base) sigue siendo fondo:
        assert color_at(surf, 100, 100) == theme.background
        # Una esquina lejana también:
        assert color_at(surf, 5, 5) == theme.background

    def test_es_translucida_no_negra_pura(self):
        # La sombra mezcla con el fondo (alpha < 255): NO debe ser (0,0,0).
        surf = make_surface()
        theme = Theme.default()
        draw_background(surf, theme)
        center = Point(100.0, 100.0)
        obj_h = 30
        _draw_ground_shadow(surf, center, (48, obj_h), theme)
        sy = 100 + obj_h // 2 + theme.shadow_offset_y
        px = color_at(surf, 100, sy)
        assert px != (0, 0, 0)
        assert _luma(px) < _luma(theme.background)

    def test_shadow_enabled_false_no_dibuja_nada(self):
        surf = make_surface()
        theme = Theme.default().with_overrides(shadow_enabled=False)
        draw_background(surf, theme)
        center = Point(100.0, 100.0)
        _draw_ground_shadow(surf, center, (48, 30), theme)
        assert surface_is_uniform(surf)
        assert color_at(surf, 100, 115) == theme.background

    def test_alpha_mayor_oscurece_mas(self):
        theme_lo = Theme.default().with_overrides(shadow_alpha=40)
        theme_hi = Theme.default().with_overrides(shadow_alpha=200)
        center = Point(100.0, 100.0)
        s_lo = make_surface(); draw_background(s_lo, theme_lo)
        s_hi = make_surface(); draw_background(s_hi, theme_hi)
        _draw_ground_shadow(s_lo, center, (48, 30), theme_lo)
        _draw_ground_shadow(s_hi, center, (48, 30), theme_hi)
        sy = 100 + 30 // 2 + theme_lo.shadow_offset_y
        assert _luma(color_at(s_hi, 100, sy)) < _luma(color_at(s_lo, 100, sy))


class TestDrawObjectsWithShadow:
    """draw_package / draw_transporter con flag shadow."""

    def test_package_shadow_true_oscurece_bajo_la_caja(self):
        surf = make_surface()
        theme = Theme.default()
        draw_background(surf, theme)
        side = theme.package_size
        # Punto bajo la base de la caja, fuera del cuadrado dibujado encima.
        below_y = 100 + side // 2 + 2
        draw_package(surf, Point(100.0, 100.0), "medicina", theme, shadow=True)
        assert _luma(color_at(surf, 100, below_y)) < _luma(theme.background)

    def test_package_shadow_false_no_oscurece_bajo_la_caja(self):
        surf = make_surface()
        theme = Theme.default()
        draw_background(surf, theme)
        side = theme.package_size
        below_y = 100 + side // 2 + 2
        draw_package(surf, Point(100.0, 100.0), "medicina", theme, shadow=False)
        assert color_at(surf, 100, below_y) == theme.background

    def test_package_shadow_default_es_false(self):
        # Compatibilidad: sin pasar shadow, no hay sombra.
        surf = make_surface()
        theme = Theme.default()
        draw_background(surf, theme)
        side = theme.package_size
        below_y = 100 + side // 2 + 2
        draw_package(surf, Point(100.0, 100.0), "medicina", theme)
        assert color_at(surf, 100, below_y) == theme.background

    def test_package_shadow_no_tapa_el_centro_de_la_caja(self):
        # La caja se dibuja ENCIMA de la sombra: su centro conserva color.
        surf = make_surface()
        theme = Theme.default()
        draw_background(surf, theme)
        draw_package(surf, Point(100.0, 100.0), "medicina", theme, shadow=True)
        expected = color_for_content("medicina", theme)
        assert color_at(surf, 100, 100) == expected

    def test_transporter_shadow_true_oscurece_bajo_el_transp(self):
        surf = make_surface()
        theme = Theme.default()
        draw_background(surf, theme)
        h = theme.transporter_height
        below_y = 100 + h // 2 + 3
        draw_transporter(
            surf, Point(100.0, 100.0), "t1", contents_count=0, capacity=4,
            theme=theme, label=False, shadow=True,
        )
        assert _luma(color_at(surf, 100, below_y)) < _luma(theme.background)

    def test_transporter_shadow_no_tapa_el_centro(self):
        # El rectángulo del transp se dibuja sobre la sombra: centro = fill.
        surf = make_surface()
        theme = Theme.default()
        draw_background(surf, theme)
        draw_transporter(
            surf, Point(100.0, 100.0), "t1", contents_count=0, capacity=4,
            theme=theme, label=False, shadow=True,
        )
        assert color_at(surf, 100, 100) == theme.transporter_fill


# ---------------------------------------------------------------------------
# Fuente de las etiquetas (Monogram vía font_path) y caché de fuentes
# ---------------------------------------------------------------------------


def _a_real_ttf_path() -> str:
    """Ruta a un .ttf real (la fuente por defecto que empaqueta pygame).

    Evita depender de un asset de la app (Monogram vive en droneplan_viz_app);
    para probar la rama font_path basta cualquier .ttf válido.
    """
    import os
    return os.path.join(
        os.path.dirname(pygame.font.__file__), pygame.font.get_default_font()
    )


class TestFontSelection:
    def test_get_font_devuelve_un_font(self):
        f = _get_font(16)
        assert isinstance(f, pygame.font.Font)

    def test_get_font_cachea_mismo_objeto(self):
        _FONT_CACHE.clear()
        a = _get_font(16, None, None)
        b = _get_font(16, None, None)
        assert a is b  # mismo objeto cacheado

    def test_get_font_claves_distintas_objetos_distintos(self):
        _FONT_CACHE.clear()
        a = _get_font(16, None, None)
        b = _get_font(18, None, None)  # distinto size
        assert a is not b

    def test_font_path_valido_se_usa(self):
        _FONT_CACHE.clear()
        ttf = _a_real_ttf_path()
        f = _get_font(16, None, ttf)
        assert isinstance(f, pygame.font.Font)
        # Quedó cacheado bajo la clave con font_path:
        assert (None, ttf, 16) in _FONT_CACHE

    def test_font_path_invalido_cae_a_sysfont_sin_romper(self):
        _FONT_CACHE.clear()
        # Ruta inexistente: NO debe propagar excepción; cae a SysFont.
        f = _get_font(16, None, "/no/existe/fuente.ttf")
        assert isinstance(f, pygame.font.Font)

    def test_render_text_con_font_path_produce_surface(self):
        ttf = _a_real_ttf_path()
        theme = Theme.default()
        surf = _render_text("dron_01", theme.text, 16, theme.font_name, ttf)
        assert isinstance(surf, pygame.Surface)
        assert surf.get_width() > 0 and surf.get_height() > 0

    def test_render_text_sobrevive_a_font_quit(self):
        # Regresión: un Font cacheado tras pygame.font.quit() quedaba
        # inválido y _render_text lanzaba "Invalid font". Debe auto-sanar.
        theme = Theme.default()
        _ = _render_text("x", theme.text, 16, theme.font_name, None)
        pygame.font.quit()  # invalida los Font cacheados
        # No debe lanzar: _ensure_font_ready reinicia y purga el caché.
        surf = _render_text("y", theme.text, 16, theme.font_name, None)
        assert isinstance(surf, pygame.Surface)


class TestThemeFontFields:
    def test_font_path_default_none(self):
        assert Theme.default().font_path is None

    def test_font_sizes_son_legibles(self):
        # Tras el cambio a Monogram, las etiquetas usan 16px (tamaño nítido).
        t = Theme.default()
        assert t.font_size_label == 16
        assert t.font_size_id == 16

    def test_with_overrides_fija_font_path(self):
        t = Theme.default().with_overrides(font_path="/algun/path.ttf")
        assert t.font_path == "/algun/path.ttf"


# ---------------------------------------------------------------------------
# Outline de color por contenido en draw_package (camino sprite). box.png es
# monocromo; el outline recupera la distinción por tipo en el suelo.
# ---------------------------------------------------------------------------
from droneplan_viz.render.sprite_manager import SpriteManager  # noqa: E402


def _png(path, size, color):
    s = pygame.Surface(size, pygame.SRCALPHA)
    s.fill(color)
    pygame.image.save(s, str(path))


class TestPackageOutline:
    def _sm_con_box(self, tmp_path):
        # box.png monocromo (gris), tamaño suficiente para escalar a package_size.
        _png(tmp_path / "box.png", (24, 24), (150, 150, 150, 255))
        theme = Theme.default().with_overrides(
            sprite_dir=str(tmp_path),
            content_colors={"medicina": (255, 0, 255)},  # color de outline conocido
        )
        return SpriteManager(theme), theme

    def test_outline_pinta_color_de_contenido_en_el_borde(self, tmp_path):
        sm, theme = self._sm_con_box(tmp_path)
        surf = make_surface()
        draw_package(surf, Point(100.0, 100.0), "medicina", theme, sprite_manager=sm)
        half = theme.package_size // 2
        # El píxel del borde superior de la caja debe ser el color de outline.
        top_edge = surf.get_at((100, 100 - half))[:3]
        assert top_edge == (255, 0, 255)

    def test_outline_width_cero_no_pinta_borde(self, tmp_path):
        _png(tmp_path / "box.png", (24, 24), (150, 150, 150, 255))
        theme = Theme.default().with_overrides(
            sprite_dir=str(tmp_path),
            content_colors={"medicina": (255, 0, 255)},
            package_outline_width=0,
        )
        sm = SpriteManager(theme)
        surf = make_surface()
        draw_package(surf, Point(100.0, 100.0), "medicina", theme, sprite_manager=sm)
        half = theme.package_size // 2
        # Sin outline: el borde es el gris del box.png, no el magenta.
        assert surf.get_at((100, 100 - half))[:3] != (255, 0, 255)


class TestDroneMultiBox:
    """Camino sprite: el dron con >=2 cajas usa obj 'box2' (dos cajas),
    distinto de 'box' (una). El estado de 2 cajas por brazo del dominio se
    ve como dos cajas, no como una con número."""

    def test_box2_composite_difiere_de_box(self):
        theme = Theme.default()
        sm = SpriteManager(theme)
        r = theme.drone_radius
        c1 = sm.get_drone_composite(
            body="drone", face="face", obj="box", dest_size=(2 * r, 2 * r)
        )
        c2 = sm.get_drone_composite(
            body="drone", face="face", obj="box2", dest_size=(2 * r, 2 * r)
        )
        if c1 is None or c2 is None:
            pytest.skip("sin assets de dron/caja")
        assert pygame.image.tostring(c1, "RGBA") != pygame.image.tostring(c2, "RGBA")

    def test_draw_drone_box_count_dos_usa_box2(self):
        theme = Theme.default()
        r = theme.drone_radius
        if SpriteManager(theme).get_drone_composite(
            body="drone", face="face", obj="box", dest_size=(2 * r, 2 * r)
        ) is None:
            pytest.skip("sin assets")
        sm = SpriteManager(theme)
        surf = make_surface()
        draw_drone(
            surf, Point(100.0, 100.0), "d", DroneState.IDLE, theme,
            sprite_manager=sm, held_object="box", box_count=2, label=False,
        )
        keys = [k for k in sm._composite_cache if len(k) > 3 and k[3] == "box2"]
        assert len(keys) >= 1

    def test_draw_drone_box_count_uno_usa_box(self):
        theme = Theme.default()
        r = theme.drone_radius
        if SpriteManager(theme).get_drone_composite(
            body="drone", face="face", obj="box", dest_size=(2 * r, 2 * r)
        ) is None:
            pytest.skip("sin assets")
        sm = SpriteManager(theme)
        surf = make_surface()
        draw_drone(
            surf, Point(100.0, 100.0), "d", DroneState.IDLE, theme,
            sprite_manager=sm, held_object="box", box_count=1, label=False,
        )
        keys = [k for k in sm._composite_cache if len(k) > 3 and k[3] == "box2"]
        assert len(keys) == 0
