"""Tests para droneplan_viz.render.geometry.

Todos sin pygame. Verifican propiedades aritméticas y de inmutabilidad
de los tipos puros del módulo: Point, lerp, lerp_point, ease_in_out_cubic,
ViewBox.

Convención: cada clase de tests cubre un tipo o función. Los asserts usan
pytest.approx para floats por seguridad, aunque para los casos en los
que el resultado es exacto (lerp(a, b, 0.0) = a) se compara directamente.
"""
import math

import pytest
from dataclasses import FrozenInstanceError

from droneplan_viz.render.geometry import (
    Point,
    ViewBox,
    ease_in_out_cubic,
    lateral_tilt_deg,
    lerp,
    lerp_point,
    tilt_envelope,
)


class TestPoint:
    """Point: par (x, y) inmutable, comparable por valor, con suma y resta."""

    def test_construccion_simple(self):
        p = Point(1.0, 2.0)
        assert p.x == 1.0
        assert p.y == 2.0

    def test_acepta_ints_pero_los_guarda_como_estan(self):
        # Aunque el docstring dice "tolerar interpolación", no forzamos
        # cast a float: los dataclasses no validan tipos en runtime y eso
        # encaja con la filosofía del proyecto "el tipo no valida".
        p = Point(3, 4)
        assert p.x == 3
        assert p.y == 4

    def test_igualdad_estructural(self):
        assert Point(1.0, 2.0) == Point(1.0, 2.0)
        assert Point(1.0, 2.0) != Point(1.0, 3.0)
        assert Point(1.0, 2.0) != Point(2.0, 2.0)

    def test_hashable(self):
        # frozen+slots ⇒ hashable. Útil para set/dict de Points.
        p1 = Point(1.0, 2.0)
        p2 = Point(1.0, 2.0)
        assert hash(p1) == hash(p2)
        s = {p1, p2}
        assert len(s) == 1

    def test_inmutable_lanza_excepcion_al_reasignar(self):
        p = Point(1.0, 2.0)
        with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
            p.x = 99.0  # type: ignore[misc]

    def test_inmutable_lanza_excepcion_al_anadir_atributo(self):
        p = Point(1.0, 2.0)
        with pytest.raises((AttributeError, TypeError)):
            p.z = 5.0  # type: ignore[attr-defined]

    def test_suma_componente_a_componente(self):
        assert Point(1.0, 2.0) + Point(10.0, 20.0) == Point(11.0, 22.0)

    def test_resta_componente_a_componente(self):
        assert Point(5.0, 7.0) - Point(2.0, 3.0) == Point(3.0, 4.0)

    def test_scaled_factor_positivo(self):
        assert Point(2.0, 3.0).scaled(2.0) == Point(4.0, 6.0)

    def test_scaled_factor_cero(self):
        assert Point(2.0, 3.0).scaled(0.0) == Point(0.0, 0.0)

    def test_scaled_factor_negativo(self):
        assert Point(2.0, 3.0).scaled(-1.0) == Point(-2.0, -3.0)

    def test_as_int_tuple_redondea_correctamente(self):
        # round() en Python usa banker's rounding (a par); cubrimos los
        # casos típicos sin caer en el caso ambiguo de 0.5.
        assert Point(1.4, 2.6).as_int_tuple() == (1, 3)
        assert Point(-1.4, -2.6).as_int_tuple() == (-1, -3)
        assert Point(0.0, 0.0).as_int_tuple() == (0, 0)

    def test_as_int_tuple_devuelve_ints_no_floats(self):
        result = Point(1.5, 2.5).as_int_tuple()
        assert isinstance(result[0], int)
        assert isinstance(result[1], int)


class TestLerp:
    """lerp(a, b, t): interpolación lineal saneada en [0, 1]."""

    def test_t_cero_devuelve_a(self):
        assert lerp(10.0, 20.0, 0.0) == 10.0

    def test_t_uno_devuelve_b(self):
        assert lerp(10.0, 20.0, 1.0) == 20.0

    def test_t_intermedio(self):
        assert lerp(0.0, 10.0, 0.5) == pytest.approx(5.0)
        assert lerp(10.0, 20.0, 0.25) == pytest.approx(12.5)

    def test_t_negativo_se_satura_a_cero(self):
        # Defensa en profundidad: t fuera de rango no rompe.
        assert lerp(10.0, 20.0, -0.5) == 10.0
        assert lerp(10.0, 20.0, -100.0) == 10.0

    def test_t_mayor_que_uno_se_satura(self):
        assert lerp(10.0, 20.0, 1.5) == 20.0
        assert lerp(10.0, 20.0, 100.0) == 20.0

    def test_a_igual_b(self):
        # Caso degenerado: extremos coinciden, resultado constante.
        assert lerp(7.0, 7.0, 0.0) == 7.0
        assert lerp(7.0, 7.0, 0.5) == 7.0
        assert lerp(7.0, 7.0, 1.0) == 7.0

    def test_lerp_con_negativos(self):
        assert lerp(-10.0, 10.0, 0.5) == pytest.approx(0.0)
        assert lerp(-5.0, -15.0, 0.5) == pytest.approx(-10.0)


class TestLerpPoint:
    """lerp_point: interpolación lineal componente a componente sobre Point."""

    def test_t_cero_devuelve_a(self):
        a = Point(0.0, 0.0)
        b = Point(10.0, 20.0)
        assert lerp_point(a, b, 0.0) == a

    def test_t_uno_devuelve_b(self):
        a = Point(0.0, 0.0)
        b = Point(10.0, 20.0)
        assert lerp_point(a, b, 1.0) == b

    def test_t_medio(self):
        a = Point(0.0, 0.0)
        b = Point(10.0, 20.0)
        assert lerp_point(a, b, 0.5) == Point(5.0, 10.0)

    def test_t_negativo_se_satura(self):
        a = Point(0.0, 0.0)
        b = Point(10.0, 20.0)
        assert lerp_point(a, b, -0.3) == a

    def test_t_mayor_uno_se_satura(self):
        a = Point(0.0, 0.0)
        b = Point(10.0, 20.0)
        assert lerp_point(a, b, 2.0) == b

    def test_componente_a_componente_independiente(self):
        # X interpola entre 0 y 100, Y se mantiene.
        a = Point(0.0, 50.0)
        b = Point(100.0, 50.0)
        result = lerp_point(a, b, 0.25)
        assert result.x == pytest.approx(25.0)
        assert result.y == pytest.approx(50.0)

    def test_devuelve_point(self):
        result = lerp_point(Point(0.0, 0.0), Point(1.0, 1.0), 0.5)
        assert isinstance(result, Point)


class TestEaseInOutCubic:
    """ease_in_out_cubic: easing C1 simétrico entre [0, 1] → [0, 1]."""

    def test_f_de_cero(self):
        assert ease_in_out_cubic(0.0) == 0.0

    def test_f_de_uno(self):
        assert ease_in_out_cubic(1.0) == 1.0

    def test_f_de_medio_es_medio(self):
        # Punto de simetría: f(0.5) = 0.5 exacto.
        assert ease_in_out_cubic(0.5) == pytest.approx(0.5)

    def test_negativos_saturan_a_cero(self):
        assert ease_in_out_cubic(-0.1) == 0.0
        assert ease_in_out_cubic(-100.0) == 0.0

    def test_mayores_uno_saturan(self):
        assert ease_in_out_cubic(1.1) == 1.0
        assert ease_in_out_cubic(100.0) == 1.0

    def test_monotonia_creciente(self):
        # En 20 puntos equiespaciados, cada uno debe ser >= al anterior.
        ts = [i / 20.0 for i in range(21)]
        ys = [ease_in_out_cubic(t) for t in ts]
        for prev, curr in zip(ys, ys[1:]):
            assert curr >= prev

    def test_no_lineal_estricto(self):
        # f(0.25) NO debe ser 0.25 (eso sería lineal puro) ni 0.0625 (eso
        # sería cúbico puro). El easing es una MEZCLA cúbico+lineal, así
        # que f(0.25) cae entre ambos extremos: 0.0625 < f(0.25) < 0.25.
        v = ease_in_out_cubic(0.25)
        assert v != pytest.approx(0.25), "no debe ser lineal puro"
        assert v != pytest.approx(0.0625), "no debe ser cúbico puro"
        assert 0.0625 < v < 0.25, f"debe estar entre cúbico y lineal: {v}"

    def test_simetria_respecto_a_medio(self):
        # f(t) + f(1-t) = 1 para todo t en [0, 1]. Propiedad fundamental
        # del easing in-out simétrico.
        for t in [0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0]:
            assert ease_in_out_cubic(t) + ease_in_out_cubic(1.0 - t) == pytest.approx(1.0)

    def test_arranque_y_final_suaves(self):
        # La derivada en los extremos es 0, así que valores muy cerca de
        # 0 o de 1 dan resultados muy cerca de 0 o 1 respectivamente
        # (más cerca que un lerp lineal).
        assert ease_in_out_cubic(0.05) < 0.05   # arranque más lento que lineal
        assert ease_in_out_cubic(0.95) > 0.95   # final más cerca de 1 que lineal


class TestViewBox:
    """ViewBox: rectángulo lógico con utilidades de mapeo a Surface."""

    def test_construccion_directa(self):
        vb = ViewBox(left=10.0, top=20.0, width=100.0, height=50.0)
        assert vb.left == 10.0
        assert vb.top == 20.0
        assert vb.width == 100.0
        assert vb.height == 50.0

    def test_inmutable(self):
        vb = ViewBox(0.0, 0.0, 100.0, 100.0)
        with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
            vb.left = 5.0  # type: ignore[misc]

    def test_right_y_bottom_derivados(self):
        vb = ViewBox(left=10.0, top=20.0, width=100.0, height=50.0)
        assert vb.right == 110.0
        assert vb.bottom == 70.0

    def test_center(self):
        vb = ViewBox(left=0.0, top=0.0, width=100.0, height=50.0)
        assert vb.center == Point(50.0, 25.0)

    def test_fit_into_sin_padding(self):
        vb = ViewBox.fit_into((800, 600))
        assert vb.left == 0.0
        assert vb.top == 0.0
        assert vb.width == 800.0
        assert vb.height == 600.0

    def test_fit_into_con_padding(self):
        vb = ViewBox.fit_into((800, 600), padding=20)
        assert vb.left == 20.0
        assert vb.top == 20.0
        assert vb.width == 760.0   # 800 - 2*20
        assert vb.height == 560.0  # 600 - 2*20

    def test_fit_into_padding_excesivo_produce_dimensiones_negativas(self):
        # Caso límite documentado: padding > mitad de la dimensión.
        # No es responsabilidad del ViewBox manejarlo; el consumidor
        # debe estar preparado o usar padding razonable.
        vb = ViewBox.fit_into((100, 100), padding=60)
        assert vb.width < 0
        assert vb.height < 0

    def test_inset_reduce_uniformemente(self):
        vb = ViewBox(left=10.0, top=10.0, width=100.0, height=100.0)
        inset = vb.inset(5.0)
        assert inset.left == 15.0
        assert inset.top == 15.0
        assert inset.width == 90.0
        assert inset.height == 90.0

    def test_inset_devuelve_nuevo_no_muta(self):
        vb = ViewBox(left=10.0, top=10.0, width=100.0, height=100.0)
        vb.inset(5.0)
        # El original sigue intacto:
        assert vb.left == 10.0
        assert vb.width == 100.0

    def test_inset_cero_es_identidad(self):
        vb = ViewBox(left=10.0, top=10.0, width=100.0, height=100.0)
        assert vb.inset(0.0) == vb

    def test_contains_dentro(self):
        vb = ViewBox(left=0.0, top=0.0, width=100.0, height=100.0)
        assert vb.contains(Point(50.0, 50.0))
        assert vb.contains(Point(0.0, 0.0))   # esquina superior izquierda
        assert vb.contains(Point(100.0, 100.0))  # esquina inferior derecha
        assert vb.contains(Point(0.0, 50.0))  # borde izquierdo

    def test_contains_fuera(self):
        vb = ViewBox(left=0.0, top=0.0, width=100.0, height=100.0)
        assert not vb.contains(Point(-0.1, 50.0))
        assert not vb.contains(Point(101.0, 50.0))
        assert not vb.contains(Point(50.0, -0.1))
        assert not vb.contains(Point(50.0, 101.0))

    def test_igualdad_estructural(self):
        a = ViewBox(0.0, 0.0, 100.0, 100.0)
        b = ViewBox(0.0, 0.0, 100.0, 100.0)
        assert a == b

    def test_diferente_si_algun_campo_distinto(self):
        a = ViewBox(0.0, 0.0, 100.0, 100.0)
        b = ViewBox(0.0, 0.0, 100.0, 101.0)
        assert a != b


class TestTiltEnvelope:
    """tilt_envelope: envoltura tanh 0→1→0 para la inclinación del dron.

    Sustituye a sin(π·t). Propiedades clave: extremos exactos en 0, pico
    exacto 1 en el centro, simetría, y ENTRADA/SALIDA MÁS SUAVE que el seno
    (pendiente ~0 en los bordes en vez de ±π).
    """

    def test_extremos_exactamente_cero(self):
        assert tilt_envelope(0.0) == 0.0
        assert tilt_envelope(1.0) == 0.0

    def test_pico_unitario_en_el_centro(self):
        assert tilt_envelope(0.5) == pytest.approx(1.0)

    def test_satura_fuera_de_rango(self):
        assert tilt_envelope(-0.3) == 0.0
        assert tilt_envelope(-100.0) == 0.0
        assert tilt_envelope(1.3) == 0.0
        assert tilt_envelope(100.0) == 0.0

    def test_simetrica_respecto_a_medio(self):
        for t in (0.1, 0.2, 0.35, 0.45):
            assert tilt_envelope(t) == pytest.approx(tilt_envelope(1.0 - t))

    def test_rango_en_cero_uno(self):
        for i in range(0, 101):
            v = tilt_envelope(i / 100.0)
            assert 0.0 <= v <= 1.0 + 1e-9

    def test_sube_hasta_el_centro_y_baja_despues(self):
        # Monótona creciente en [0, 0.5] y decreciente en [0.5, 1].
        rising = [tilt_envelope(i / 50.0) for i in range(0, 26)]
        falling = [tilt_envelope(0.5 + i / 50.0) for i in range(0, 26)]
        assert all(b >= a - 1e-9 for a, b in zip(rising, rising[1:]))
        assert all(b <= a + 1e-9 for a, b in zip(falling, falling[1:]))

    def test_arranque_mas_suave_que_el_seno(self):
        # La pendiente numérica cerca de t=0 debe ser mucho menor que la del
        # seno (π ≈ 3.14): ahí está la mejora de suavidad pedida.
        h = 1e-4
        slope = (tilt_envelope(h) - tilt_envelope(0.0)) / h
        assert slope < 1.0  # holgadamente por debajo de π

    def test_steepness_mayor_aplana_la_meseta(self):
        # Con k mayor la envoltura es más plana cerca del pico (meseta de
        # crucero: valor más alto en t=0.4) y se mantiene más baja al inicio
        # (arranque aún más suave: valor menor en t=0.1).
        assert tilt_envelope(0.4, steepness=12.0) > tilt_envelope(0.4, steepness=3.0)
        assert tilt_envelope(0.1, steepness=12.0) < tilt_envelope(0.1, steepness=3.0)

    def test_steepness_degenerado_no_revienta(self):
        # k=0 → denominador nulo: la función devuelve 0 defensivamente.
        assert tilt_envelope(0.5, steepness=0.0) == 0.0


class TestLateralTiltDeg:
    """lateral_tilt_deg: inclinación proporcional a la componente horizontal."""

    MD = 12.0   # max_deg de referencia
    K = 7.0     # steepness

    def _t(self, dx, dy, p=0.5):
        return lateral_tilt_deg(dx, dy, p, self.MD, self.K)

    def test_horizontal_puro_da_pico(self):
        # frac=1, envelope(0.5)=1, scale=0.5 → max_deg*0.5 = 6°.
        assert self._t(100, 0) == pytest.approx(6.0)

    def test_vertical_puro_no_inclina(self):
        assert self._t(0, 100) == 0.0

    def test_sin_movimiento_no_inclina(self):
        assert self._t(0, 0) == 0.0

    def test_signo_segun_direccion_horizontal(self):
        assert self._t(100, 0) > 0      # derecha → horario (+)
        assert self._t(-100, 0) < 0     # izquierda → antihorario (-)

    def test_45_grados_es_pico_por_raiz_de_dos(self):
        # frac = 1/√2 ≈ 0.707 → 6° * 0.707.
        assert self._t(100, 100) == pytest.approx(6.0 / (2 ** 0.5))

    def test_proporcional_a_fraccion_horizontal(self):
        # Más empinado (menos horizontal) → menos inclinación.
        assert abs(self._t(30, 90)) < abs(self._t(70, 70)) < abs(self._t(100, 10))

    def test_magnitud_independiente_de_la_distancia(self):
        # Solo importa la DIRECCIÓN (fracción), no cuánto se recorre:
        # (100,0) y (10,0) son ambos horizontales puros → mismo tilt.
        assert self._t(100, 0) == pytest.approx(self._t(10, 0))

    def test_envelope_modula_en_el_tiempo(self):
        # En los extremos del progress la inclinación es ~0 aunque el
        # movimiento sea horizontal (la envoltura tanh vale 0 en 0 y 1).
        assert self._t(100, 0, p=0.0) == 0.0
        assert self._t(100, 0, p=1.0) == 0.0
        assert self._t(100, 0, p=0.5) > self._t(100, 0, p=0.1)

    def test_scale_personalizable(self):
        # scale=1.0 → pico 12° en horizontal puro.
        assert lateral_tilt_deg(100, 0, 0.5, self.MD, self.K, scale=1.0) == pytest.approx(12.0)
