"""Tests de Location.

Cubre construcción con y sin posición en pantalla, inmutabilidad,
y especialmente la igualdad e indexación por id (que sobrescribimos
respecto al default de frozen=True).
"""

import pytest
from dataclasses import FrozenInstanceError

from droneplan_viz.domain.location import Location


class TestLocationConstruccion:
    def test_solo_con_id_posicion_es_none(self):
        loc = Location(id="deposito")
        assert loc.id == "deposito"
        assert loc.position_screen is None

    def test_con_posicion_en_pantalla(self):
        loc = Location(id="casa1", position_screen=(400, 300))
        assert loc.id == "casa1"
        assert loc.position_screen == (400, 300)


class TestLocationInmutabilidad:
    def test_no_se_puede_reasignar_id(self):
        loc = Location(id="deposito")
        with pytest.raises(FrozenInstanceError):
            loc.id = "otro"  # type: ignore[misc]

    def test_no_se_puede_reasignar_posicion(self):
        loc = Location(id="deposito", position_screen=(0, 0))
        with pytest.raises(FrozenInstanceError):
            loc.position_screen = (1, 1)  # type: ignore[misc]

    def test_no_se_pueden_anadir_atributos_nuevos(self):
        # Ver nota equivalente en test_content.py: la excepción exacta
        # depende de cómo interactúan frozen y slots en CPython.
        loc = Location(id="deposito")
        with pytest.raises((AttributeError, FrozenInstanceError, TypeError)):
            loc.vecinos = []  # type: ignore[attr-defined]


class TestLocationIgualdadPorId:
    """La identidad de una Location es su id, no su posición en pantalla.

    Si reposicionamos casa1 en el render, sigue siendo casa1 a efectos
    del dominio: los paquetes que están "en casa1" siguen estándolo.
    """

    def test_mismo_id_iguales_aunque_posicion_distinta(self):
        a = Location(id="casa1", position_screen=(0, 0))
        b = Location(id="casa1", position_screen=(500, 500))
        assert a == b

    def test_mismo_id_uno_con_posicion_otro_sin(self):
        a = Location(id="casa1")
        b = Location(id="casa1", position_screen=(10, 10))
        assert a == b

    def test_distinto_id_distintos_aun_con_misma_posicion(self):
        a = Location(id="casa1", position_screen=(0, 0))
        b = Location(id="casa2", position_screen=(0, 0))
        assert a != b

    def test_no_igual_a_otro_tipo(self):
        loc = Location(id="casa1")
        assert loc != "casa1"
        assert loc != 42


class TestLocationHashPorId:
    def test_hashable_en_set(self):
        s = {Location("a"), Location("b")}
        assert len(s) == 2

    def test_dos_locations_mismo_id_colapsan_en_set(self):
        # Aunque tengan posición distinta, comparten id y por tanto hash.
        s = {
            Location("casa1", position_screen=(0, 0)),
            Location("casa1", position_screen=(99, 99)),
        }
        assert len(s) == 1

    def test_sirve_como_clave_de_dict_indexada_por_id_logico(self):
        # Caso de uso real: costs[(origen, destino)] o índices por Location.
        d: dict[Location, str] = {Location("deposito"): "almacen central"}
        # Recuperamos con otra instancia que tenga el mismo id.
        assert d[Location("deposito")] == "almacen central"
