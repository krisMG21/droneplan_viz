"""Tests de Content.

Content es la clase más simple del dominio: dato inmutable con un único
campo. Los tests cubren construcción, inmutabilidad, igualdad por valor,
hashabilidad y la red de seguridad de slots (no se pueden añadir atributos
al vuelo).
"""

import pytest
from dataclasses import FrozenInstanceError

from droneplan_viz.domain.content import Content


class TestContentConstruccion:
    def test_se_construye_con_id(self):
        c = Content(id="medicina")
        assert c.id == "medicina"

    def test_id_posicional(self):
        # Aceptamos también construcción posicional, es ergonómica.
        c = Content("comida")
        assert c.id == "comida"


class TestContentInmutabilidad:
    def test_no_se_puede_reasignar_id(self):
        c = Content(id="medicina")
        with pytest.raises(FrozenInstanceError):
            c.id = "veneno"  # type: ignore[misc]

    def test_no_se_pueden_anadir_atributos_nuevos(self):
        # slots=True impide crear atributos no declarados. Esto cazaría
        # typos como c.descripcion = "..." cuando el atributo correcto
        # tuviera otro nombre.
        #
        # La excepción exacta depende del orden en que actúan frozen y slots
        # en CPython: puede ser FrozenInstanceError, AttributeError o
        # TypeError. Lo que nos importa es que la asignación falla, no la
        # clase concreta de la excepción.
        c = Content(id="medicina")
        with pytest.raises((AttributeError, FrozenInstanceError, TypeError)):
            c.descripcion = "Medicamentos varios"  # type: ignore[attr-defined]


class TestContentIgualdad:
    def test_dos_content_con_mismo_id_son_iguales(self):
        assert Content("medicina") == Content("medicina")

    def test_distinto_id_distintos(self):
        assert Content("medicina") != Content("comida")

    def test_no_igual_a_otro_tipo(self):
        assert Content("medicina") != "medicina"


class TestContentHash:
    def test_hashable(self):
        # Tiene que poder vivir en sets y como clave de dict.
        s = {Content("medicina"), Content("comida")}
        assert len(s) == 2

    def test_dos_iguales_colapsan_en_set(self):
        s = {Content("medicina"), Content("medicina")}
        assert len(s) == 1

    def test_sirve_como_clave_de_dict(self):
        d = {Content("medicina"): 3, Content("comida"): 5}
        assert d[Content("medicina")] == 3
