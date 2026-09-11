"""Tests de droneplan_viz.facade.errors.

Verifican el CONTRATO de la taxonomía de errores de construcción:

  1. Jerarquía: todas las clases concretas heredan de FacadeError (un solo
     `except` las atrapa todas) y de Exception.
  2. Datos estructurados: cada error guarda como atributos los datos que
     recibe, para que el código consumidor (y estos tests) no dependa de
     hacer match de substrings frágiles.
  3. Mensajes didácticos: cada mensaje cita la llamada del builder que
     arregla el problema y, donde aplica, usa el énfasis "ANTES" y nombra
     la regla violada.

Son lógica pura: no tocan dominio, runtime ni pygame.
"""
from __future__ import annotations

import pytest

from droneplan_viz.facade import errors as err_mod
from droneplan_viz.facade.errors import (
    DuplicateIdError,
    FacadeError,
    MixedTimingError,
    UnknownArmError,
    UnknownContentError,
    UnknownDroneError,
    UnknownLocationError,
    UnknownPackageError,
    UnknownPersonError,
    UnknownTransporterError,
    WrongReferenceKindError,
)


# Todas las clases concretas (excluida la base) para los tests paramétricos
# de jerarquía. La base se prueba aparte.
_CONCRETE_ERRORS = [
    UnknownLocationError,
    UnknownContentError,
    UnknownPersonError,
    UnknownPackageError,
    UnknownDroneError,
    UnknownTransporterError,
    UnknownArmError,
    WrongReferenceKindError,
    DuplicateIdError,
    MixedTimingError,
]


# ---------------------------------------------------------------------------
# Jerarquía
# ---------------------------------------------------------------------------


class TestJerarquia:
    def test_facade_error_es_exception(self):
        assert issubclass(FacadeError, Exception)

    @pytest.mark.parametrize("cls", _CONCRETE_ERRORS)
    def test_concreto_hereda_de_facade_error(self, cls):
        assert issubclass(cls, FacadeError)

    def test_un_solo_except_facade_error_atrapa_un_concreto(self):
        with pytest.raises(FacadeError):
            raise UnknownLocationError("casa9")

    def test_all_exporta_todas_las_clases(self):
        exportadas = set(err_mod.__all__)
        esperadas = {"FacadeError"} | {c.__name__ for c in _CONCRETE_ERRORS}
        assert exportadas == esperadas


# ---------------------------------------------------------------------------
# Familia "no declarado": atributos + mensaje que cita la llamada que arregla
# ---------------------------------------------------------------------------


class TestUnknownLocationError:
    def test_guarda_loc_id(self):
        e = UnknownLocationError("casa9")
        assert e.loc_id == "casa9"
        assert e.contexto is None

    def test_mensaje_cita_la_llamada_que_lo_arregla(self):
        e = UnknownLocationError("casa9")
        msg = str(e)
        assert "casa9" in msg
        assert "viz.world.location('casa9')" in msg

    def test_mensaje_enfatiza_antes(self):
        """El mensaje dice CUÁNDO declararla, no solo que falta."""
        assert "ANTES" in str(UnknownLocationError("casa9"))

    def test_contexto_se_inserta_en_el_mensaje(self):
        e = UnknownLocationError("casa9", contexto="al declarar el dron 'd1'")
        assert e.contexto == "al declarar el dron 'd1'"
        assert "al declarar el dron 'd1'" in str(e)

    def test_sin_contexto_usa_generico_aqui(self):
        assert "aquí" in str(UnknownLocationError("casa9"))


class TestUnknownContentError:
    def test_guarda_content_id_y_mensaje(self):
        e = UnknownContentError("medicina")
        assert e.content_id == "medicina"
        assert "viz.world.content('medicina')" in str(e)
        assert "ANTES" in str(e)


class TestUnknownPersonError:
    def test_guarda_person_id_y_mensaje(self):
        e = UnknownPersonError("p1")
        assert e.person_id == "p1"
        assert "viz.world.person('p1'" in str(e)
        assert "ANTES" in str(e)


class TestUnknownPackageError:
    def test_guarda_package_id_y_mensaje(self):
        e = UnknownPackageError("caja1")
        assert e.package_id == "caja1"
        assert "viz.world.package('caja1'" in str(e)
        assert "ANTES" in str(e)


class TestUnknownDroneError:
    def test_guarda_drone_id_y_mensaje(self):
        e = UnknownDroneError("dron1")
        assert e.drone_id == "dron1"
        assert "viz.agents.drone('dron1'" in str(e)
        assert "ANTES" in str(e)


class TestUnknownTransporterError:
    def test_guarda_transporter_id_y_mensaje(self):
        e = UnknownTransporterError("t1")
        assert e.transporter_id == "t1"
        assert "viz.agents.transporter('t1'" in str(e)
        assert "ANTES" in str(e)


# ---------------------------------------------------------------------------
# UnknownArmError: existencia, no ocupación; dos casos de mensaje
# ---------------------------------------------------------------------------


class TestUnknownArmError:
    def test_guarda_atributos(self):
        e = UnknownArmError("central", "d1", brazos_disponibles=("izq", "der"))
        assert e.arm_id == "central"
        assert e.drone_id == "d1"
        assert e.brazos_disponibles == ("izq", "der")

    def test_con_brazos_los_lista(self):
        e = UnknownArmError("central", "d1", brazos_disponibles=("izq", "der"))
        msg = str(e)
        assert "d1" in msg
        assert "central" in msg
        assert "'izq'" in msg and "'der'" in msg

    def test_dron_explorador_sin_brazos_mensaje_especifico(self):
        """arms=[] -> menciona 'explorador' y sugiere declararlo con brazos."""
        e = UnknownArmError("izq", "explorador", brazos_disponibles=())
        msg = str(e)
        assert "explorador" in msg
        assert "arms=" in msg

    def test_default_brazos_disponibles_vacio(self):
        e = UnknownArmError("izq", "explorador")
        assert e.brazos_disponibles == ()

    def test_normaliza_brazos_a_tupla(self):
        """Acepta cualquier iterable y lo guarda como tupla inmutable."""
        e = UnknownArmError("x", "d1", brazos_disponibles=["izq", "der"])
        assert e.brazos_disponibles == ("izq", "der")
        assert isinstance(e.brazos_disponibles, tuple)


# ---------------------------------------------------------------------------
# WrongReferenceKindError: guarda del solape de `a`
# ---------------------------------------------------------------------------


class TestWrongReferenceKindError:
    def test_guarda_atributos(self):
        e = WrongReferenceKindError(
            "p1",
            metodo="mover",
            parametro="a",
            esperado="una localización",
            recibido="la persona",
        )
        assert e.ref_id == "p1"
        assert e.metodo == "mover"
        assert e.parametro == "a"
        assert e.esperado == "una localización"
        assert e.recibido == "la persona"

    def test_mensaje_mover_recibe_persona(self):
        e = WrongReferenceKindError(
            "p1",
            metodo="mover",
            parametro="a",
            esperado="una localización",
            recibido="la persona",
        )
        # Reproduce el ejemplo acordado en la revisión de diseño.
        assert str(e) == (
            "'mover' espera una localización en 'a'; recibió la persona 'p1'"
        )

    def test_mensaje_entregar_recibe_localizacion(self):
        e = WrongReferenceKindError(
            "casa1",
            metodo="entregar",
            parametro="a",
            esperado="una persona",
            recibido="la localización",
        )
        assert str(e) == (
            "'entregar' espera una persona en 'a'; recibió la localización "
            "'casa1'"
        )


# ---------------------------------------------------------------------------
# DuplicateIdError
# ---------------------------------------------------------------------------


class TestDuplicateIdError:
    def test_guarda_atributos(self):
        e = DuplicateIdError("casa1", categoria="una localización")
        assert e.id_repetido == "casa1"
        assert e.categoria == "una localización"

    def test_mensaje_entidad(self):
        e = DuplicateIdError("d1", categoria="un dron")
        msg = str(e)
        assert "d1" in msg
        assert "un dron" in msg
        assert "únicos" in msg

    def test_mensaje_accion_con_id_explicito(self):
        """El mismo error sirve para id= de acción duplicado."""
        e = DuplicateIdError("coger_caja_brazo1_0", categoria="una acción con id")
        msg = str(e)
        assert "coger_caja_brazo1_0" in msg
        assert "una acción con id" in msg


# ---------------------------------------------------------------------------
# MixedTimingError
# ---------------------------------------------------------------------------


class TestMixedTimingError:
    def test_guarda_atributos(self):
        e = MixedTimingError("recoger_3", falta_inicio=True)
        assert e.accion_id == "recoger_3"
        assert e.falta_inicio is True

    def test_mensaje_falta_inicio(self):
        """Caso típico: media salida FF (sin inicio) tras media OPTIC."""
        msg = str(MixedTimingError("recoger_3", falta_inicio=True))
        assert "mixto" in msg
        assert "recoger_3" in msg
        assert "no lleva 'inicio='" in msg
        assert "todas" in msg and "ninguna" in msg

    def test_mensaje_sobra_inicio(self):
        """Caso simétrico: una acción con inicio tras varias sin él."""
        msg = str(MixedTimingError("mover_2", falta_inicio=False))
        assert "mixto" in msg
        assert "mover_2" in msg
        assert "lleva 'inicio='" in msg


# ---------------------------------------------------------------------------
# Comprobación transversal: todos son raisable y atrapables como FacadeError
# ---------------------------------------------------------------------------


class TestRaisable:
    def test_todos_los_concretos_se_pueden_lanzar_y_capturar(self):
        instancias = [
            UnknownLocationError("x"),
            UnknownContentError("x"),
            UnknownPersonError("x"),
            UnknownPackageError("x"),
            UnknownDroneError("x"),
            UnknownTransporterError("x"),
            UnknownArmError("x", "d1"),
            WrongReferenceKindError(
                "x", metodo="m", parametro="a", esperado="e", recibido="r"
            ),
            DuplicateIdError("x", categoria="algo"),
            MixedTimingError("x", falta_inicio=True),
        ]
        for inst in instancias:
            with pytest.raises(FacadeError):
                raise inst
            # Y el mensaje nunca está vacío.
            assert str(inst)
