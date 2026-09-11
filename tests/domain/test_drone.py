"""Tests de Drone.

Cubrimos la composición con los demás componentes, los defaults (arms
vacíos, state IDLE), inmutabilidad estricta, y las transiciones
funcionales con replace. La unicidad de arm.id dentro del drone NO se
prueba aquí: es regla del Validator, no del tipo.
"""

import pytest
from dataclasses import FrozenInstanceError, replace

from droneplan_viz.domain.arm import Arm
from droneplan_viz.domain.drone import Drone
from droneplan_viz.domain.drone_state import DroneState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dron_minimo() -> Drone:
    """Drone básico: solo id y posición, sin brazos, IDLE."""
    return Drone(id="dron1", position="deposito")


@pytest.fixture
def dron_dos_brazos() -> Drone:
    """Drone con dos brazos, configuración típica de los .pddl reales."""
    return Drone(
        id="dron1",
        position="deposito",
        arms=(Arm("izq"), Arm("der")),
        state=DroneState.IDLE,
    )


# ---------------------------------------------------------------------------
# Construcción y defaults
# ---------------------------------------------------------------------------


class TestDroneConstruccionMinima:
    def test_solo_id_y_posicion(self, dron_minimo):
        assert dron_minimo.id == "dron1"
        assert dron_minimo.position == "deposito"

    def test_arms_por_defecto_tupla_vacia(self, dron_minimo):
        assert dron_minimo.arms == ()
        assert isinstance(dron_minimo.arms, tuple)

    def test_state_por_defecto_es_idle(self, dron_minimo):
        assert dron_minimo.state is DroneState.IDLE


class TestDroneConstruccionConBrazos:
    def test_con_dos_brazos(self, dron_dos_brazos):
        assert len(dron_dos_brazos.arms) == 2
        assert dron_dos_brazos.arms[0] == Arm("izq")
        assert dron_dos_brazos.arms[1] == Arm("der")

    def test_estado_explicito(self):
        d = Drone(
            id="dron1",
            position="casa1",
            state=DroneState.MOVING,
        )
        assert d.state is DroneState.MOVING


class TestDroneConfiguracionesVariadas:
    """Documentación ejecutable de la decisión 1 (composición sobre
    herencia): el mismo tipo Drone admite todas las variantes del
    dominio sin subclases."""

    def test_drone_explorador_sin_brazos(self):
        d = Drone(id="explorador", position="deposito")
        assert d.arms == ()
        # Estructuralmente válido. Las reglas que necesiten brazos las
        # aplica el Validator, no el tipo.

    def test_drone_un_brazo(self):
        d = Drone(id="dron1", position="deposito", arms=(Arm("central"),))
        assert len(d.arms) == 1

    def test_drone_tres_brazos(self):
        # No hay límite estructural en el número de brazos.
        d = Drone(
            id="pulpo",
            position="deposito",
            arms=(Arm("a"), Arm("b"), Arm("c")),
        )
        assert len(d.arms) == 3


# ---------------------------------------------------------------------------
# Inmutabilidad
# ---------------------------------------------------------------------------


class TestDroneInmutabilidad:
    def test_no_se_puede_reasignar_id(self, dron_minimo):
        with pytest.raises(FrozenInstanceError):
            dron_minimo.id = "otro"  # type: ignore[misc]

    def test_no_se_puede_reasignar_position(self, dron_minimo):
        with pytest.raises(FrozenInstanceError):
            dron_minimo.position = "casa1"  # type: ignore[misc]

    def test_no_se_puede_reasignar_arms(self, dron_minimo):
        with pytest.raises(FrozenInstanceError):
            dron_minimo.arms = (Arm("izq"),)  # type: ignore[misc]

    def test_no_se_puede_reasignar_state(self, dron_minimo):
        with pytest.raises(FrozenInstanceError):
            dron_minimo.state = DroneState.MOVING  # type: ignore[misc]

    def test_arms_es_tupla_no_lista(self, dron_dos_brazos):
        # Garantía estructural: ni siquiera existe .append. Aunque alguien
        # consiguiera saltarse frozen, no podría añadir un brazo en sitio.
        assert isinstance(dron_dos_brazos.arms, tuple)
        assert not hasattr(dron_dos_brazos.arms, "append")

    def test_no_se_pueden_anadir_atributos_nuevos(self, dron_minimo):
        with pytest.raises((AttributeError, FrozenInstanceError, TypeError)):
            dron_minimo.modelo = "DJI-X"  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Transiciones funcionales con replace
# ---------------------------------------------------------------------------


class TestDroneTransiciones:
    """Las transiciones (movimiento, cambio de estado FSM) se hacen
    creando una nueva instancia con replace. Los handlers de la sesión
    B aplicarán este patrón; aquí solo verificamos que funciona y no
    muta la original."""

    def test_replace_para_mover(self, dron_minimo):
        movido = replace(dron_minimo, position="casa1")
        assert movido.position == "casa1"
        assert dron_minimo.position == "deposito"  # original intacta

    def test_replace_para_cambiar_estado(self, dron_minimo):
        en_movimiento = replace(dron_minimo, state=DroneState.MOVING)
        assert en_movimiento.state is DroneState.MOVING
        assert dron_minimo.state is DroneState.IDLE

    def test_replace_a_error_no_modifica_la_original(self, dron_minimo):
        # Caso fundamental para la decisión "ERROR es sumidero local a
        # la rama forward". El drone original (en una rama anterior del
        # historial) sigue intacto. La nueva instancia es la que entra
        # en ERROR, y vivirá en el siguiente snapshot.
        roto = replace(dron_minimo, state=DroneState.ERROR)
        assert roto.state is DroneState.ERROR
        assert dron_minimo.state is DroneState.IDLE

    def test_ciclo_de_vida_completo(self):
        # IDLE → MOVING → IDLE (en otra posición) → INTERACTING → IDLE
        d = Drone(id="d1", position="deposito")
        d = replace(d, state=DroneState.MOVING)
        d = replace(d, position="casa1", state=DroneState.IDLE)
        d = replace(d, state=DroneState.INTERACTING)
        d = replace(d, state=DroneState.IDLE)
        assert d.position == "casa1"
        assert d.state is DroneState.IDLE


# ---------------------------------------------------------------------------
# Igualdad por valor
# ---------------------------------------------------------------------------


class TestDroneIgualdad:
    """No sobrescribimos __eq__: usamos el default de dataclass (igualdad
    por todos los campos). Razón: dos drones con mismo id pero distinto
    estado, posición o componentes están en momentos distintos de la
    simulación, no son equivalentes. Mismo criterio que Person."""

    def test_mismo_estado_completo_iguales(self):
        a = Drone(id="d1", position="deposito", arms=(Arm("izq"),))
        b = Drone(id="d1", position="deposito", arms=(Arm("izq"),))
        assert a == b

    def test_mismo_id_distinta_posicion_no_iguales(self):
        a = Drone(id="d1", position="deposito")
        b = Drone(id="d1", position="casa1")
        assert a != b

    def test_mismo_id_distinto_estado_no_iguales(self):
        a = Drone(id="d1", position="deposito", state=DroneState.IDLE)
        b = Drone(id="d1", position="deposito", state=DroneState.MOVING)
        assert a != b

    def test_mismo_id_distinta_configuracion_de_brazos(self):
        a = Drone(id="d1", position="deposito", arms=(Arm("izq"),))
        b = Drone(
            id="d1", position="deposito", arms=(Arm("izq"), Arm("der"))
        )
        assert a != b


# ---------------------------------------------------------------------------
# La validación NO ocurre en construcción
# ---------------------------------------------------------------------------


class TestDroneNoValidaEnConstruccion:
    """Coherente con el resto del dominio: el tipo es dato puro, las
    reglas semánticas viven en el Validator. Documentamos por test
    los casos que SÍ rechazará el Validator pero NO el constructor."""

    def test_brazos_con_id_duplicado_se_construye(self):
        # El Validator rechazará este drone porque tiene dos brazos "izq".
        # Pero estructuralmente el tipo lo admite: igualdad de Arm por
        # valor implica que arms = (Arm("izq"), Arm("izq")) son dos
        # objetos distintos como elementos de la tupla.
        d = Drone(
            id="d1", position="deposito", arms=(Arm("izq"), Arm("izq"))
        )
        assert len(d.arms) == 2

    def test_id_vacio_se_construye(self):
        # El builder del facade rechazará id vacíos, pero el tipo no.
        d = Drone(id="", position="deposito")
        assert d.id == ""

    def test_position_que_no_existe_en_world_se_construye(self):
        # El Validator rechazará: "no hay Location llamada 'inexistente'".
        # Pero Drone no conoce el World, así que aquí no se puede
        # comprobar y no se intenta.
        d = Drone(id="d1", position="inexistente")
        assert d.position == "inexistente"
