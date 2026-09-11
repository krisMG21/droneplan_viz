"""Tests del Validator.

Cubrimos en este turno:

    1. ValidationResult: construcción, factory methods, uso como bool.
    2. _check_drone_ready: las tres reglas transversales en aislamiento.
       Las testeamos aquí una vez para no duplicar en cada validate_X.
    3. validate_move: las reglas específicas de la acción "volar".

Los mundos de prueba se construyen explícitamente en fixtures, pequeños
pero suficientes para cubrir cada caso. No reutilizamos fixtures de
test_world.py para que cada test_validator.py sea autocontenido.
"""

import pytest
from dataclasses import replace

from droneplan_viz.domain.arm import Arm
from droneplan_viz.domain.content import Content
from droneplan_viz.domain.drone import Drone
from droneplan_viz.domain.drone_state import DroneState
from droneplan_viz.domain.location import Location
from droneplan_viz.domain.package import AtLocation, HeldByArm, InTransporter, Package
from droneplan_viz.domain.person import Person
from droneplan_viz.domain.transporter import Transporter
from droneplan_viz.domain.validator import (
    ValidationResult,
    _check_drone_ready,
    validate_deliver,
    validate_load_into_transporter,
    validate_move,
    validate_move_with_transporter,
    validate_pick_up,
    validate_unload_from_transporter,
)
from droneplan_viz.domain.world import World


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mundo_dos_locs() -> World:
    """Mundo mínimo con dos localizaciones conectadas en ambos sentidos
    y un drone IDLE en deposito."""
    locations = {
        "deposito": Location("deposito"),
        "casa1": Location("casa1"),
    }
    drones = {"dron1": Drone(id="dron1", position="deposito")}
    costs = {
        ("deposito", "casa1"): 66.0,
        ("casa1", "deposito"): 66.0,
    }
    return World(locations=locations, drones=drones, costs=costs)


# ---------------------------------------------------------------------------
# ValidationResult
# ---------------------------------------------------------------------------


class TestValidationResult:
    def test_factory_valid(self):
        r = ValidationResult.valid()
        assert r.ok is True
        assert r.reason is None

    def test_factory_invalid_requiere_razon(self):
        r = ValidationResult.invalid("algo falló")
        assert r.ok is False
        assert r.reason == "algo falló"

    def test_construccion_directa(self):
        # No prohibimos la construcción directa; los factories son
        # ergonomía, no obligación.
        r = ValidationResult(ok=True)
        assert r.ok is True
        assert r.reason is None

    def test_es_inmutable(self):
        r = ValidationResult.valid()
        with pytest.raises(Exception):
            r.ok = False  # type: ignore[misc]

    def test_bool_true_si_ok(self):
        r = ValidationResult.valid()
        assert bool(r) is True
        # En contexto de if:
        if r:
            pasa = True
        else:
            pasa = False
        assert pasa is True

    def test_bool_false_si_invalid(self):
        r = ValidationResult.invalid("fallo")
        assert bool(r) is False
        if not r:
            falla = True
        else:
            falla = False
        assert falla is True

    def test_igualdad_por_valor(self):
        assert ValidationResult.valid() == ValidationResult.valid()
        assert ValidationResult.invalid("x") == ValidationResult.invalid("x")
        assert ValidationResult.valid() != ValidationResult.invalid("x")


# ---------------------------------------------------------------------------
# _check_drone_ready: tres reglas transversales
# ---------------------------------------------------------------------------


class TestCheckDroneReady:
    """Helper privado del módulo. Testeado aislado porque aparece detrás
    de toda validate_X; probarlo aquí evita duplicar los mismos tests en
    cada acción."""

    def test_drone_idle_pasa(self, mundo_dos_locs):
        assert _check_drone_ready(mundo_dos_locs, "dron1") is None

    def test_drone_inexistente_devuelve_razon(self, mundo_dos_locs):
        razon = _check_drone_ready(mundo_dos_locs, "fantasma")
        assert razon is not None
        assert "fantasma" in razon

    def test_drone_en_error_devuelve_razon(self, mundo_dos_locs):
        # Reemplazamos el drone por uno en estado ERROR.
        roto = replace(
            mundo_dos_locs.drones["dron1"], state=DroneState.ERROR
        )
        mundo = replace(
            mundo_dos_locs, drones={**mundo_dos_locs.drones, "dron1": roto}
        )
        razon = _check_drone_ready(mundo, "dron1")
        assert razon is not None
        assert "ERROR" in razon

    def test_drone_moving_devuelve_razon(self, mundo_dos_locs):
        ocupado = replace(
            mundo_dos_locs.drones["dron1"], state=DroneState.MOVING
        )
        mundo = replace(
            mundo_dos_locs,
            drones={**mundo_dos_locs.drones, "dron1": ocupado},
        )
        razon = _check_drone_ready(mundo, "dron1")
        assert razon is not None
        assert "ocupado" in razon

    def test_drone_interacting_devuelve_razon(self, mundo_dos_locs):
        ocupado = replace(
            mundo_dos_locs.drones["dron1"], state=DroneState.INTERACTING
        )
        mundo = replace(
            mundo_dos_locs,
            drones={**mundo_dos_locs.drones, "dron1": ocupado},
        )
        razon = _check_drone_ready(mundo, "dron1")
        assert razon is not None
        assert "ocupado" in razon

    def test_orden_de_los_chequeos_existencia_primero(self, mundo_dos_locs):
        # Si el drone no existe, ni siquiera podemos consultar su estado.
        # El helper debe rechazar por inexistencia y no fallar con
        # KeyError ni nada parecido.
        razon = _check_drone_ready(mundo_dos_locs, "fantasma")
        assert razon is not None
        assert "no existe" in razon


# ---------------------------------------------------------------------------
# validate_move: reglas específicas
# ---------------------------------------------------------------------------


class TestValidateMoveCasoFeliz:
    def test_drone_idle_a_localizacion_adyacente(self, mundo_dos_locs):
        r = validate_move(mundo_dos_locs, "dron1", "casa1")
        assert r.ok is True
        assert r.reason is None

    def test_resultado_es_consultable_como_bool(self, mundo_dos_locs):
        # Patrón de uso: `if validate_move(...)`.
        if validate_move(mundo_dos_locs, "dron1", "casa1"):
            entra = True
        else:
            entra = False
        assert entra is True


class TestValidateMoveReglasTransversales:
    """Comprueba que las reglas transversales se propagan a validate_move.
    No reproducimos toda la casuística (eso ya está en TestCheckDroneReady):
    una muestra basta para verificar que el helper se aplica."""

    def test_rechaza_drone_inexistente(self, mundo_dos_locs):
        r = validate_move(mundo_dos_locs, "fantasma", "casa1")
        assert r.ok is False
        assert r.reason is not None
        assert "fantasma" in r.reason

    def test_rechaza_drone_en_error(self, mundo_dos_locs):
        roto = replace(
            mundo_dos_locs.drones["dron1"], state=DroneState.ERROR
        )
        mundo = replace(
            mundo_dos_locs, drones={**mundo_dos_locs.drones, "dron1": roto}
        )
        r = validate_move(mundo, "dron1", "casa1")
        assert r.ok is False
        assert "ERROR" in r.reason  # type: ignore[operator]

    def test_rechaza_drone_ocupado(self, mundo_dos_locs):
        ocupado = replace(
            mundo_dos_locs.drones["dron1"], state=DroneState.MOVING
        )
        mundo = replace(
            mundo_dos_locs,
            drones={**mundo_dos_locs.drones, "dron1": ocupado},
        )
        r = validate_move(mundo, "dron1", "casa1")
        assert r.ok is False


class TestValidateMoveLocalizacionDestino:
    def test_rechaza_destino_inexistente(self, mundo_dos_locs):
        r = validate_move(mundo_dos_locs, "dron1", "inexistente")
        assert r.ok is False
        assert r.reason is not None
        assert "inexistente" in r.reason


class TestValidateMoveAdyacencia:
    def test_rechaza_si_no_hay_coste_declarado(self):
        # Mundo con tres localizaciones, pero solo declaramos coste
        # desde deposito hasta casa1. casa2 está aislada para el dron.
        locations = {
            "deposito": Location("deposito"),
            "casa1": Location("casa1"),
            "casa2": Location("casa2"),
        }
        drones = {"dron1": Drone(id="dron1", position="deposito")}
        costs = {
            ("deposito", "casa1"): 66.0,
            ("casa1", "deposito"): 66.0,
        }
        w = World(locations=locations, drones=drones, costs=costs)

        # Existe la localización casa2, pero no es alcanzable desde
        # deposito porque no hay arista declarada.
        r = validate_move(w, "dron1", "casa2")
        assert r.ok is False
        assert r.reason is not None
        assert "deposito" in r.reason
        assert "casa2" in r.reason

    def test_adyacencia_unidireccional_se_respeta(self):
        # Si solo se declaró deposito → casa1, no a la inversa, no se
        # puede volver. Esto modela el caso de grafos dirigidos del
        # dominio.
        locations = {
            "deposito": Location("deposito"),
            "casa1": Location("casa1"),
        }
        drones = {"dron1": Drone(id="dron1", position="casa1")}
        costs = {("deposito", "casa1"): 66.0}  # solo ida
        w = World(locations=locations, drones=drones, costs=costs)
        r = validate_move(w, "dron1", "deposito")
        assert r.ok is False


class TestValidateMoveOrdenDeChequeos:
    """Los chequeos van en orden: transversales, destino existe,
    adyacencia. Verificamos que ante varios fallos simultáneos se
    reporta el más fundamental. Esto importa para la UX: el alumno ve
    el primer problema relevante, no uno arbitrario."""

    def test_drone_inexistente_se_reporta_antes_que_destino_inexistente(
        self, mundo_dos_locs
    ):
        # Si el drone no existe, ni siquiera entramos a comprobar el
        # destino. Da igual que el destino tampoco exista.
        r = validate_move(mundo_dos_locs, "fantasma", "tampoco_existe")
        assert r.ok is False
        assert "fantasma" in r.reason  # type: ignore[operator]

    def test_destino_inexistente_se_reporta_antes_que_arista(
        self, mundo_dos_locs
    ):
        # Si el destino no existe, no tiene sentido reportar "no hay
        # arista a un nodo que no existe".
        r = validate_move(mundo_dos_locs, "dron1", "fantasma")
        assert r.ok is False
        assert "fantasma" in r.reason  # type: ignore[operator]
        # No menciona "arista" ni "transitable".
        assert "arista" not in r.reason  # type: ignore[operator]


# ---------------------------------------------------------------------------
# validate_move_with_transporter
# ---------------------------------------------------------------------------


@pytest.fixture
def mundo_con_transporter() -> World:
    """Mundo con dos localizaciones, un drone con dos brazos vacíos, y
    un transportador co-localizado con el drone en deposito."""
    locations = {
        "deposito": Location("deposito"),
        "casa1": Location("casa1"),
    }
    drones = {
        "dron1": Drone(
            id="dron1",
            position="deposito",
            arms=(Arm("izq"), Arm("der")),
        )
    }
    transporters = {
        "t1": Transporter(id="t1", position="deposito", capacity=4),
    }
    costs = {
        ("deposito", "casa1"): 66.0,
        ("casa1", "deposito"): 66.0,
    }
    return World(
        locations=locations,
        drones=drones,
        transporters=transporters,
        costs=costs,
    )


class TestValidateMoveWithTransporterCasoFeliz:
    def test_drone_idle_transporter_colocalizado_brazos_libres(
        self, mundo_con_transporter
    ):
        r = validate_move_with_transporter(
            mundo_con_transporter, "dron1", "casa1", "t1"
        )
        assert r.ok is True
        assert r.reason is None

    def test_resultado_es_consultable_como_bool(self, mundo_con_transporter):
        if validate_move_with_transporter(
            mundo_con_transporter, "dron1", "casa1", "t1"
        ):
            entra = True
        else:
            entra = False
        assert entra is True


class TestValidateMoveWithTransporterReglasReusadas:
    """Las reglas transversales y las de destino son las mismas que
    validate_move, vía _check_drone_ready y _check_move_target. Una
    muestra basta para verificar que se propagan."""

    def test_rechaza_drone_inexistente(self, mundo_con_transporter):
        r = validate_move_with_transporter(
            mundo_con_transporter, "fantasma", "casa1", "t1"
        )
        assert r.ok is False
        assert "fantasma" in r.reason  # type: ignore[operator]

    def test_rechaza_destino_inexistente(self, mundo_con_transporter):
        r = validate_move_with_transporter(
            mundo_con_transporter, "dron1", "inexistente", "t1"
        )
        assert r.ok is False
        assert "inexistente" in r.reason  # type: ignore[operator]

    def test_rechaza_arista_sin_coste(self):
        # Tres localizaciones, casa2 sin coste declarado desde deposito.
        locations = {
            "deposito": Location("deposito"),
            "casa1": Location("casa1"),
            "casa2": Location("casa2"),
        }
        drones = {
            "dron1": Drone(
                id="dron1", position="deposito", arms=(Arm("izq"),)
            )
        }
        transporters = {
            "t1": Transporter(id="t1", position="deposito", capacity=4)
        }
        costs = {("deposito", "casa1"): 66.0}
        w = World(
            locations=locations,
            drones=drones,
            transporters=transporters,
            costs=costs,
        )
        r = validate_move_with_transporter(w, "dron1", "casa2", "t1")
        assert r.ok is False


class TestValidateMoveWithTransporterTransporter:
    def test_rechaza_transporter_inexistente(self, mundo_con_transporter):
        r = validate_move_with_transporter(
            mundo_con_transporter, "dron1", "casa1", "t_fantasma"
        )
        assert r.ok is False
        assert r.reason is not None
        assert "t_fantasma" in r.reason

    def test_rechaza_transporter_en_otra_localizacion(
        self, mundo_con_transporter
    ):
        # Movemos el transporter a casa1, distinto de la posición del drone.
        t_lejos = replace(
            mundo_con_transporter.transporters["t1"], position="casa1"
        )
        w = replace(
            mundo_con_transporter,
            transporters={
                **mundo_con_transporter.transporters,
                "t1": t_lejos,
            },
        )
        r = validate_move_with_transporter(w, "dron1", "casa1", "t1")
        assert r.ok is False
        assert r.reason is not None
        assert "co-localizados" in r.reason


class TestValidateMoveWithTransporterBrazos:
    def test_rechaza_si_todos_los_brazos_estan_ocupados(
        self, mundo_con_transporter
    ):
        # Llenamos los dos brazos con paquetes. Hacemos esto añadiendo
        # paquetes al mundo, ya que la relación "qué sostiene cada brazo"
        # vive en Package.at.
        medicina = Content("medicina")
        cajas = {
            "caja1": Package(
                id="caja1",
                contains=medicina,
                at=HeldByArm("dron1", "izq"),
            ),
            "caja2": Package(
                id="caja2",
                contains=medicina,
                at=HeldByArm("dron1", "der"),
            ),
        }
        w = replace(mundo_con_transporter, packages=cajas)
        r = validate_move_with_transporter(w, "dron1", "casa1", "t1")
        assert r.ok is False
        assert r.reason is not None
        assert "brazo libre" in r.reason

    def test_acepta_si_al_menos_un_brazo_libre(self, mundo_con_transporter):
        # Solo uno de los dos brazos ocupado. Refleja el caso PDDL donde
        # el drone arrastra el transporter con un brazo mientras sostiene
        # una caja con el otro.
        medicina = Content("medicina")
        cajas = {
            "caja1": Package(
                id="caja1",
                contains=medicina,
                at=HeldByArm("dron1", "izq"),
            )
        }
        w = replace(mundo_con_transporter, packages=cajas)
        r = validate_move_with_transporter(w, "dron1", "casa1", "t1")
        assert r.ok is True

    def test_drone_sin_brazos_no_puede_arrastrar(self):
        # Caso límite: drone explorador con arms=(). Estructuralmente
        # no tiene ningún brazo libre porque no tiene ningún brazo.
        # El mensaje sigue siendo coherente.
        locations = {
            "deposito": Location("deposito"),
            "casa1": Location("casa1"),
        }
        drones = {
            "explorador": Drone(id="explorador", position="deposito")
        }
        transporters = {
            "t1": Transporter(id="t1", position="deposito", capacity=4)
        }
        costs = {("deposito", "casa1"): 66.0}
        w = World(
            locations=locations,
            drones=drones,
            transporters=transporters,
            costs=costs,
        )
        r = validate_move_with_transporter(w, "explorador", "casa1", "t1")
        assert r.ok is False
        assert "brazo libre" in r.reason  # type: ignore[operator]


class TestValidateMoveWithTransporterOrdenDeChequeos:
    """Verificamos el orden jerárquico: transversales primero, después
    destino, después transporter (existencia y posición), por último
    brazos. Cuando hay varios fallos a la vez, se reporta el más básico."""

    def test_drone_inexistente_se_reporta_antes_que_transporter_inexistente(
        self, mundo_con_transporter
    ):
        r = validate_move_with_transporter(
            mundo_con_transporter, "fantasma", "casa1", "t_fantasma"
        )
        assert r.ok is False
        # Reporta el drone, no el transporter.
        assert "fantasma" in r.reason  # type: ignore[operator]
        assert "t_fantasma" not in r.reason  # type: ignore[operator]

    def test_destino_inexistente_se_reporta_antes_que_transporter(
        self, mundo_con_transporter
    ):
        r = validate_move_with_transporter(
            mundo_con_transporter,
            "dron1",
            "inexistente",
            "t_fantasma",
        )
        assert r.ok is False
        assert "inexistente" in r.reason  # type: ignore[operator]
        assert "t_fantasma" not in r.reason  # type: ignore[operator]

    def test_transporter_inexistente_se_reporta_antes_que_brazos(
        self, mundo_con_transporter
    ):
        # Llenamos los dos brazos para que el chequeo de brazos también
        # fallaría. Pero el transporter no existe: ese fallo va primero.
        medicina = Content("medicina")
        cajas = {
            "c1": Package(
                id="c1", contains=medicina, at=HeldByArm("dron1", "izq")
            ),
            "c2": Package(
                id="c2", contains=medicina, at=HeldByArm("dron1", "der")
            ),
        }
        w = replace(mundo_con_transporter, packages=cajas)
        r = validate_move_with_transporter(
            w, "dron1", "casa1", "t_fantasma"
        )
        assert r.ok is False
        assert "t_fantasma" in r.reason  # type: ignore[operator]
        assert "brazo libre" not in r.reason  # type: ignore[operator]


# ---------------------------------------------------------------------------
# validate_pick_up
# ---------------------------------------------------------------------------


@pytest.fixture
def mundo_para_recoger() -> World:
    """Mundo con un drone de dos brazos en deposito, y una caja libre
    en deposito esperando a ser recogida."""
    medicina = Content("medicina")
    locations = {
        "deposito": Location("deposito"),
        "casa1": Location("casa1"),
    }
    drones = {
        "dron1": Drone(
            id="dron1",
            position="deposito",
            arms=(Arm("izq"), Arm("der")),
        )
    }
    packages = {
        "caja1": Package(
            id="caja1", contains=medicina, at=AtLocation("deposito")
        )
    }
    costs = {
        ("deposito", "casa1"): 66.0,
        ("casa1", "deposito"): 66.0,
    }
    return World(
        locations=locations,
        drones=drones,
        packages=packages,
        costs=costs,
    )


class TestValidatePickUpCasoFeliz:
    def test_drone_idle_brazo_libre_paquete_colocalizado(
        self, mundo_para_recoger
    ):
        r = validate_pick_up(mundo_para_recoger, "dron1", "caja1", "izq")
        assert r.ok is True
        assert r.reason is None

    def test_se_puede_recoger_con_cualquier_brazo_libre(
        self, mundo_para_recoger
    ):
        r1 = validate_pick_up(mundo_para_recoger, "dron1", "caja1", "izq")
        r2 = validate_pick_up(mundo_para_recoger, "dron1", "caja1", "der")
        assert r1.ok is True
        assert r2.ok is True


class TestValidatePickUpReglasTransversales:
    def test_rechaza_drone_inexistente(self, mundo_para_recoger):
        r = validate_pick_up(
            mundo_para_recoger, "fantasma", "caja1", "izq"
        )
        assert r.ok is False
        assert "fantasma" in r.reason  # type: ignore[operator]

    def test_rechaza_drone_en_error(self, mundo_para_recoger):
        roto = replace(
            mundo_para_recoger.drones["dron1"], state=DroneState.ERROR
        )
        w = replace(
            mundo_para_recoger,
            drones={**mundo_para_recoger.drones, "dron1": roto},
        )
        r = validate_pick_up(w, "dron1", "caja1", "izq")
        assert r.ok is False
        assert "ERROR" in r.reason  # type: ignore[operator]


class TestValidatePickUpExistencia:
    def test_rechaza_paquete_inexistente(self, mundo_para_recoger):
        r = validate_pick_up(
            mundo_para_recoger, "dron1", "caja_fantasma", "izq"
        )
        assert r.ok is False
        assert "caja_fantasma" in r.reason  # type: ignore[operator]

    def test_rechaza_brazo_inexistente(self, mundo_para_recoger):
        r = validate_pick_up(
            mundo_para_recoger, "dron1", "caja1", "central"
        )
        assert r.ok is False
        assert "central" in r.reason  # type: ignore[operator]


class TestValidatePickUpBrazoOcupado:
    def test_rechaza_si_el_brazo_ya_sostiene_otro_paquete(
        self, mundo_para_recoger
    ):
        # Añadimos otra caja ya sostenida por el brazo izq.
        medicina = Content("medicina")
        cajas_actualizadas = {
            **mundo_para_recoger.packages,
            "caja2": Package(
                id="caja2",
                contains=medicina,
                at=HeldByArm("dron1", "izq"),
            ),
        }
        w = replace(mundo_para_recoger, packages=cajas_actualizadas)
        # Intentamos recoger caja1 con el brazo izq, que ya está ocupado.
        r = validate_pick_up(w, "dron1", "caja1", "izq")
        assert r.ok is False
        assert r.reason is not None
        assert "caja2" in r.reason  # menciona qué hay ya en el brazo

    def test_acepta_si_el_otro_brazo_esta_libre(self, mundo_para_recoger):
        # Mismo setup que el test anterior, pero usamos el brazo der.
        medicina = Content("medicina")
        cajas_actualizadas = {
            **mundo_para_recoger.packages,
            "caja2": Package(
                id="caja2",
                contains=medicina,
                at=HeldByArm("dron1", "izq"),
            ),
        }
        w = replace(mundo_para_recoger, packages=cajas_actualizadas)
        r = validate_pick_up(w, "dron1", "caja1", "der")
        assert r.ok is True


class TestValidatePickUpEstadoDelPaquete:
    def test_rechaza_si_el_paquete_esta_sostenido_por_otro_brazo(
        self, mundo_para_recoger
    ):
        # La caja ya está sostenida, no libre en una localización.
        caja_en_mano = Package(
            id="caja1",
            contains=Content("medicina"),
            at=HeldByArm("dron1", "izq"),
        )
        w = replace(
            mundo_para_recoger, packages={"caja1": caja_en_mano}
        )
        r = validate_pick_up(w, "dron1", "caja1", "der")
        assert r.ok is False
        assert r.reason is not None
        assert "no está libre" in r.reason

    def test_rechaza_si_el_paquete_esta_en_transporter(
        self, mundo_para_recoger
    ):
        # Añadimos un transporter al mundo y metemos la caja dentro.
        t1 = Transporter(id="t1", position="deposito", capacity=4)
        caja_dentro = Package(
            id="caja1",
            contains=Content("medicina"),
            at=InTransporter("t1"),
        )
        w = replace(
            mundo_para_recoger,
            transporters={"t1": t1},
            packages={"caja1": caja_dentro},
        )
        r = validate_pick_up(w, "dron1", "caja1", "izq")
        assert r.ok is False
        assert "no está libre" in r.reason  # type: ignore[operator]

    def test_rechaza_paquete_en_otra_localizacion(self, mundo_para_recoger):
        # La caja está libre, pero en casa1, no en deposito.
        caja_lejos = Package(
            id="caja1",
            contains=Content("medicina"),
            at=AtLocation("casa1"),
        )
        w = replace(mundo_para_recoger, packages={"caja1": caja_lejos})
        r = validate_pick_up(w, "dron1", "caja1", "izq")
        assert r.ok is False
        assert r.reason is not None
        assert "casa1" in r.reason
        assert "deposito" in r.reason


class TestValidatePickUpOrdenDeChequeos:
    def test_drone_inexistente_se_reporta_antes_que_paquete(
        self, mundo_para_recoger
    ):
        r = validate_pick_up(
            mundo_para_recoger, "fantasma", "caja_fantasma", "izq"
        )
        assert r.ok is False
        assert "fantasma" in r.reason  # type: ignore[operator]
        # No menciona la caja.
        assert "caja_fantasma" not in r.reason  # type: ignore[operator]

    def test_paquete_inexistente_se_reporta_antes_que_brazo(
        self, mundo_para_recoger
    ):
        r = validate_pick_up(
            mundo_para_recoger,
            "dron1",
            "caja_fantasma",
            "brazo_fantasma",
        )
        assert r.ok is False
        assert "caja_fantasma" in r.reason  # type: ignore[operator]
        assert "brazo_fantasma" not in r.reason  # type: ignore[operator]

    def test_brazo_inexistente_se_reporta_antes_que_brazo_ocupado(
        self, mundo_para_recoger
    ):
        # No tiene sentido decir "el brazo X está ocupado" si X no existe.
        r = validate_pick_up(
            mundo_para_recoger, "dron1", "caja1", "fantasmal"
        )
        assert r.ok is False
        assert "fantasmal" in r.reason  # type: ignore[operator]
        assert "ya sostiene" not in r.reason  # type: ignore[operator]


# ---------------------------------------------------------------------------
# validate_deliver
# ---------------------------------------------------------------------------


@pytest.fixture
def mundo_para_entregar() -> World:
    """Mundo donde el drone está en casa1 sosteniendo una caja de medicina
    con el brazo izq, y hay una persona en casa1 que necesita medicina."""
    medicina = Content("medicina")
    locations = {
        "deposito": Location("deposito"),
        "casa1": Location("casa1"),
    }
    drones = {
        "dron1": Drone(
            id="dron1",
            position="casa1",
            arms=(Arm("izq"), Arm("der")),
        )
    }
    packages = {
        "caja1": Package(
            id="caja1",
            contains=medicina,
            at=HeldByArm("dron1", "izq"),
        )
    }
    persons = {
        "p1": Person(id="p1", position="casa1", needs=(medicina,))
    }
    return World(
        locations=locations,
        drones=drones,
        packages=packages,
        persons=persons,
    )


class TestValidateDeliverCasoFeliz:
    def test_persona_colocalizada_paquete_sostenido_contenido_necesario(
        self, mundo_para_entregar
    ):
        r = validate_deliver(mundo_para_entregar, "dron1", "caja1", "p1")
        assert r.ok is True
        assert r.reason is None


class TestValidateDeliverReglasTransversales:
    def test_rechaza_drone_inexistente(self, mundo_para_entregar):
        r = validate_deliver(
            mundo_para_entregar, "fantasma", "caja1", "p1"
        )
        assert r.ok is False
        assert "fantasma" in r.reason  # type: ignore[operator]

    def test_rechaza_drone_en_error(self, mundo_para_entregar):
        roto = replace(
            mundo_para_entregar.drones["dron1"], state=DroneState.ERROR
        )
        w = replace(
            mundo_para_entregar,
            drones={**mundo_para_entregar.drones, "dron1": roto},
        )
        r = validate_deliver(w, "dron1", "caja1", "p1")
        assert r.ok is False
        assert "ERROR" in r.reason  # type: ignore[operator]

    def test_rechaza_drone_ocupado(self, mundo_para_entregar):
        ocupado = replace(
            mundo_para_entregar.drones["dron1"], state=DroneState.MOVING
        )
        w = replace(
            mundo_para_entregar,
            drones={**mundo_para_entregar.drones, "dron1": ocupado},
        )
        r = validate_deliver(w, "dron1", "caja1", "p1")
        assert r.ok is False


class TestValidateDeliverExistencias:
    def test_rechaza_persona_inexistente(self, mundo_para_entregar):
        r = validate_deliver(
            mundo_para_entregar, "dron1", "caja1", "p_fantasma"
        )
        assert r.ok is False
        assert "p_fantasma" in r.reason  # type: ignore[operator]

    def test_rechaza_paquete_inexistente(self, mundo_para_entregar):
        r = validate_deliver(
            mundo_para_entregar, "dron1", "caja_fantasma", "p1"
        )
        assert r.ok is False
        assert "caja_fantasma" in r.reason  # type: ignore[operator]


class TestValidateDeliverPaqueteSostenido:
    def test_rechaza_paquete_libre_en_localizacion(
        self, mundo_para_entregar
    ):
        # La caja está en el suelo, no sostenida.
        caja_libre = Package(
            id="caja1",
            contains=Content("medicina"),
            at=AtLocation("casa1"),
        )
        w = replace(mundo_para_entregar, packages={"caja1": caja_libre})
        r = validate_deliver(w, "dron1", "caja1", "p1")
        assert r.ok is False
        assert r.reason is not None
        assert "no está siendo sostenido" in r.reason

    def test_rechaza_paquete_sostenido_por_otro_drone(self):
        # Dos drones, ambos en casa1. La caja está en el brazo de dron2,
        # pero el comando es para dron1.
        medicina = Content("medicina")
        locations = {"casa1": Location("casa1")}
        drones = {
            "dron1": Drone(
                id="dron1", position="casa1", arms=(Arm("izq"),)
            ),
            "dron2": Drone(
                id="dron2", position="casa1", arms=(Arm("izq"),)
            ),
        }
        packages = {
            "caja1": Package(
                id="caja1",
                contains=medicina,
                at=HeldByArm("dron2", "izq"),
            )
        }
        persons = {
            "p1": Person(id="p1", position="casa1", needs=(medicina,))
        }
        w = World(
            locations=locations,
            drones=drones,
            packages=packages,
            persons=persons,
        )
        r = validate_deliver(w, "dron1", "caja1", "p1")
        assert r.ok is False
        assert "no está siendo sostenido" in r.reason  # type: ignore[operator]

    def test_rechaza_paquete_en_transporter(self, mundo_para_entregar):
        t1 = Transporter(id="t1", position="casa1", capacity=4)
        caja_dentro = Package(
            id="caja1",
            contains=Content("medicina"),
            at=InTransporter("t1"),
        )
        w = replace(
            mundo_para_entregar,
            transporters={"t1": t1},
            packages={"caja1": caja_dentro},
        )
        r = validate_deliver(w, "dron1", "caja1", "p1")
        assert r.ok is False
        assert "no está siendo sostenido" in r.reason  # type: ignore[operator]


class TestValidateDeliverColocalizacion:
    def test_rechaza_persona_en_otra_localizacion(self, mundo_para_entregar):
        # Movemos la persona a deposito; el drone sigue en casa1.
        p1_lejos = replace(
            mundo_para_entregar.persons["p1"], position="deposito"
        )
        w = replace(
            mundo_para_entregar,
            persons={**mundo_para_entregar.persons, "p1": p1_lejos},
        )
        r = validate_deliver(w, "dron1", "caja1", "p1")
        assert r.ok is False
        assert r.reason is not None
        assert "deposito" in r.reason
        assert "casa1" in r.reason
        assert "co-localizados" in r.reason


class TestValidateDeliverNecesidad:
    """La regla 17, la novedad que trajo el repaso del PDDL: no se puede
    entregar a alguien que no necesita ese contenido. El validator
    rechaza el comando como lo haría el planificador real al verificar
    la precondition (necesita ?p ?co)."""

    def test_rechaza_si_persona_no_necesita_el_contenido(
        self, mundo_para_entregar
    ):
        # La persona necesita medicina; le intentamos dar comida.
        comida = Content("comida")
        caja_de_comida = Package(
            id="caja1", contains=comida, at=HeldByArm("dron1", "izq")
        )
        w = replace(mundo_para_entregar, packages={"caja1": caja_de_comida})
        r = validate_deliver(w, "dron1", "caja1", "p1")
        assert r.ok is False
        assert r.reason is not None
        assert "no necesita" in r.reason
        assert "comida" in r.reason

    def test_rechaza_si_persona_ya_recibio_todo_lo_que_necesitaba(
        self, mundo_para_entregar
    ):
        # La persona ya no necesita nada (needs=()). El paquete sigue
        # siendo medicina, pero la persona no lo necesita.
        p1_saciada = replace(
            mundo_para_entregar.persons["p1"],
            needs=(),
            has_received=(Content("medicina"),),
        )
        w = replace(
            mundo_para_entregar,
            persons={**mundo_para_entregar.persons, "p1": p1_saciada},
        )
        r = validate_deliver(w, "dron1", "caja1", "p1")
        assert r.ok is False
        assert "no necesita" in r.reason  # type: ignore[operator]

    def test_acepta_si_persona_necesita_ese_contenido_entre_varios(self):
        # La persona necesita comida Y medicina. Le entregamos medicina:
        # válido aunque también necesite comida.
        medicina = Content("medicina")
        comida = Content("comida")
        locations = {"casa1": Location("casa1")}
        drones = {
            "dron1": Drone(
                id="dron1", position="casa1", arms=(Arm("izq"),)
            )
        }
        packages = {
            "caja1": Package(
                id="caja1",
                contains=medicina,
                at=HeldByArm("dron1", "izq"),
            )
        }
        persons = {
            "p1": Person(
                id="p1", position="casa1", needs=(comida, medicina)
            )
        }
        w = World(
            locations=locations,
            drones=drones,
            packages=packages,
            persons=persons,
        )
        r = validate_deliver(w, "dron1", "caja1", "p1")
        assert r.ok is True


class TestValidateDeliverOrdenDeChequeos:
    def test_drone_inexistente_se_reporta_antes_que_persona(
        self, mundo_para_entregar
    ):
        r = validate_deliver(
            mundo_para_entregar, "fantasma", "caja1", "p_fantasma"
        )
        assert r.ok is False
        assert "fantasma" in r.reason  # type: ignore[operator]
        # No menciona la persona.
        assert "p_fantasma" not in r.reason  # type: ignore[operator]

    def test_persona_inexistente_se_reporta_antes_que_paquete(
        self, mundo_para_entregar
    ):
        r = validate_deliver(
            mundo_para_entregar,
            "dron1",
            "caja_fantasma",
            "p_fantasma",
        )
        assert r.ok is False
        assert "p_fantasma" in r.reason  # type: ignore[operator]
        assert "caja_fantasma" not in r.reason  # type: ignore[operator]

    def test_paquete_no_sostenido_se_reporta_antes_que_necesidad(
        self, mundo_para_entregar
    ):
        # La caja está libre Y la persona no necesita comida. El primer
        # fallo (paquete no sostenido) se reporta antes que el segundo
        # (no necesita), porque "qué tiene en la mano" es más
        # fundamental que "es lo que la persona quiere".
        comida = Content("comida")
        caja_libre = Package(
            id="caja1", contains=comida, at=AtLocation("casa1")
        )
        w = replace(mundo_para_entregar, packages={"caja1": caja_libre})
        r = validate_deliver(w, "dron1", "caja1", "p1")
        assert r.ok is False
        assert "no está siendo sostenido" in r.reason  # type: ignore[operator]
        assert "no necesita" not in r.reason  # type: ignore[operator]


# ---------------------------------------------------------------------------
# validate_load_into_transporter
# ---------------------------------------------------------------------------


@pytest.fixture
def mundo_para_cargar() -> World:
    """Mundo donde el drone está en deposito sosteniendo una caja con
    el brazo izq, y hay un transportador co-localizado con capacidad 4
    y vacío."""
    medicina = Content("medicina")
    locations = {"deposito": Location("deposito")}
    drones = {
        "dron1": Drone(
            id="dron1",
            position="deposito",
            arms=(Arm("izq"), Arm("der")),
        )
    }
    transporters = {
        "t1": Transporter(id="t1", position="deposito", capacity=4)
    }
    packages = {
        "caja1": Package(
            id="caja1",
            contains=medicina,
            at=HeldByArm("dron1", "izq"),
        )
    }
    return World(
        locations=locations,
        drones=drones,
        transporters=transporters,
        packages=packages,
    )


class TestValidateLoadCasoFeliz:
    def test_caja_sostenida_transporter_colocalizado_no_lleno(
        self, mundo_para_cargar
    ):
        r = validate_load_into_transporter(
            mundo_para_cargar, "dron1", "caja1", "t1"
        )
        assert r.ok is True
        assert r.reason is None


class TestValidateLoadReglasTransversales:
    def test_rechaza_drone_inexistente(self, mundo_para_cargar):
        r = validate_load_into_transporter(
            mundo_para_cargar, "fantasma", "caja1", "t1"
        )
        assert r.ok is False
        assert "fantasma" in r.reason  # type: ignore[operator]

    def test_rechaza_drone_en_error(self, mundo_para_cargar):
        roto = replace(
            mundo_para_cargar.drones["dron1"], state=DroneState.ERROR
        )
        w = replace(
            mundo_para_cargar,
            drones={**mundo_para_cargar.drones, "dron1": roto},
        )
        r = validate_load_into_transporter(w, "dron1", "caja1", "t1")
        assert r.ok is False
        assert "ERROR" in r.reason  # type: ignore[operator]


class TestValidateLoadExistencias:
    def test_rechaza_paquete_inexistente(self, mundo_para_cargar):
        r = validate_load_into_transporter(
            mundo_para_cargar, "dron1", "caja_fantasma", "t1"
        )
        assert r.ok is False
        assert "caja_fantasma" in r.reason  # type: ignore[operator]

    def test_rechaza_transporter_inexistente(self, mundo_para_cargar):
        r = validate_load_into_transporter(
            mundo_para_cargar, "dron1", "caja1", "t_fantasma"
        )
        assert r.ok is False
        assert "t_fantasma" in r.reason  # type: ignore[operator]


class TestValidateLoadColocalizacion:
    def test_rechaza_transporter_en_otra_localizacion(
        self, mundo_para_cargar
    ):
        # Necesitamos otra localización para mover el transporter allí.
        locations = {
            "deposito": Location("deposito"),
            "casa1": Location("casa1"),
        }
        t_lejos = replace(
            mundo_para_cargar.transporters["t1"], position="casa1"
        )
        w = replace(
            mundo_para_cargar,
            locations=locations,
            transporters={"t1": t_lejos},
        )
        r = validate_load_into_transporter(w, "dron1", "caja1", "t1")
        assert r.ok is False
        assert r.reason is not None
        assert "co-localizados" in r.reason
        assert "casa1" in r.reason
        assert "deposito" in r.reason


class TestValidateLoadPaqueteSostenido:
    def test_rechaza_paquete_libre(self, mundo_para_cargar):
        caja_libre = Package(
            id="caja1",
            contains=Content("medicina"),
            at=AtLocation("deposito"),
        )
        w = replace(mundo_para_cargar, packages={"caja1": caja_libre})
        r = validate_load_into_transporter(w, "dron1", "caja1", "t1")
        assert r.ok is False
        assert "no está siendo sostenido" in r.reason  # type: ignore[operator]

    def test_rechaza_paquete_ya_en_transporter(self, mundo_para_cargar):
        # La caja ya está dentro del transportador.
        caja_dentro = Package(
            id="caja1",
            contains=Content("medicina"),
            at=InTransporter("t1"),
        )
        w = replace(mundo_para_cargar, packages={"caja1": caja_dentro})
        r = validate_load_into_transporter(w, "dron1", "caja1", "t1")
        assert r.ok is False
        assert "no está siendo sostenido" in r.reason  # type: ignore[operator]

    def test_rechaza_paquete_sostenido_por_otro_drone(self):
        # Dos drones co-localizados, la caja está en el brazo del segundo.
        medicina = Content("medicina")
        locations = {"deposito": Location("deposito")}
        drones = {
            "dron1": Drone(
                id="dron1", position="deposito", arms=(Arm("izq"),)
            ),
            "dron2": Drone(
                id="dron2", position="deposito", arms=(Arm("izq"),)
            ),
        }
        transporters = {
            "t1": Transporter(id="t1", position="deposito", capacity=4)
        }
        packages = {
            "caja1": Package(
                id="caja1",
                contains=medicina,
                at=HeldByArm("dron2", "izq"),
            )
        }
        w = World(
            locations=locations,
            drones=drones,
            transporters=transporters,
            packages=packages,
        )
        r = validate_load_into_transporter(w, "dron1", "caja1", "t1")
        assert r.ok is False
        assert "no está siendo sostenido" in r.reason  # type: ignore[operator]


class TestValidateLoadCapacidad:
    """La regla 21: el transportador no excede su capacidad.
    Representa el mecanismo PDDL de `capacidad-disponible` + `siguiente`
    de forma directa con una comparación entera."""

    def test_rechaza_transporter_lleno(self, mundo_para_cargar):
        # Llenamos t1 con 4 cajas (la capacidad). caja1 sigue en mano.
        medicina = Content("medicina")
        cajas_dentro = {
            f"c{i}": Package(
                id=f"c{i}",
                contains=medicina,
                at=InTransporter("t1"),
            )
            for i in range(4)
        }
        # Añadimos las 4 al world, manteniendo caja1 en mano.
        packages = {**mundo_para_cargar.packages, **cajas_dentro}
        w = replace(mundo_para_cargar, packages=packages)
        r = validate_load_into_transporter(w, "dron1", "caja1", "t1")
        assert r.ok is False
        assert r.reason is not None
        assert "lleno" in r.reason
        assert "4/4" in r.reason

    def test_acepta_si_queda_un_hueco(self, mundo_para_cargar):
        # Cargamos t1 con 3 cajas; capacidad 4. Queda un hueco.
        medicina = Content("medicina")
        cajas_dentro = {
            f"c{i}": Package(
                id=f"c{i}",
                contains=medicina,
                at=InTransporter("t1"),
            )
            for i in range(3)
        }
        packages = {**mundo_para_cargar.packages, **cajas_dentro}
        w = replace(mundo_para_cargar, packages=packages)
        r = validate_load_into_transporter(w, "dron1", "caja1", "t1")
        assert r.ok is True

    def test_rechaza_transporter_con_capacidad_cero(self, mundo_para_cargar):
        # Capacidad declarada 0: el Validator es coherente con la
        # comparación. Aunque el builder del facade rechazará capacities
        # no positivas, si el dato llega así, lo tratamos correctamente.
        t_cero = replace(
            mundo_para_cargar.transporters["t1"], capacity=0
        )
        w = replace(mundo_para_cargar, transporters={"t1": t_cero})
        r = validate_load_into_transporter(w, "dron1", "caja1", "t1")
        assert r.ok is False
        assert "0/0" in r.reason  # type: ignore[operator]


class TestValidateLoadOrdenDeChequeos:
    def test_drone_inexistente_se_reporta_antes_que_paquete(
        self, mundo_para_cargar
    ):
        r = validate_load_into_transporter(
            mundo_para_cargar, "fantasma", "caja_fantasma", "t_fantasma"
        )
        assert r.ok is False
        assert "fantasma" in r.reason  # type: ignore[operator]
        # No menciona la caja ni el transporter.
        assert "caja_fantasma" not in r.reason  # type: ignore[operator]
        assert "t_fantasma" not in r.reason  # type: ignore[operator]

    def test_paquete_inexistente_se_reporta_antes_que_transporter(
        self, mundo_para_cargar
    ):
        r = validate_load_into_transporter(
            mundo_para_cargar, "dron1", "caja_fantasma", "t_fantasma"
        )
        assert r.ok is False
        assert "caja_fantasma" in r.reason  # type: ignore[operator]
        assert "t_fantasma" not in r.reason  # type: ignore[operator]

    def test_transporter_inexistente_se_reporta_antes_que_colocalizacion(
        self, mundo_para_cargar
    ):
        # Si el transporter ni existe, no tiene sentido hablar de
        # co-localización.
        r = validate_load_into_transporter(
            mundo_para_cargar, "dron1", "caja1", "t_fantasma"
        )
        assert r.ok is False
        assert "no existe" in r.reason  # type: ignore[operator]
        assert "co-localizados" not in r.reason  # type: ignore[operator]

    def test_colocalizacion_se_reporta_antes_que_capacidad(self):
        # Transporter en otra localización Y lleno. El primer fallo
        # (no co-localizado) se reporta antes que el segundo (lleno),
        # porque la posición es más fundamental.
        medicina = Content("medicina")
        locations = {
            "deposito": Location("deposito"),
            "casa1": Location("casa1"),
        }
        drones = {
            "dron1": Drone(
                id="dron1", position="deposito", arms=(Arm("izq"),)
            )
        }
        transporters = {
            "t1": Transporter(id="t1", position="casa1", capacity=2)
        }
        # Lleno el transporter (que además está en casa1).
        packages = {
            "caja_en_mano": Package(
                id="caja_en_mano",
                contains=medicina,
                at=HeldByArm("dron1", "izq"),
            ),
            "c0": Package(
                id="c0", contains=medicina, at=InTransporter("t1")
            ),
            "c1": Package(
                id="c1", contains=medicina, at=InTransporter("t1")
            ),
        }
        w = World(
            locations=locations,
            drones=drones,
            transporters=transporters,
            packages=packages,
        )
        r = validate_load_into_transporter(
            w, "dron1", "caja_en_mano", "t1"
        )
        assert r.ok is False
        assert "co-localizados" in r.reason  # type: ignore[operator]
        assert "lleno" not in r.reason  # type: ignore[operator]

    def test_paquete_no_sostenido_se_reporta_antes_que_capacidad(
        self, mundo_para_cargar
    ):
        # Llenamos t1 al máximo Y dejamos caja1 libre (no sostenida).
        # Esperamos que se reporte primero "no sostenido", porque qué
        # tienes en la mano es más fundamental que cuánto cabe.
        medicina = Content("medicina")
        caja_libre = Package(
            id="caja1", contains=medicina, at=AtLocation("deposito")
        )
        cajas_dentro = {
            f"c{i}": Package(
                id=f"c{i}",
                contains=medicina,
                at=InTransporter("t1"),
            )
            for i in range(4)
        }
        w = replace(
            mundo_para_cargar,
            packages={"caja1": caja_libre, **cajas_dentro},
        )
        r = validate_load_into_transporter(w, "dron1", "caja1", "t1")
        assert r.ok is False
        assert "no está siendo sostenido" in r.reason  # type: ignore[operator]
        assert "lleno" not in r.reason  # type: ignore[operator]


# ---------------------------------------------------------------------------
# validate_unload_from_transporter
# ---------------------------------------------------------------------------


@pytest.fixture
def mundo_para_descargar() -> World:
    """Mundo donde el drone está en deposito con dos brazos libres,
    y hay un transportador co-localizado con una caja dentro."""
    medicina = Content("medicina")
    locations = {"deposito": Location("deposito")}
    drones = {
        "dron1": Drone(
            id="dron1",
            position="deposito",
            arms=(Arm("izq"), Arm("der")),
        )
    }
    transporters = {
        "t1": Transporter(id="t1", position="deposito", capacity=4)
    }
    packages = {
        "caja1": Package(
            id="caja1", contains=medicina, at=InTransporter("t1")
        )
    }
    return World(
        locations=locations,
        drones=drones,
        transporters=transporters,
        packages=packages,
    )


class TestValidateUnloadCasoFeliz:
    def test_paquete_en_transporter_brazo_libre_colocalizados(
        self, mundo_para_descargar
    ):
        r = validate_unload_from_transporter(
            mundo_para_descargar, "dron1", "caja1", "t1", "izq"
        )
        assert r.ok is True
        assert r.reason is None

    def test_se_puede_descargar_con_cualquier_brazo_libre(
        self, mundo_para_descargar
    ):
        r1 = validate_unload_from_transporter(
            mundo_para_descargar, "dron1", "caja1", "t1", "izq"
        )
        r2 = validate_unload_from_transporter(
            mundo_para_descargar, "dron1", "caja1", "t1", "der"
        )
        assert r1.ok is True
        assert r2.ok is True


class TestValidateUnloadReglasTransversales:
    def test_rechaza_drone_inexistente(self, mundo_para_descargar):
        r = validate_unload_from_transporter(
            mundo_para_descargar, "fantasma", "caja1", "t1", "izq"
        )
        assert r.ok is False
        assert "fantasma" in r.reason  # type: ignore[operator]

    def test_rechaza_drone_en_error(self, mundo_para_descargar):
        roto = replace(
            mundo_para_descargar.drones["dron1"], state=DroneState.ERROR
        )
        w = replace(
            mundo_para_descargar,
            drones={**mundo_para_descargar.drones, "dron1": roto},
        )
        r = validate_unload_from_transporter(w, "dron1", "caja1", "t1", "izq")
        assert r.ok is False
        assert "ERROR" in r.reason  # type: ignore[operator]


class TestValidateUnloadExistencias:
    def test_rechaza_paquete_inexistente(self, mundo_para_descargar):
        r = validate_unload_from_transporter(
            mundo_para_descargar, "dron1", "caja_fantasma", "t1", "izq"
        )
        assert r.ok is False
        assert "caja_fantasma" in r.reason  # type: ignore[operator]

    def test_rechaza_transporter_inexistente(self, mundo_para_descargar):
        r = validate_unload_from_transporter(
            mundo_para_descargar,
            "dron1",
            "caja1",
            "t_fantasma",
            "izq",
        )
        assert r.ok is False
        assert "t_fantasma" in r.reason  # type: ignore[operator]


class TestValidateUnloadColocalizacion:
    def test_rechaza_transporter_en_otra_localizacion(
        self, mundo_para_descargar
    ):
        locations = {
            "deposito": Location("deposito"),
            "casa1": Location("casa1"),
        }
        t_lejos = replace(
            mundo_para_descargar.transporters["t1"], position="casa1"
        )
        w = replace(
            mundo_para_descargar,
            locations=locations,
            transporters={"t1": t_lejos},
        )
        r = validate_unload_from_transporter(w, "dron1", "caja1", "t1", "izq")
        assert r.ok is False
        assert "co-localizados" in r.reason  # type: ignore[operator]


class TestValidateUnloadRelacionPaqueteTransporter:
    """El paquete debe estar dentro de ESE transporter concreto. Tres
    situaciones a rechazar: paquete libre, paquete en otro brazo,
    paquete en otro transporter."""

    def test_rechaza_paquete_libre_en_localizacion(
        self, mundo_para_descargar
    ):
        caja_libre = Package(
            id="caja1",
            contains=Content("medicina"),
            at=AtLocation("deposito"),
        )
        w = replace(mundo_para_descargar, packages={"caja1": caja_libre})
        r = validate_unload_from_transporter(w, "dron1", "caja1", "t1", "izq")
        assert r.ok is False
        assert r.reason is not None
        assert "no está en el transportador" in r.reason

    def test_rechaza_paquete_sostenido_por_un_brazo(
        self, mundo_para_descargar
    ):
        caja_en_mano = Package(
            id="caja1",
            contains=Content("medicina"),
            at=HeldByArm("dron1", "der"),
        )
        w = replace(mundo_para_descargar, packages={"caja1": caja_en_mano})
        r = validate_unload_from_transporter(w, "dron1", "caja1", "t1", "izq")
        assert r.ok is False
        assert "no está en el transportador" in r.reason  # type: ignore[operator]

    def test_rechaza_paquete_en_otro_transporter(self):
        # Dos transporters co-localizados con el drone. La caja está
        # en t2, intentamos descargarla de t1.
        medicina = Content("medicina")
        locations = {"deposito": Location("deposito")}
        drones = {
            "dron1": Drone(
                id="dron1", position="deposito", arms=(Arm("izq"),)
            )
        }
        transporters = {
            "t1": Transporter(id="t1", position="deposito", capacity=4),
            "t2": Transporter(id="t2", position="deposito", capacity=4),
        }
        packages = {
            "caja1": Package(
                id="caja1", contains=medicina, at=InTransporter("t2")
            )
        }
        w = World(
            locations=locations,
            drones=drones,
            transporters=transporters,
            packages=packages,
        )
        r = validate_unload_from_transporter(w, "dron1", "caja1", "t1", "izq")
        assert r.ok is False
        assert "no está en el transportador" in r.reason  # type: ignore[operator]
        assert "t1" in r.reason  # type: ignore[operator]


class TestValidateUnloadBrazo:
    def test_rechaza_brazo_inexistente(self, mundo_para_descargar):
        r = validate_unload_from_transporter(
            mundo_para_descargar, "dron1", "caja1", "t1", "central"
        )
        assert r.ok is False
        assert "central" in r.reason  # type: ignore[operator]

    def test_rechaza_brazo_ocupado(self, mundo_para_descargar):
        # El brazo izq ya sostiene otra caja.
        medicina = Content("medicina")
        otra_caja = Package(
            id="otra",
            contains=medicina,
            at=HeldByArm("dron1", "izq"),
        )
        packages = {**mundo_para_descargar.packages, "otra": otra_caja}
        w = replace(mundo_para_descargar, packages=packages)
        r = validate_unload_from_transporter(w, "dron1", "caja1", "t1", "izq")
        assert r.ok is False
        assert r.reason is not None
        assert "otra" in r.reason  # menciona el paquete que ocupa

    def test_acepta_si_el_otro_brazo_esta_libre(self, mundo_para_descargar):
        # Mismo setup que el test anterior, pero usamos el brazo der.
        medicina = Content("medicina")
        otra_caja = Package(
            id="otra",
            contains=medicina,
            at=HeldByArm("dron1", "izq"),
        )
        packages = {**mundo_para_descargar.packages, "otra": otra_caja}
        w = replace(mundo_para_descargar, packages=packages)
        r = validate_unload_from_transporter(w, "dron1", "caja1", "t1", "der")
        assert r.ok is True


class TestValidateUnloadOrdenDeChequeos:
    def test_drone_inexistente_se_reporta_primero(
        self, mundo_para_descargar
    ):
        r = validate_unload_from_transporter(
            mundo_para_descargar,
            "fantasma",
            "caja_fantasma",
            "t_fantasma",
            "brazo_fantasma",
        )
        assert r.ok is False
        assert "fantasma" in r.reason  # type: ignore[operator]
        # No menciona ninguno de los otros que tampoco existen.
        assert "caja_fantasma" not in r.reason  # type: ignore[operator]
        assert "t_fantasma" not in r.reason  # type: ignore[operator]
        assert "brazo_fantasma" not in r.reason  # type: ignore[operator]

    def test_paquete_inexistente_se_reporta_antes_que_transporter(
        self, mundo_para_descargar
    ):
        r = validate_unload_from_transporter(
            mundo_para_descargar,
            "dron1",
            "caja_fantasma",
            "t_fantasma",
            "izq",
        )
        assert r.ok is False
        assert "caja_fantasma" in r.reason  # type: ignore[operator]
        assert "t_fantasma" not in r.reason  # type: ignore[operator]

    def test_relacion_se_reporta_antes_que_brazo(self, mundo_para_descargar):
        # La caja no está en t1 Y el brazo no existe. El primer fallo
        # (relación paquete-transporter) se reporta antes que el brazo
        # inexistente, porque la pregunta "¿qué hay en el transporter?"
        # es más fundamental que "¿cómo lo agarro?".
        caja_libre = Package(
            id="caja1",
            contains=Content("medicina"),
            at=AtLocation("deposito"),
        )
        w = replace(mundo_para_descargar, packages={"caja1": caja_libre})
        r = validate_unload_from_transporter(
            w, "dron1", "caja1", "t1", "fantasmal"
        )
        assert r.ok is False
        assert "no está en el transportador" in r.reason  # type: ignore[operator]
        assert "fantasmal" not in r.reason  # type: ignore[operator]

    def test_brazo_inexistente_se_reporta_antes_que_brazo_ocupado(
        self, mundo_para_descargar
    ):
        # Brazo "fantasmal" no existe. Aunque ningún brazo existente
        # esté libre (caso construido), se reporta el inexistente
        # primero porque no tiene sentido preguntar si un brazo que no
        # existe está libre.
        r = validate_unload_from_transporter(
            mundo_para_descargar, "dron1", "caja1", "t1", "fantasmal"
        )
        assert r.ok is False
        assert "fantasmal" in r.reason  # type: ignore[operator]
        assert "ya sostiene" not in r.reason  # type: ignore[operator]
