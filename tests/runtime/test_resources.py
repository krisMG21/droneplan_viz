"""Tests de runtime/resources.py: resources_of, ResourceTable, CollisionInfo.

Cubre:
- resources_of(): qué recursos ocupa cada uno de los seis tipos de
  Command. Esta es la traducción directa de las cinco reglas del PDF
  parte 3 a código.
- _intervals_overlap (indirectamente vía check): casos borde de
  intervalos solapados, disjuntos, back-to-back, puntuales.
- ResourceTable: check, reserve, release_before, reservations_for,
  __len__, is_empty.
- CollisionInfo.describe(): mensajes legibles.

NO testea aquí el runner completo. Eso vendrá en pasos posteriores
con tests de integración que ejercen estas piezas juntas.
"""
from __future__ import annotations

import pytest

from droneplan_viz.commands import (
    Deliver,
    LoadIntoTransporter,
    Move,
    MoveWithTransporter,
    PickUp,
    UnloadFromTransporter,
)
from droneplan_viz.runtime.resources import (
    CollisionInfo,
    Reservation,
    Resource,
    ResourceTable,
    _intervals_overlap,
    resources_of,
)


# ===========================================================================
# resources_of: mapeo Command -> recursos
# ===========================================================================
class TestResourcesOfMove:
    def test_move_solo_ocupa_drone(self):
        cmd = Move(drone_id="d1", destination_id="casa1")
        assert resources_of(cmd) == (("drone", "d1"),)

    def test_move_drone_distinto(self):
        cmd = Move(drone_id="d_otro", destination_id="x")
        assert resources_of(cmd) == (("drone", "d_otro"),)

    def test_move_no_ocupa_localizacion(self):
        """REGLA NEGATIVA crítica: el PDF parte 3 no menciona
        localizaciones como recurso exclusivo. Dos drones pueden
        coexistir en la misma Location."""
        cmd = Move(drone_id="d1", destination_id="casa1")
        resources = resources_of(cmd)
        assert not any(r[0] == "location" for r in resources)


class TestResourcesOfMoveWithTransporter:
    def test_ocupa_drone_y_transportador(self):
        cmd = MoveWithTransporter(
            drone_id="d1", transporter_id="t1", destination_id="casa1"
        )
        result = resources_of(cmd)
        assert ("drone", "d1") in result
        assert ("transporter", "t1") in result
        assert len(result) == 2

    def test_no_ocupa_brazo(self):
        """El PDF parte 3 no menciona los brazos como recurso. La regla
        1 (un dron, una acción) cubre la exclusión sin granularidad
        adicional."""
        cmd = MoveWithTransporter(
            drone_id="d1", transporter_id="t1", destination_id="casa1"
        )
        resources = resources_of(cmd)
        assert not any(r[0] == "arm" for r in resources)


class TestResourcesOfPickUp:
    def test_ocupa_drone_y_paquete(self):
        cmd = PickUp(drone_id="d1", arm_id="izq", package_id="p1")
        result = resources_of(cmd)
        assert ("drone", "d1") in result
        assert ("package", "p1") in result
        assert len(result) == 2

    def test_no_ocupa_brazo_ni_transportador_ni_persona(self):
        cmd = PickUp(drone_id="d1", arm_id="izq", package_id="p1")
        result = resources_of(cmd)
        assert not any(r[0] == "arm" for r in result)
        assert not any(r[0] == "transporter" for r in result)
        assert not any(r[0] == "person" for r in result)


class TestResourcesOfDeliver:
    def test_ocupa_drone_paquete_y_persona(self):
        cmd = Deliver(drone_id="d1", package_id="p1", person_id="ana")
        result = resources_of(cmd)
        assert ("drone", "d1") in result
        assert ("package", "p1") in result
        assert ("person", "ana") in result
        assert len(result) == 3

    def test_no_ocupa_transportador(self):
        cmd = Deliver(drone_id="d1", package_id="p1", person_id="ana")
        result = resources_of(cmd)
        assert not any(r[0] == "transporter" for r in result)


class TestResourcesOfLoadIntoTransporter:
    def test_ocupa_drone_paquete_y_transportador(self):
        cmd = LoadIntoTransporter(
            drone_id="d1", package_id="p1", transporter_id="t1"
        )
        result = resources_of(cmd)
        assert ("drone", "d1") in result
        assert ("package", "p1") in result
        assert ("transporter", "t1") in result
        assert len(result) == 3

    def test_no_ocupa_persona(self):
        cmd = LoadIntoTransporter(
            drone_id="d1", package_id="p1", transporter_id="t1"
        )
        result = resources_of(cmd)
        assert not any(r[0] == "person" for r in result)


class TestResourcesOfUnloadFromTransporter:
    def test_ocupa_drone_paquete_y_transportador(self):
        cmd = UnloadFromTransporter(
            drone_id="d1", arm_id="izq",
            package_id="p1", transporter_id="t1",
        )
        result = resources_of(cmd)
        assert ("drone", "d1") in result
        assert ("package", "p1") in result
        assert ("transporter", "t1") in result
        assert len(result) == 3

    def test_no_ocupa_brazo_ni_persona(self):
        cmd = UnloadFromTransporter(
            drone_id="d1", arm_id="izq",
            package_id="p1", transporter_id="t1",
        )
        result = resources_of(cmd)
        assert not any(r[0] == "arm" for r in result)
        assert not any(r[0] == "person" for r in result)


class TestResourcesOfTypeError:
    def test_objeto_no_command_lanza_typeerror(self):
        with pytest.raises(TypeError, match="no es un Command"):
            resources_of("no soy un command")  # type: ignore[arg-type]

    def test_none_lanza_typeerror(self):
        with pytest.raises(TypeError):
            resources_of(None)  # type: ignore[arg-type]


# ===========================================================================
# _intervals_overlap: casos borde
# ===========================================================================
class TestIntervalsOverlap:
    """La función auxiliar que decide si dos intervalos [start, end)
    solapan. Es la base de toda la lógica de concurrencia."""

    def test_intervalos_iguales_solapan(self):
        assert _intervals_overlap(0.0, 10.0, 0.0, 10.0) is True

    def test_intervalos_disjuntos_no_solapan(self):
        assert _intervals_overlap(0.0, 5.0, 10.0, 15.0) is False

    def test_disjuntos_orden_inverso_no_solapan(self):
        assert _intervals_overlap(10.0, 15.0, 0.0, 5.0) is False

    def test_back_to_back_no_solapan(self):
        """Cerrado-abierto: [0, 5) y [5, 10) NO solapan."""
        assert _intervals_overlap(0.0, 5.0, 5.0, 10.0) is False

    def test_back_to_back_inverso_no_solapan(self):
        assert _intervals_overlap(5.0, 10.0, 0.0, 5.0) is False

    def test_solape_parcial_inicio(self):
        """[0,10) y [5,15): solapan en [5,10)."""
        assert _intervals_overlap(0.0, 10.0, 5.0, 15.0) is True

    def test_solape_parcial_fin(self):
        assert _intervals_overlap(5.0, 15.0, 0.0, 10.0) is True

    def test_contenido_solapa(self):
        """[0,20) contiene [5,15)."""
        assert _intervals_overlap(0.0, 20.0, 5.0, 15.0) is True

    def test_contenido_inverso_solapa(self):
        assert _intervals_overlap(5.0, 15.0, 0.0, 20.0) is True

    def test_punto_no_solapa_con_intervalo_real(self):
        """Intervalo puntual [5,5): vacío por convención. NO solapa."""
        assert _intervals_overlap(5.0, 5.0, 0.0, 10.0) is False

    def test_punto_no_solapa_con_otro_punto(self):
        assert _intervals_overlap(5.0, 5.0, 5.0, 5.0) is False

    def test_segundo_argumento_punto_no_solapa(self):
        assert _intervals_overlap(0.0, 10.0, 7.0, 7.0) is False


# ===========================================================================
# ResourceTable: estado inicial
# ===========================================================================
class TestResourceTableInicial:
    def test_tabla_vacia_al_crearse(self):
        t = ResourceTable()
        assert t.is_empty() is True
        assert len(t) == 0

    def test_check_en_tabla_vacia_no_da_colision(self):
        t = ResourceTable()
        result = t.check((("drone", "d1"),), 0.0, 10.0)
        assert result is None

    def test_reservations_for_recurso_inexistente_es_tupla_vacia(self):
        t = ResourceTable()
        assert t.reservations_for(("drone", "d1")) == ()


# ===========================================================================
# ResourceTable.reserve + check: caso simple
# ===========================================================================
class TestResourceTableReservasBasicas:
    def test_reservar_un_recurso_aumenta_len(self):
        t = ResourceTable()
        t.reserve((("drone", "d1"),), 0.0, 10.0, owner="cmd_a")
        assert len(t) == 1

    def test_reservar_varios_recursos_aumenta_len_proporcionalmente(self):
        """Cada par (recurso, intervalo) cuenta como una reserva.
        Un Command que ocupa 3 recursos genera 3 entradas."""
        t = ResourceTable()
        t.reserve(
            (("drone", "d1"), ("package", "p1"), ("person", "ana")),
            0.0, 10.0, owner="entrega_1",
        )
        assert len(t) == 3

    def test_reservas_dos_commands_distintos_acumulan(self):
        t = ResourceTable()
        t.reserve((("drone", "d1"),), 0.0, 10.0, owner="a")
        t.reserve((("drone", "d2"),), 0.0, 10.0, owner="b")
        assert len(t) == 2

    def test_reservations_for_devuelve_la_reserva(self):
        t = ResourceTable()
        t.reserve((("drone", "d1"),), 0.0, 10.0, owner="cmd_a")
        res = t.reservations_for(("drone", "d1"))
        assert len(res) == 1
        assert res[0].start == 0.0
        assert res[0].end == 10.0
        assert res[0].owner == "cmd_a"

    def test_reservations_for_devuelve_tupla_no_lista(self):
        """No exponemos la lista interna mutable."""
        t = ResourceTable()
        t.reserve((("drone", "d1"),), 0.0, 10.0, owner="a")
        res = t.reservations_for(("drone", "d1"))
        assert isinstance(res, tuple)


class TestResourceTableReservaVacia:
    """Reservas con duración 0 (start >= end) no se almacenan."""

    def test_reserva_start_igual_end_no_almacena(self):
        t = ResourceTable()
        t.reserve((("drone", "d1"),), 5.0, 5.0, owner="cmd_a")
        assert t.is_empty()

    def test_reserva_start_mayor_que_end_no_almacena(self):
        """Caso patológico: el caller pasa un intervalo invertido. No
        almacenamos en lugar de lanzar; el runtime ya valida duraciones
        no negativas en Plan.__post_init__."""
        t = ResourceTable()
        t.reserve((("drone", "d1"),), 10.0, 5.0, owner="cmd_a")
        assert t.is_empty()

    def test_check_intervalo_vacio_no_colisiona(self):
        t = ResourceTable()
        t.reserve((("drone", "d1"),), 0.0, 20.0, owner="ocupa_largo")
        # Comprobamos un intervalo puntual EN MEDIO de uno ocupado:
        # como el intervalo es vacío, no colisiona.
        result = t.check((("drone", "d1"),), 10.0, 10.0)
        assert result is None


# ===========================================================================
# ResourceTable.check: detección de colisiones
# ===========================================================================
class TestResourceTableCheck:
    def test_colision_intervalos_iguales(self):
        t = ResourceTable()
        t.reserve((("drone", "d1"),), 0.0, 10.0, owner="a")
        result = t.check((("drone", "d1"),), 0.0, 10.0)
        assert result is not None
        assert result.resource == ("drone", "d1")
        assert result.existing.owner == "a"

    def test_colision_solape_parcial(self):
        t = ResourceTable()
        t.reserve((("drone", "d1"),), 0.0, 10.0, owner="a")
        result = t.check((("drone", "d1"),), 5.0, 15.0)
        assert result is not None
        assert result.attempted_start == 5.0
        assert result.attempted_end == 15.0

    def test_no_colision_intervalos_disjuntos(self):
        t = ResourceTable()
        t.reserve((("drone", "d1"),), 0.0, 10.0, owner="a")
        result = t.check((("drone", "d1"),), 20.0, 30.0)
        assert result is None

    def test_no_colision_back_to_back(self):
        """Caso critical: dos Commands en el mismo dron, el segundo
        empieza exactamente cuando el primero termina. NO debe colisionar."""
        t = ResourceTable()
        t.reserve((("drone", "d1"),), 0.0, 10.0, owner="a")
        result = t.check((("drone", "d1"),), 10.0, 20.0)
        assert result is None

    def test_no_colision_recursos_distintos(self):
        """Drones distintos no comparten recurso. Pueden actuar en
        paralelo aunque sus intervalos se solapen completamente."""
        t = ResourceTable()
        t.reserve((("drone", "d1"),), 0.0, 10.0, owner="a")
        result = t.check((("drone", "d2"),), 0.0, 10.0)
        assert result is None

    def test_colision_detecta_primer_recurso_compartido(self):
        """Si el Command nuevo comparte VARIOS recursos con reservas
        existentes, check devuelve la primera colisión encontrada
        (orden de recursos del Command nuevo)."""
        t = ResourceTable()
        # Una reserva ocupa solo el paquete.
        t.reserve((("package", "p1"),), 0.0, 10.0, owner="pkg_owner")
        # Otra reserva ocupa solo el drone.
        t.reserve((("drone", "d1"),), 0.0, 10.0, owner="dr_owner")
        # Intentamos un PickUp: ocupa (drone, d1), (package, p1).
        # Primero choca con drone (porque está primero en resources_of).
        result = t.check(
            (("drone", "d1"), ("package", "p1")), 5.0, 15.0
        )
        assert result is not None
        assert result.resource == ("drone", "d1")
        assert result.existing.owner == "dr_owner"

    def test_colision_chequea_solo_recursos_solicitados(self):
        """Reservar en drone no afecta a chequeo sobre paquete."""
        t = ResourceTable()
        t.reserve((("drone", "d1"),), 0.0, 10.0, owner="a")
        result = t.check((("package", "p1"),), 0.0, 10.0)
        assert result is None


# ===========================================================================
# ResourceTable.check: separación entre check y reserve
# ===========================================================================
class TestResourceTableCheckYReserveSonSeparados:
    """El runner llama check() primero; si pasa, llama reserve(). Si
    check falla, NO debe haber side-effects."""

    def test_check_no_modifica_tabla(self):
        t = ResourceTable()
        t.reserve((("drone", "d1"),), 0.0, 10.0, owner="a")
        len_antes = len(t)
        t.check((("drone", "d1"),), 5.0, 15.0)
        assert len(t) == len_antes

    def test_check_que_falla_no_almacena_intento(self):
        t = ResourceTable()
        t.reserve((("drone", "d1"),), 0.0, 10.0, owner="a")
        result = t.check((("drone", "d1"),), 5.0, 15.0)
        assert result is not None  # ha fallado
        # La tabla sigue con una sola reserva (la original).
        assert len(t) == 1
        assert t.reservations_for(("drone", "d1"))[0].owner == "a"


# ===========================================================================
# ResourceTable.release_before: poda de reservas obsoletas
# ===========================================================================
class TestResourceTableReleaseBefore:
    def test_release_before_descarta_reservas_expiradas(self):
        t = ResourceTable()
        t.reserve((("drone", "d1"),), 0.0, 10.0, owner="vieja")
        t.reserve((("drone", "d1"),), 20.0, 30.0, owner="nueva")
        t.release_before(15.0)
        res = t.reservations_for(("drone", "d1"))
        assert len(res) == 1
        assert res[0].owner == "nueva"

    def test_release_before_con_end_igual_a_now_descarta(self):
        """Cerrado-abierto: una reserva con end == now ya está completada.
        Se descarta."""
        t = ResourceTable()
        t.reserve((("drone", "d1"),), 0.0, 10.0, owner="x")
        t.release_before(10.0)
        assert t.is_empty()

    def test_release_before_no_toca_reservas_aun_activas(self):
        t = ResourceTable()
        t.reserve((("drone", "d1"),), 0.0, 20.0, owner="larga")
        t.release_before(10.0)  # release a la mitad
        assert len(t) == 1

    def test_release_before_elimina_clave_si_queda_vacia(self):
        """Optimización: si tras release todos los recursos están vacíos,
        las claves se eliminan del dict para mantener la tabla compacta."""
        t = ResourceTable()
        t.reserve((("drone", "d1"),), 0.0, 10.0, owner="x")
        t.release_before(20.0)
        assert t.reservations_for(("drone", "d1")) == ()
        # La tabla está completamente vacía:
        assert t.is_empty()

    def test_release_before_en_tabla_vacia_no_falla(self):
        t = ResourceTable()
        t.release_before(100.0)  # no lanza
        assert t.is_empty()


# ===========================================================================
# CollisionInfo: información y mensaje
# ===========================================================================
class TestCollisionInfo:
    def test_construccion(self):
        existing = Reservation(start=0.0, end=10.0, owner="a")
        info = CollisionInfo(
            resource=("drone", "d1"),
            existing=existing,
            attempted_start=5.0,
            attempted_end=15.0,
        )
        assert info.resource == ("drone", "d1")
        assert info.existing is existing
        assert info.attempted_start == 5.0
        assert info.attempted_end == 15.0

    def test_describe_incluye_recurso(self):
        existing = Reservation(start=0.0, end=10.0, owner="a")
        info = CollisionInfo(
            resource=("drone", "d1"),
            existing=existing,
            attempted_start=5.0,
            attempted_end=15.0,
        )
        msg = info.describe()
        assert "drone" in msg
        assert "d1" in msg

    def test_describe_incluye_intervalos(self):
        existing = Reservation(start=0.0, end=10.0, owner="a")
        info = CollisionInfo(
            resource=("drone", "d1"),
            existing=existing,
            attempted_start=5.0,
            attempted_end=15.0,
        )
        msg = info.describe()
        # No nos atamos al formato exacto; sí a que los datos aparezcan.
        assert "0.0" in msg
        assert "10.0" in msg
        assert "5.0" in msg
        assert "15.0" in msg

    def test_describe_incluye_owner(self):
        existing = Reservation(start=0.0, end=10.0, owner="cmd_x")
        info = CollisionInfo(
            resource=("package", "p1"),
            existing=existing,
            attempted_start=2.0,
            attempted_end=7.0,
        )
        assert "cmd_x" in info.describe()


# ===========================================================================
# Reservation: dataclass inmutable
# ===========================================================================
class TestReservation:
    def test_construccion_y_atributos(self):
        r = Reservation(start=0.0, end=10.0, owner="x")
        assert r.start == 0.0
        assert r.end == 10.0
        assert r.owner == "x"

    def test_inmutable(self):
        from dataclasses import FrozenInstanceError
        r = Reservation(start=0.0, end=10.0, owner="x")
        with pytest.raises(FrozenInstanceError):
            r.start = 99.0  # type: ignore[misc]

    def test_igualdad_estructural(self):
        r1 = Reservation(start=0.0, end=10.0, owner="x")
        r2 = Reservation(start=0.0, end=10.0, owner="x")
        assert r1 == r2


# ===========================================================================
# Tests de las CINCO REGLAS del PDF Parte 3
#
# Estos tests son la materialización directa del enunciado. Cada uno
# replica una de las cinco reglas usando ResourceTable + resources_of
# como bloques de Lego: reserva un Command, intenta un segundo Command
# que viola la regla, verifica que check() detecta la colisión.
#
# Solo se ejercitan recursos+intervalos; la integración con el runner
# y la transición del drone a ERROR llega en pasos posteriores.
# ===========================================================================
class TestReglaPDF1UnDronUnaAccion:
    """REGLA 1: 'Cada dron solo puede realizar una acción al mismo tiempo.'"""

    def test_dos_moves_mismo_dron_solapados_chocan(self):
        t = ResourceTable()
        m1 = Move(drone_id="d1", destination_id="x", duration=10.0)
        t.reserve(resources_of(m1), 0.0, 10.0, owner=m1.command_id)
        m2 = Move(drone_id="d1", destination_id="y", duration=10.0)
        result = t.check(resources_of(m2), 5.0, 15.0)
        assert result is not None
        assert result.resource == ("drone", "d1")

    def test_dos_acciones_distintas_mismo_dron_solapadas_chocan(self):
        """Move + PickUp del mismo dron solapadas. La regla es del DRON,
        no del tipo de acción."""
        t = ResourceTable()
        m = Move(drone_id="d1", destination_id="x", duration=10.0)
        t.reserve(resources_of(m), 0.0, 10.0, owner=m.command_id)
        p = PickUp(
            drone_id="d1", arm_id="izq", package_id="p1", duration=5.0
        )
        result = t.check(resources_of(p), 3.0, 8.0)
        assert result is not None
        assert result.resource == ("drone", "d1")

    def test_dos_drones_distintos_pueden_actuar_en_paralelo(self):
        """REGLA NEGATIVA: la regla 1 no impide drones distintos."""
        t = ResourceTable()
        m1 = Move(drone_id="d1", destination_id="x", duration=10.0)
        t.reserve(resources_of(m1), 0.0, 10.0, owner=m1.command_id)
        m2 = Move(drone_id="d2", destination_id="x", duration=10.0)
        result = t.check(resources_of(m2), 0.0, 10.0)
        assert result is None


class TestReglaPDF2CajaExclusiva:
    """REGLA 2 (cajas): 'Una misma caja [...] solo puede ser cogida
    por un dron.'"""

    def test_dos_pickup_misma_caja_distintos_drones_chocan_en_paquete(self):
        t = ResourceTable()
        p1 = PickUp(
            drone_id="d1", arm_id="izq", package_id="caja", duration=5.0
        )
        t.reserve(resources_of(p1), 0.0, 5.0, owner=p1.command_id)
        p2 = PickUp(
            drone_id="d2", arm_id="izq", package_id="caja", duration=5.0
        )
        result = t.check(resources_of(p2), 2.0, 7.0)
        assert result is not None
        assert result.resource == ("package", "caja")

    def test_deliver_y_load_misma_caja_chocan_en_paquete(self):
        """Una caja que está siendo entregada por d1 no puede ser
        simultáneamente cargada en transporter por d2."""
        t = ResourceTable()
        d = Deliver(
            drone_id="d1", package_id="caja", person_id="ana", duration=5.0
        )
        t.reserve(resources_of(d), 0.0, 5.0, owner=d.command_id)
        l = LoadIntoTransporter(
            drone_id="d2", package_id="caja", transporter_id="t1",
            duration=5.0,
        )
        result = t.check(resources_of(l), 2.0, 7.0)
        assert result is not None
        assert result.resource == ("package", "caja")

    def test_cajas_distintas_no_chocan(self):
        t = ResourceTable()
        p1 = PickUp(
            drone_id="d1", arm_id="izq", package_id="caja_a", duration=5.0
        )
        t.reserve(resources_of(p1), 0.0, 5.0, owner=p1.command_id)
        p2 = PickUp(
            drone_id="d2", arm_id="izq", package_id="caja_b", duration=5.0
        )
        result = t.check(resources_of(p2), 0.0, 5.0)
        assert result is None


class TestReglaPDF2TransportadorExclusivo:
    """REGLA 2 (transportadores): 'Un mismo transportador solo puede
    ser cogido por un dron.'"""

    def test_dos_move_with_transporter_mismo_transporter_chocan(self):
        t = ResourceTable()
        m1 = MoveWithTransporter(
            drone_id="d1", transporter_id="t1",
            destination_id="x", duration=10.0,
        )
        t.reserve(resources_of(m1), 0.0, 10.0, owner=m1.command_id)
        m2 = MoveWithTransporter(
            drone_id="d2", transporter_id="t1",
            destination_id="y", duration=10.0,
        )
        result = t.check(resources_of(m2), 5.0, 15.0)
        assert result is not None
        assert result.resource == ("transporter", "t1")

    def test_transportadores_distintos_no_chocan(self):
        t = ResourceTable()
        m1 = MoveWithTransporter(
            drone_id="d1", transporter_id="t1",
            destination_id="x", duration=10.0,
        )
        t.reserve(resources_of(m1), 0.0, 10.0, owner=m1.command_id)
        m2 = MoveWithTransporter(
            drone_id="d2", transporter_id="t2",
            destination_id="x", duration=10.0,
        )
        result = t.check(resources_of(m2), 0.0, 10.0)
        assert result is None


class TestReglaPDF3LoadUnloadMutex:
    """REGLA 3: 'Mientras un dron mete o saca una caja de un
    transportador, ningún otro dron puede hacerlo en paralelo, ni
    tampoco coger dicho transportador.'

    Caso particular reforzado de la regla 2: las acciones Load/Unload
    también ocupan el transportador (que es lo que resources_of ya
    codifica), de modo que esta regla se cumple por el mismo
    mecanismo."""

    def test_load_y_load_mismo_transporter_distintos_drones_chocan(self):
        t = ResourceTable()
        l1 = LoadIntoTransporter(
            drone_id="d1", package_id="cajaA",
            transporter_id="t1", duration=5.0,
        )
        t.reserve(resources_of(l1), 0.0, 5.0, owner=l1.command_id)
        l2 = LoadIntoTransporter(
            drone_id="d2", package_id="cajaB",
            transporter_id="t1", duration=5.0,
        )
        result = t.check(resources_of(l2), 2.0, 7.0)
        assert result is not None
        assert result.resource == ("transporter", "t1")

    def test_load_y_unload_mismo_transporter_chocan(self):
        t = ResourceTable()
        l = LoadIntoTransporter(
            drone_id="d1", package_id="cajaA",
            transporter_id="t1", duration=5.0,
        )
        t.reserve(resources_of(l), 0.0, 5.0, owner=l.command_id)
        u = UnloadFromTransporter(
            drone_id="d2", arm_id="izq",
            package_id="cajaB", transporter_id="t1",
            duration=5.0,
        )
        result = t.check(resources_of(u), 2.0, 7.0)
        assert result is not None
        assert result.resource == ("transporter", "t1")

    def test_load_y_move_with_transporter_mismo_transporter_chocan(self):
        """REGLA 3 expansión: si d1 está cargando t1, d2 NO puede
        cogerlo para volar con él."""
        t = ResourceTable()
        l = LoadIntoTransporter(
            drone_id="d1", package_id="caja",
            transporter_id="t1", duration=5.0,
        )
        t.reserve(resources_of(l), 0.0, 5.0, owner=l.command_id)
        m = MoveWithTransporter(
            drone_id="d2", transporter_id="t1",
            destination_id="x", duration=10.0,
        )
        result = t.check(resources_of(m), 2.0, 12.0)
        assert result is not None
        assert result.resource == ("transporter", "t1")


class TestReglaPDF4PersonaExclusiva:
    """REGLA 4: 'Una persona solo puede recibir una entrega de un dron
    al mismo tiempo.'"""

    def test_dos_deliver_misma_persona_chocan_en_persona(self):
        """Aunque sean paquetes distintos, la persona está ocupada."""
        t = ResourceTable()
        d1 = Deliver(
            drone_id="d1", package_id="cajaA", person_id="ana", duration=5.0
        )
        t.reserve(resources_of(d1), 0.0, 5.0, owner=d1.command_id)
        d2 = Deliver(
            drone_id="d2", package_id="cajaB", person_id="ana", duration=5.0
        )
        result = t.check(resources_of(d2), 2.0, 7.0)
        assert result is not None
        assert result.resource == ("person", "ana")

    def test_deliver_personas_distintas_no_chocan(self):
        t = ResourceTable()
        d1 = Deliver(
            drone_id="d1", package_id="cajaA", person_id="ana", duration=5.0
        )
        t.reserve(resources_of(d1), 0.0, 5.0, owner=d1.command_id)
        d2 = Deliver(
            drone_id="d2", package_id="cajaB", person_id="bob", duration=5.0
        )
        result = t.check(resources_of(d2), 0.0, 5.0)
        assert result is None


# ===========================================================================
# Patrón runtime: check -> reserve (workflow esperado)
# ===========================================================================
class TestPatronCheckReserve:
    """Documenta el flujo que el runner seguirá."""

    def test_workflow_completo_dos_commands_compatibles(self):
        """Dos Commands en drones distintos, intervalos solapados.
        Pattern: check (None) -> reserve -> check otro (None) -> reserve."""
        t = ResourceTable()
        c1 = Move(drone_id="d1", destination_id="x", duration=10.0)
        c2 = Move(drone_id="d2", destination_id="y", duration=10.0)

        # Programar c1 en [0,10): no hay colisión, reservar.
        assert t.check(resources_of(c1), 0.0, 10.0) is None
        t.reserve(resources_of(c1), 0.0, 10.0, owner=c1.command_id)

        # Programar c2 en [0,10): no hay colisión, reservar.
        assert t.check(resources_of(c2), 0.0, 10.0) is None
        t.reserve(resources_of(c2), 0.0, 10.0, owner=c2.command_id)

        assert len(t) == 2

    def test_workflow_segundo_command_choca_y_no_reserva(self):
        """Primer Command pasa; segundo choca; el caller (runner) no
        debe llamar a reserve(). La tabla queda con una sola reserva."""
        t = ResourceTable()
        c1 = Move(drone_id="d1", destination_id="x", duration=10.0)
        c2 = Move(drone_id="d1", destination_id="y", duration=10.0)

        assert t.check(resources_of(c1), 0.0, 10.0) is None
        t.reserve(resources_of(c1), 0.0, 10.0, owner=c1.command_id)

        collision = t.check(resources_of(c2), 5.0, 15.0)
        assert collision is not None
        # NO llamamos a reserve; simulamos el comportamiento del runner.

        assert len(t) == 1


# ===========================================================================
# Resource es la tupla esperada
# ===========================================================================
class TestResourceTipo:
    """Verifica que el alias Resource se comporta como tuple[str, str]."""

    def test_resource_es_alias_de_tupla(self):
        r: Resource = ("drone", "d1")
        assert r == ("drone", "d1")
        assert isinstance(r, tuple)

    def test_recursos_son_hashables(self):
        """Crítico: se usan como claves de dict en ResourceTable."""
        r: Resource = ("drone", "d1")
        d = {r: "valor"}
        assert d[("drone", "d1")] == "valor"
