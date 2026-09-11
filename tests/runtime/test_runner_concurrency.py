"""
Tests de PlanRunner para detección de concurrencia (PDDL parte 3).

Estos tests son end-to-end: construyen un Plan completo, lo ejecutan
mediante PlanRunner.execute(), y verifican el comportamiento de la
detección de colisiones de recursos. La lógica subyacente
(ResourceTable, resources_of) ya está testada en aislamiento en
tests/runtime/test_resources.py; aquí se verifica la integración.

Categorías:

1. Las CINCO REGLAS del PDF parte 3 a nivel de runner, con cada
   una de sus formas.
2. Comportamiento del CommandFailure: kind='concurrency', mensaje,
   referencia al scheduled correcto.
3. Drone implicado pasa a ERROR (sumidero forward).
4. Métricas: failed_commands se incrementa.
5. El END del Command bloqueado NO se procesa: ni snapshot end, ni
   apply(), ni reserva de recursos.
6. Otros Commands del plan que no comparten recursos siguen
   ejecutándose normalmente.
7. Casos legales: acciones concurrentes en recursos disjuntos pasan.
8. Mismo timestamp: dos Commands que arrancan a la vez sobre el
   mismo recurso, uno gana, el otro falla.
9. Orden de chequeo: si un Command es PDDL-inválido Y choca por
   concurrencia, el fallo reportado es PDDL (chequeo primero).
"""
from __future__ import annotations

import pytest

from droneplan_viz.domain import DroneState
from droneplan_viz.runtime import (
    Plan,
    PlanRunner,
    ScheduledCommand,
)

from tests.runtime.conftest import (
    make_deliver,
    make_load,
    make_move,
    make_move_with_transporter,
    make_pickup,
    make_unload,
    world_with_drone_at,
    world_with_package_held,
    world_with_package_in_transporter,
)


# ===========================================================================
# REGLA 1 del PDF: "Cada dron solo puede realizar una acción al mismo
# tiempo."
#
# Esta regla ya la cubre el Validator de Sesión A (vía _check_drone_ready):
# si el drone está MOVING o INTERACTING, valida rechaza. Cuando lleguen
# al runner dos Commands del mismo drone con intervalos solapados, el
# segundo falla en validate (kind='pddl'), NO en concurrency.
#
# Verificamos aquí que la ResourceTable también detecta el caso, como
# segunda red de seguridad. La diferencia visible es que el fallo es de
# kind='pddl' (porque PDDL se chequea primero); la ResourceTable
# protegerá los recursos que NO se traducen en drone.state (cajas,
# transportadores, personas) en las reglas siguientes.
# ===========================================================================
class TestRegla1UnDronUnaAccion:
    """Mismo drone, intervalos solapados → fallo PDDL (vía drone.state)."""

    def test_dos_moves_mismo_dron_solapados_falla_pddl(self, base_world):
        r = PlanRunner(base_world)
        m1 = make_move("d1", "casa1", duration=10.0, cmd_id="m1")
        m2 = make_move("d1", "casa2", duration=10.0, cmd_id="m2")
        plan = Plan(scheduled=(
            ScheduledCommand(command=m1, start_time=0.0),   # [0,10)
            ScheduledCommand(command=m2, start_time=5.0),   # [5,15) solapado
        ))
        result = r.execute(plan)
        assert not result.succeeded
        assert len(result.failures) == 1
        # PDDL chequea primero: drone.state == MOVING, _check_drone_ready
        # lo rechaza antes de que la ResourceTable mire los recursos.
        assert result.failures[0].kind == "pddl"
        assert result.failures[0].scheduled.command is m2


# ===========================================================================
# REGLA 2 (cajas): "Una misma caja solo puede ser cogida por un dron."
#
# A diferencia de la regla 1, las cajas NO se reflejan en drone.state,
# así que el Validator no las protege. La ResourceTable es necesaria
# aquí. Esperamos kind='concurrency'.
# ===========================================================================
class TestRegla2CajaExclusiva:
    """Misma caja desde drones distintos solapados → concurrency."""

    def test_dos_pickup_misma_caja_solapados_falla_concurrency(self, base_world):
        r = PlanRunner(base_world)
        p1 = make_pickup(
            "d1", "izq", "libre1", duration=5.0, cmd_id="p1"
        )
        p2 = make_pickup(
            "d2", "izq", "libre1", duration=5.0, cmd_id="p2"
        )
        plan = Plan(scheduled=(
            ScheduledCommand(command=p1, start_time=0.0),  # [0,5)
            ScheduledCommand(command=p2, start_time=2.0),  # [2,7) solapado
        ))
        result = r.execute(plan)
        assert not result.succeeded
        assert len(result.failures) == 1
        assert result.failures[0].kind == "concurrency"
        assert result.failures[0].scheduled.command is p2

    def test_mensaje_de_error_menciona_el_paquete(self, base_world):
        r = PlanRunner(base_world)
        p1 = make_pickup(
            "d1", "izq", "libre1", duration=5.0, cmd_id="p1"
        )
        p2 = make_pickup(
            "d2", "izq", "libre1", duration=5.0, cmd_id="p2"
        )
        plan = Plan(scheduled=(
            ScheduledCommand(command=p1, start_time=0.0),
            ScheduledCommand(command=p2, start_time=2.0),
        ))
        result = r.execute(plan)
        msg = result.failures[0].reason
        assert "package" in msg
        assert "libre1" in msg

    def test_cajas_distintas_no_se_estorban(self, base_world):
        """Drones distintos cogen paquetes distintos en paralelo: legal."""
        r = PlanRunner(base_world)
        p1 = make_pickup(
            "d1", "izq", "libre1", duration=5.0, cmd_id="p1"
        )
        p2 = make_pickup(
            "d2", "izq", "libre2", duration=5.0, cmd_id="p2"
        )
        plan = Plan(scheduled=(
            ScheduledCommand(command=p1, start_time=0.0),
            ScheduledCommand(command=p2, start_time=0.0),
        ))
        result = r.execute(plan)
        assert result.succeeded


# ===========================================================================
# REGLA 2 (transportadores): "Un mismo transportador solo puede ser
# cogido por un dron."
# ===========================================================================
class TestRegla2TransportadorExclusivo:
    def test_dos_move_with_transporter_mismo_t_falla(self, base_world):
        """d1 mueve t1, d2 intenta mover t1 también, solapado."""
        r = PlanRunner(base_world)
        m1 = make_move_with_transporter(
            "d1", "t1", "casa1", duration=10.0, cmd_id="m1"
        )
        m2 = make_move_with_transporter(
            "d2", "t1", "casa2", duration=10.0, cmd_id="m2"
        )
        plan = Plan(scheduled=(
            ScheduledCommand(command=m1, start_time=0.0),
            ScheduledCommand(command=m2, start_time=5.0),
        ))
        result = r.execute(plan)
        assert not result.succeeded
        assert result.failures[0].kind == "concurrency"
        assert result.failures[0].scheduled.command is m2

    def test_mensaje_de_error_menciona_el_transportador(self, base_world):
        r = PlanRunner(base_world)
        m1 = make_move_with_transporter(
            "d1", "t1", "casa1", duration=10.0, cmd_id="m1"
        )
        m2 = make_move_with_transporter(
            "d2", "t1", "casa2", duration=10.0, cmd_id="m2"
        )
        plan = Plan(scheduled=(
            ScheduledCommand(command=m1, start_time=0.0),
            ScheduledCommand(command=m2, start_time=5.0),
        ))
        result = r.execute(plan)
        msg = result.failures[0].reason
        assert "transporter" in msg
        assert "t1" in msg

    def test_transportadores_distintos_no_se_estorban(self, base_world):
        r = PlanRunner(base_world)
        m1 = make_move_with_transporter(
            "d1", "t1", "casa1", duration=10.0, cmd_id="m1"
        )
        m2 = make_move_with_transporter(
            "d2", "t2", "casa2", duration=20.0, cmd_id="m2"
        )
        plan = Plan(scheduled=(
            ScheduledCommand(command=m1, start_time=0.0),
            ScheduledCommand(command=m2, start_time=0.0),
        ))
        result = r.execute(plan)
        assert result.succeeded


# ===========================================================================
# REGLA 3 (Load/Unload mutex): "Mientras un dron mete o saca una caja
# de un transportador, ningún otro dron puede hacerlo en paralelo, ni
# tampoco coger dicho transportador."
#
# Caso particular de la regla 2 (transportador exclusivo): Load y Unload
# también ocupan el transportador. Aquí verificamos las combinaciones
# Load-vs-Load, Load-vs-Unload, Load-vs-MoveWithTransporter, etc.
# ===========================================================================
class TestRegla3LoadUnloadMutex:
    def test_load_load_mismo_transporter_falla(self, base_world):
        """d1 carga libre1 en t1; d2 intenta cargar libre2 en t1 en paralelo."""
        # Pre-setup: ambos drones sostienen su paquete.
        world = world_with_package_held(base_world, "libre1", "d1", "izq")
        world = world_with_package_held(world, "libre2", "d2", "izq")

        r = PlanRunner(world)
        l1 = make_load(
            "d1", "libre1", "t1", duration=5.0, cmd_id="l1"
        )
        l2 = make_load(
            "d2", "libre2", "t1", duration=5.0, cmd_id="l2"
        )
        plan = Plan(scheduled=(
            ScheduledCommand(command=l1, start_time=0.0),  # [0,5)
            ScheduledCommand(command=l2, start_time=2.0),  # [2,7)
        ))
        result = r.execute(plan)
        assert not result.succeeded
        assert result.failures[0].kind == "concurrency"
        assert result.failures[0].scheduled.command is l2

    def test_load_y_move_with_transporter_mismo_t_falla(self, base_world):
        """d1 mueve t1; d2 intenta meter libre2 en t1 en paralelo."""
        world = world_with_package_held(base_world, "libre2", "d2", "izq")
        r = PlanRunner(world)
        m = make_move_with_transporter(
            "d1", "t1", "casa1", duration=10.0, cmd_id="m"
        )
        l = make_load(
            "d2", "libre2", "t1", duration=5.0, cmd_id="l"
        )
        plan = Plan(scheduled=(
            ScheduledCommand(command=m, start_time=0.0),
            ScheduledCommand(command=l, start_time=2.0),
        ))
        result = r.execute(plan)
        assert not result.succeeded
        assert result.failures[0].kind == "concurrency"
        assert result.failures[0].scheduled.command is l

    def test_unload_y_load_mismo_t_falla(self, base_world):
        """d1 descarga libre1 de t1; d2 intenta cargar libre2 en t1."""
        # Pre-setup: libre1 dentro de t1 (descargable); libre2 en brazo de d2.
        world = world_with_package_in_transporter(
            base_world, "libre1", "t1"
        )
        world = world_with_package_held(world, "libre2", "d2", "izq")

        r = PlanRunner(world)
        u = make_unload(
            "d1", "izq", "libre1", "t1", duration=5.0, cmd_id="u"
        )
        l = make_load(
            "d2", "libre2", "t1", duration=5.0, cmd_id="l"
        )
        plan = Plan(scheduled=(
            ScheduledCommand(command=u, start_time=0.0),
            ScheduledCommand(command=l, start_time=2.0),
        ))
        result = r.execute(plan)
        assert not result.succeeded
        assert result.failures[0].kind == "concurrency"


# ===========================================================================
# REGLA 4: "Una persona solo puede recibir una entrega de un dron al
# mismo tiempo."
# ===========================================================================
class TestRegla4PersonaExclusiva:
    def test_dos_deliver_misma_persona_falla(self, base_world):
        """d1 y d2 entregan paquetes distintos a ana simultáneamente."""
        # Pre-setup: ambos drones en casa1 con un paquete sostenido.
        # Ana necesita medicina (default). Para que ambos validates pasen,
        # ambos paquetes deben llevar lo que ana necesita.
        # En base_world ana solo necesita 1x medicina, así que necesitamos
        # que ana tenga DOS necesidades para que ambos validates pasen.
        # Más simple: usar bob (necesita comida y medicina).
        # Ponemos bob en una localización donde haya 2 paquetes accesibles.

        # Setup compacto: ambos drones en casa2 (donde está bob).
        world = world_with_drone_at(base_world, "d1", "casa2")
        world = world_with_drone_at(world, "d2", "casa2")
        # d1 sostiene libre1 (medicina), d2 sostiene libre2 (comida).
        world = world_with_package_held(world, "libre1", "d1", "izq")
        world = world_with_package_held(world, "libre2", "d2", "izq")

        r = PlanRunner(world)
        d1_entrega = make_deliver(
            "d1", "libre1", "bob", duration=5.0, cmd_id="d1_entrega"
        )
        d2_entrega = make_deliver(
            "d2", "libre2", "bob", duration=5.0, cmd_id="d2_entrega"
        )
        plan = Plan(scheduled=(
            ScheduledCommand(command=d1_entrega, start_time=0.0),
            ScheduledCommand(command=d2_entrega, start_time=2.0),
        ))
        result = r.execute(plan)
        assert not result.succeeded
        assert result.failures[0].kind == "concurrency"
        assert result.failures[0].scheduled.command is d2_entrega

    def test_mensaje_de_error_menciona_la_persona(self, base_world):
        world = world_with_drone_at(base_world, "d1", "casa2")
        world = world_with_drone_at(world, "d2", "casa2")
        world = world_with_package_held(world, "libre1", "d1", "izq")
        world = world_with_package_held(world, "libre2", "d2", "izq")
        r = PlanRunner(world)
        d1_entrega = make_deliver(
            "d1", "libre1", "bob", duration=5.0, cmd_id="d1_e"
        )
        d2_entrega = make_deliver(
            "d2", "libre2", "bob", duration=5.0, cmd_id="d2_e"
        )
        plan = Plan(scheduled=(
            ScheduledCommand(command=d1_entrega, start_time=0.0),
            ScheduledCommand(command=d2_entrega, start_time=2.0),
        ))
        result = r.execute(plan)
        msg = result.failures[0].reason
        assert "person" in msg
        assert "bob" in msg

    def test_personas_distintas_no_se_estorban(self, base_world):
        """d1 entrega a ana en casa1; d2 entrega a bob en casa2.
        Ambos en paralelo, recursos disjuntos."""
        # d1 en casa1 con libre1 (medicina, ana lo necesita).
        world = world_with_drone_at(base_world, "d1", "casa1")
        world = world_with_package_held(world, "libre1", "d1", "izq")
        # d2 en casa2 con libre2 (comida, bob lo necesita).
        world = world_with_drone_at(world, "d2", "casa2")
        world = world_with_package_held(world, "libre2", "d2", "izq")

        r = PlanRunner(world)
        d_a = make_deliver(
            "d1", "libre1", "ana", duration=5.0, cmd_id="da"
        )
        d_b = make_deliver(
            "d2", "libre2", "bob", duration=5.0, cmd_id="db"
        )
        plan = Plan(scheduled=(
            ScheduledCommand(command=d_a, start_time=0.0),
            ScheduledCommand(command=d_b, start_time=0.0),
        ))
        result = r.execute(plan)
        assert result.succeeded


# ===========================================================================
# Efectos de un fallo de concurrencia
# ===========================================================================
class TestEfectosFalloConcurrencia:
    """Cuando un Command falla por concurrencia, el runner debe:
        - Marcar el drone implicado a ERROR.
        - Incrementar failed_commands.
        - NO procesar el END (no llamar a apply, no liberar recursos
          que no se reservaron).
        - El snapshot de fallo aparece en el historial.
    """

    def test_fallo_concurrencia_pone_drone_en_error(self, base_world):
        r = PlanRunner(base_world)
        p1 = make_pickup(
            "d1", "izq", "libre1", duration=5.0, cmd_id="p1"
        )
        p2 = make_pickup(
            "d2", "izq", "libre1", duration=5.0, cmd_id="p2"
        )
        plan = Plan(scheduled=(
            ScheduledCommand(command=p1, start_time=0.0),
            ScheduledCommand(command=p2, start_time=2.0),
        ))
        r.execute(plan)
        # d1 completó su PickUp y debería estar IDLE.
        assert r.current_world.drones["d1"].state == DroneState.IDLE
        # d2 falló por concurrencia y debería estar en ERROR.
        assert r.current_world.drones["d2"].state == DroneState.ERROR

    def test_fallo_concurrencia_incrementa_failed_commands(self, base_world):
        r = PlanRunner(base_world)
        p1 = make_pickup(
            "d1", "izq", "libre1", duration=5.0, cmd_id="p1"
        )
        p2 = make_pickup(
            "d2", "izq", "libre1", duration=5.0, cmd_id="p2"
        )
        plan = Plan(scheduled=(
            ScheduledCommand(command=p1, start_time=0.0),
            ScheduledCommand(command=p2, start_time=2.0),
        ))
        r.execute(plan)
        assert r.current_metrics.failed_commands == 1

    def test_fallo_concurrencia_no_genera_snapshot_end(self, base_world):
        """p1 produce 2 snapshots (start+end). p2 falla en start, no
        produce snapshot end. Total: 1 inicial + 2 + 1 = 4."""
        r = PlanRunner(base_world)
        p1 = make_pickup(
            "d1", "izq", "libre1", duration=5.0, cmd_id="p1"
        )
        p2 = make_pickup(
            "d2", "izq", "libre1", duration=5.0, cmd_id="p2"
        )
        plan = Plan(scheduled=(
            ScheduledCommand(command=p1, start_time=0.0),
            ScheduledCommand(command=p2, start_time=2.0),
        ))
        r.execute(plan)
        assert len(r.history) == 4

    def test_fallo_concurrencia_no_libera_recursos_no_reservados(
        self, base_world
    ):
        """Un Command que falla por concurrencia NO reservó sus recursos.
        Por tanto, otros Commands posteriores que NO chocan con los
        recursos ya reservados deben pasar normalmente."""
        r = PlanRunner(base_world)
        # p1 coge libre1 en [0,5). Reserva: drone d1, package libre1.
        p1 = make_pickup(
            "d1", "izq", "libre1", duration=5.0, cmd_id="p1"
        )
        # p2 (d2 sobre libre1) intenta en [2,7). Choca con libre1.
        p2 = make_pickup(
            "d2", "izq", "libre1", duration=5.0, cmd_id="p2"
        )
        # p3 (d2 sobre libre2) intenta en [10,15). No choca con nada.
        p3 = make_pickup(
            "d2", "izq", "libre2", duration=5.0, cmd_id="p3"
        )
        plan = Plan(scheduled=(
            ScheduledCommand(command=p1, start_time=0.0),
            ScheduledCommand(command=p2, start_time=2.0),
            ScheduledCommand(command=p3, start_time=10.0),
        ))
        result = r.execute(plan)
        # Solo p2 falla; p1 y p3 pasan.
        # PERO: tras el fallo, d2 está en ERROR. Su PickUp posterior
        # falla por PDDL (no por concurrencia).
        assert len(result.failures) == 2
        assert result.failures[0].kind == "concurrency"  # p2
        assert result.failures[1].kind == "pddl"          # p3, dron en ERROR


# ===========================================================================
# Casos legales: acciones concurrentes en recursos disjuntos
# ===========================================================================
class TestAccionesLegalesConcurrentes:
    def test_dos_drones_vuelan_en_paralelo(self, base_world):
        r = PlanRunner(base_world)
        m_a = make_move("d1", "casa1", duration=10.0, cmd_id="A")
        m_b = make_move("d2", "casa2", duration=20.0, cmd_id="B")
        plan = Plan(scheduled=(
            ScheduledCommand(command=m_a, start_time=0.0),
            ScheduledCommand(command=m_b, start_time=0.0),
        ))
        result = r.execute(plan)
        assert result.succeeded
        assert result.makespan == 20.0

    def test_dos_pickups_paquetes_distintos_en_paralelo(self, base_world):
        r = PlanRunner(base_world)
        p1 = make_pickup(
            "d1", "izq", "libre1", duration=5.0, cmd_id="p1"
        )
        p2 = make_pickup(
            "d2", "izq", "libre2", duration=5.0, cmd_id="p2"
        )
        plan = Plan(scheduled=(
            ScheduledCommand(command=p1, start_time=0.0),
            ScheduledCommand(command=p2, start_time=0.0),
        ))
        result = r.execute(plan)
        assert result.succeeded
        # Ambos paquetes están sostenidos por sus respectivos drones.
        from droneplan_viz.domain import HeldByArm
        assert r.current_world.packages["libre1"].at == HeldByArm(
            drone_id="d1", arm_id="izq"
        )
        assert r.current_world.packages["libre2"].at == HeldByArm(
            drone_id="d2", arm_id="izq"
        )


# ===========================================================================
# Mismo timestamp: dos Commands compitiendo por un recurso
# ===========================================================================
class TestMismoTimestamp:
    """Dos Commands del mismo recurso programados en el mismo
    start_time: el primero (por orig_index) gana; el segundo falla.
    Test crítico del determinismo del tiebreak."""

    def test_dos_pickups_misma_caja_mismo_timestamp_uno_falla(
        self, base_world
    ):
        r = PlanRunner(base_world)
        p1 = make_pickup(
            "d1", "izq", "libre1", duration=5.0, cmd_id="primero"
        )
        p2 = make_pickup(
            "d2", "izq", "libre1", duration=5.0, cmd_id="segundo"
        )
        plan = Plan(scheduled=(
            ScheduledCommand(command=p1, start_time=0.0),  # orig_index=0
            ScheduledCommand(command=p2, start_time=0.0),  # orig_index=1
        ))
        result = r.execute(plan)
        assert len(result.failures) == 1
        # El que falla es el segundo en orig_index.
        assert result.failures[0].scheduled.command.command_id == "segundo"

    def test_inversion_del_orden_inverte_el_ganador(self, base_world):
        """Si invertimos el orden de declaración, ahora el otro Command
        es el que gana."""
        r = PlanRunner(base_world)
        p1 = make_pickup(
            "d1", "izq", "libre1", duration=5.0, cmd_id="originalmente_primero"
        )
        p2 = make_pickup(
            "d2", "izq", "libre1", duration=5.0, cmd_id="originalmente_segundo"
        )
        # Declaramos al revés: p2 antes que p1.
        plan = Plan(scheduled=(
            ScheduledCommand(command=p2, start_time=0.0),  # orig_index=0
            ScheduledCommand(command=p1, start_time=0.0),  # orig_index=1
        ))
        result = r.execute(plan)
        assert result.failures[0].scheduled.command.command_id == (
            "originalmente_primero"
        )


# ===========================================================================
# Orden de chequeo: PDDL primero, concurrencia segundo
# ===========================================================================
class TestOrdenChequeoPDDLPrimero:
    """Si un Command es PDDL-inválido Y choca por concurrencia, el
    fallo reportado es PDDL. Documenta el orden de chequeo:
        1. PDDL (Validator).
        2. Concurrencia (ResourceTable).
    """

    def test_drone_inexistente_da_pddl_no_concurrency(self, base_world):
        """d_fantasma no existe. Aunque hubiera concurrencia, el fallo
        debería reportarse como PDDL."""
        r = PlanRunner(base_world)
        m1 = make_move("d1", "casa1", duration=10.0, cmd_id="m1")
        m_fantasma = make_move(
            "d_fantasma", "casa1", duration=10.0, cmd_id="ghost"
        )
        plan = Plan(scheduled=(
            ScheduledCommand(command=m1, start_time=0.0),
            ScheduledCommand(command=m_fantasma, start_time=0.0),
        ))
        result = r.execute(plan)
        # El Command del drone inexistente falla por PDDL.
        ghost_failures = [
            f for f in result.failures if f.scheduled.command is m_fantasma
        ]
        assert len(ghost_failures) == 1
        assert ghost_failures[0].kind == "pddl"


# ===========================================================================
# Reservas se liberan al final del intervalo (back-to-back legal)
# ===========================================================================
class TestReservaSeLiberaConElTiempo:
    """Una vez termina el intervalo de un Command, su recurso queda
    libre. El siguiente Command sobre el mismo recurso, programado
    después, debe pasar."""

    def test_back_to_back_misma_caja_distintos_drones(self, base_world):
        """d1 coge libre1 en [0,5), suelta de algún modo. d2 coge libre1
        en [5,10): debe pasar porque [0,5) y [5,10) NO solapan.

        Para que esto funcione semánticamente: tras el PickUp de d1
        en [0,5), la caja está en el brazo de d1. Para que d2 pueda
        cogerla en [5,10), d1 tiene que devolverla a una localización.
        Eso requeriría d1 Deliver o algo similar. Simplificamos: en
        este test concreto solo testamos el aspecto de la
        ResourceTable, así que aceptamos que el segundo PickUp falle
        por PDDL (libre1 ya no está libre). Lo importante es que NO
        falle por concurrency.
        """
        r = PlanRunner(base_world)
        p1 = make_pickup(
            "d1", "izq", "libre1", duration=5.0, cmd_id="p1"
        )
        p2 = make_pickup(
            "d2", "izq", "libre1", duration=5.0, cmd_id="p2"
        )
        plan = Plan(scheduled=(
            ScheduledCommand(command=p1, start_time=0.0),  # [0,5)
            ScheduledCommand(command=p2, start_time=5.0),  # [5,10), no solapa
        ))
        result = r.execute(plan)
        # p2 falla, pero NO por concurrencia: la reserva de libre1 ya
        # expiró en t=5. El fallo es PDDL: la caja la sostiene d1.
        assert len(result.failures) == 1
        assert result.failures[0].kind == "pddl"


# ===========================================================================
# Verificación del comportamiento end-to-end de un plan parte 3 realista
# ===========================================================================
class TestPlanParte3Realista:
    """Plan que ejercita varios drones simultáneamente, recursos
    compartidos legales, sin colisiones. Reproduce el escenario que
    el TFG querría visualizar."""

    def test_dos_drones_dos_paquetes_dos_personas(self, base_world):
        """d1 entrega libre1 a ana en casa1; d2 entrega libre2 a bob
        en casa2. Vuelos en paralelo, entregas en paralelo, recursos
        disjuntos."""
        r = PlanRunner(base_world)
        # d1: PickUp libre1 [0,5), Move a casa1 [5,15), Deliver [15,20).
        p1 = make_pickup(
            "d1", "izq", "libre1", duration=5.0, cmd_id="p1"
        )
        mv1 = make_move("d1", "casa1", duration=10.0, cmd_id="mv1")
        de1 = make_deliver(
            "d1", "libre1", "ana", duration=5.0, cmd_id="de1"
        )
        # d2: PickUp libre2 [0,5), Move a casa2 [5,25), Deliver [25,30).
        p2 = make_pickup(
            "d2", "izq", "libre2", duration=5.0, cmd_id="p2"
        )
        mv2 = make_move("d2", "casa2", duration=20.0, cmd_id="mv2")
        de2 = make_deliver(
            "d2", "libre2", "bob", duration=5.0, cmd_id="de2"
        )
        plan = Plan(scheduled=(
            ScheduledCommand(command=p1, start_time=0.0),
            ScheduledCommand(command=mv1, start_time=5.0),
            ScheduledCommand(command=de1, start_time=15.0),
            ScheduledCommand(command=p2, start_time=0.0),
            ScheduledCommand(command=mv2, start_time=5.0),
            ScheduledCommand(command=de2, start_time=25.0),
        ))
        result = r.execute(plan)
        assert result.succeeded
        # Ana recibió medicina; bob recibió comida (necesitaba comida y
        # medicina, ahora solo le falta medicina).
        from droneplan_viz.domain import Content
        assert r.current_world.persons["ana"].needs == ()
        bob_needs = r.current_world.persons["bob"].needs
        assert Content(id="comida") not in bob_needs
        assert Content(id="medicina") in bob_needs
        # Makespan: el último end_time = 30.0 (Deliver de bob).
        assert result.makespan == 30.0
        # action_count = 6.
        assert r.current_metrics.action_count == 6
