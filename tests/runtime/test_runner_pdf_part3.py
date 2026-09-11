"""
Tests trazables al PDF "Práctica 1 — Parte 3: Planificación con tiempos".

Cada test cita literalmente la regla del enunciado que verifica y
construye un escenario canónico que un alumno o evaluador del TFG
podría inspeccionar manualmente. Este archivo NO duplica los tests
de test_runner_concurrency.py: aquellos cubren la mecánica del runner
con muchos casos sintéticos; estos cubren los CASOS CANÓNICOS del PDF
con una correspondencia 1:1 trazable al enunciado.

Las cinco reglas del Ejercicio 3 del PDF (cita literal):

    "Cada dron solo puede realizar una acción al mismo tiempo."
    "Una misma caja y un mismo transportador solo pueden ser cogidos
     por un dron."
    "Mientras un dron mete o saca una caja de un transportador, ningún
     otro dron puede hacerlo en paralelo, ni tampoco coger dicho
     transportador."
    "Una persona solo puede recibir una entrega de un dron al mismo
     tiempo."
    "Todas las acciones que no sean de vuelo tendrán una duración de
     5 segundos."

Más, sobre vuelos:
    "Las acciones con costes deben ser sustituidas por durative actions.
     La duración de las acciones de vuelo seguirá dependiendo de la
     función 'fly-cost'."

Cada bloque de tests está rotulado con la cita y los tests dentro
encarnan situaciones que un planificador LPG-TD podría producir.
"""
from __future__ import annotations

import pytest

from droneplan_viz.domain import Content, DroneState, HeldByArm
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
# REGLA 1: "Cada dron solo puede realizar una acción al mismo tiempo."
# ===========================================================================
class TestRegla1_UnDronUnaAccion:
    """Cita literal del PDF Parte 3:
        'Cada dron solo puede realizar una acción al mismo tiempo.'
    """

    def test_caso_canonico_vuelo_y_pickup_solapados_mismo_dron(
        self, base_world
    ):
        """Caso canónico: el dron d1 está volando (acción Move en
        [0, 10)). Un planificador defectuoso le ordena recoger un
        paquete en [5, 10). El runner rechaza."""
        r = PlanRunner(base_world)
        vuelo = make_move(
            "d1", "casa1", duration=10.0, cmd_id="vuelo"
        )
        recoger = make_pickup(
            "d1", "izq", "libre1",
            duration=5.0, cmd_id="recoger_en_vuelo",
        )
        plan = Plan(scheduled=(
            ScheduledCommand(command=vuelo, start_time=0.0),
            ScheduledCommand(command=recoger, start_time=5.0),
        ))
        result = r.execute(plan)
        assert not result.succeeded
        assert result.failures[0].scheduled.command is recoger

    def test_caso_legal_dos_acciones_del_mismo_dron_sin_solape(
        self, base_world
    ):
        """Caso legal: el dron acaba una acción, empieza la siguiente.
        Intervalos disjuntos [0, 5) y [5, 15). Debe pasar."""
        r = PlanRunner(base_world)
        recoger = make_pickup(
            "d1", "izq", "libre1", duration=5.0, cmd_id="r"
        )
        volar = make_move("d1", "casa1", duration=10.0, cmd_id="v")
        plan = Plan(scheduled=(
            ScheduledCommand(command=recoger, start_time=0.0),
            ScheduledCommand(command=volar, start_time=5.0),
        ))
        result = r.execute(plan)
        assert result.succeeded


# ===========================================================================
# REGLA 2 (caja): "Una misma caja [...] solo puede ser cogida por un dron."
# ===========================================================================
class TestRegla2_UnaCajaUnDron:
    """Cita literal del PDF Parte 3:
        'Una misma caja y un mismo transportador solo pueden ser
         cogidos por un dron.'

    Esta clase verifica la parte de la caja; el transportador se trata
    en TestRegla2_UnTransportadorUnDron.
    """

    def test_caso_canonico_dos_drones_compiten_por_libre1(
        self, base_world
    ):
        """Escenario LPG-TD plausible: dos drones en el mismo depósito,
        ambos planificados para recoger 'libre1' simultáneamente. El
        segundo en el orden de ejecución falla."""
        r = PlanRunner(base_world)
        coge_d1 = make_pickup(
            "d1", "izq", "libre1", duration=5.0, cmd_id="coge_d1"
        )
        coge_d2 = make_pickup(
            "d2", "izq", "libre1", duration=5.0, cmd_id="coge_d2"
        )
        plan = Plan(scheduled=(
            ScheduledCommand(command=coge_d1, start_time=0.0),
            ScheduledCommand(command=coge_d2, start_time=2.0),
        ))
        result = r.execute(plan)
        assert not result.succeeded
        assert result.failures[0].kind == "concurrency"
        assert "libre1" in result.failures[0].reason

    def test_caso_legal_dos_drones_dos_cajas_distintas(self, base_world):
        """En el depósito hay dos cajas. Dos drones pueden cogerlas
        simultáneamente. Caso fundamental de paralelización útil."""
        r = PlanRunner(base_world)
        coge_d1 = make_pickup(
            "d1", "izq", "libre1", duration=5.0, cmd_id="coge_d1"
        )
        coge_d2 = make_pickup(
            "d2", "izq", "libre2", duration=5.0, cmd_id="coge_d2"
        )
        plan = Plan(scheduled=(
            ScheduledCommand(command=coge_d1, start_time=0.0),
            ScheduledCommand(command=coge_d2, start_time=0.0),
        ))
        result = r.execute(plan)
        assert result.succeeded
        assert r.current_world.packages["libre1"].at == HeldByArm(
            drone_id="d1", arm_id="izq"
        )
        assert r.current_world.packages["libre2"].at == HeldByArm(
            drone_id="d2", arm_id="izq"
        )


# ===========================================================================
# REGLA 2 (transportador): "Un mismo transportador solo puede ser
# cogido por un dron."
# ===========================================================================
class TestRegla2_UnTransportadorUnDron:
    """Cita literal del PDF Parte 3:
        'Una misma caja y un mismo transportador solo pueden ser
         cogidos por un dron.'

    Aquí se ejercita la parte del transportador.
    """

    def test_caso_canonico_dos_drones_compiten_por_t1(self, base_world):
        """Dos drones intentan mover el mismo transportador a destinos
        distintos en intervalos solapados. El segundo falla."""
        r = PlanRunner(base_world)
        d1_con_t1 = make_move_with_transporter(
            "d1", "t1", "casa1", duration=10.0, cmd_id="d1_t1"
        )
        d2_con_t1 = make_move_with_transporter(
            "d2", "t1", "casa2", duration=20.0, cmd_id="d2_t1"
        )
        plan = Plan(scheduled=(
            ScheduledCommand(command=d1_con_t1, start_time=0.0),
            ScheduledCommand(command=d2_con_t1, start_time=5.0),
        ))
        result = r.execute(plan)
        assert not result.succeeded
        assert result.failures[0].kind == "concurrency"
        assert "t1" in result.failures[0].reason

    def test_caso_legal_dos_drones_dos_transportadores(self, base_world):
        """base_world tiene dos transportadores (t1 y t2). Dos drones
        pueden moverlos simultáneamente. Patrón de logística humanitaria
        con múltiples vehículos."""
        r = PlanRunner(base_world)
        d1_con_t1 = make_move_with_transporter(
            "d1", "t1", "casa1", duration=10.0, cmd_id="d1_t1"
        )
        d2_con_t2 = make_move_with_transporter(
            "d2", "t2", "casa2", duration=20.0, cmd_id="d2_t2"
        )
        plan = Plan(scheduled=(
            ScheduledCommand(command=d1_con_t1, start_time=0.0),
            ScheduledCommand(command=d2_con_t2, start_time=0.0),
        ))
        result = r.execute(plan)
        assert result.succeeded
        assert r.current_world.transporters["t1"].position == "casa1"
        assert r.current_world.transporters["t2"].position == "casa2"


# ===========================================================================
# REGLA 3: "Mientras un dron mete o saca una caja de un transportador,
# ningún otro dron puede hacerlo en paralelo, ni tampoco coger dicho
# transportador."
# ===========================================================================
class TestRegla3_LoadUnloadMutexTransportador:
    """Cita literal del PDF Parte 3:
        'Mientras un dron mete o saca una caja de un transportador,
         ningún otro dron puede hacerlo en paralelo, ni tampoco coger
         dicho transportador.'

    Esta regla refuerza la regla 2 con tres formas concretas que
    verificamos por separado:
        (a) d1 mete; d2 mete -> rechazado
        (b) d1 mete; d2 saca -> rechazado
        (c) d1 mete; d2 coge el transportador (MoveWithTransporter) ->
            rechazado
    """

    def test_a_dos_drones_metiendo_en_el_mismo_transportador(
        self, base_world
    ):
        """Forma (a) de la regla 3."""
        world = world_with_package_held(base_world, "libre1", "d1", "izq")
        world = world_with_package_held(world, "libre2", "d2", "izq")
        r = PlanRunner(world)
        d1_mete = make_load(
            "d1", "libre1", "t1", duration=5.0, cmd_id="d1_mete"
        )
        d2_mete = make_load(
            "d2", "libre2", "t1", duration=5.0, cmd_id="d2_mete"
        )
        plan = Plan(scheduled=(
            ScheduledCommand(command=d1_mete, start_time=0.0),
            ScheduledCommand(command=d2_mete, start_time=2.0),
        ))
        result = r.execute(plan)
        assert not result.succeeded
        assert result.failures[0].kind == "concurrency"
        assert result.failures[0].scheduled.command is d2_mete

    def test_b_un_dron_mete_otro_saca_del_mismo_transportador(
        self, base_world
    ):
        """Forma (b) de la regla 3: cargar y descargar simultáneamente
        sobre el mismo transportador es ilegal."""
        # d1 sostiene libre2 (para meter); libre1 ya está dentro de t1
        # (para que d2 lo saque).
        world = world_with_package_held(base_world, "libre2", "d1", "izq")
        world = world_with_package_in_transporter(world, "libre1", "t1")
        r = PlanRunner(world)
        d1_mete = make_load(
            "d1", "libre2", "t1", duration=5.0, cmd_id="d1_mete"
        )
        d2_saca = make_unload(
            "d2", "izq", "libre1", "t1",
            duration=5.0, cmd_id="d2_saca",
        )
        plan = Plan(scheduled=(
            ScheduledCommand(command=d1_mete, start_time=0.0),
            ScheduledCommand(command=d2_saca, start_time=2.0),
        ))
        result = r.execute(plan)
        assert not result.succeeded
        assert result.failures[0].kind == "concurrency"
        assert result.failures[0].scheduled.command is d2_saca

    def test_c_un_dron_mete_otro_coge_el_transportador(
        self, base_world
    ):
        """Forma (c) de la regla 3: cargar el transportador y volar
        con él al mismo tiempo es ilegal. La cita literal habla de
        'coger dicho transportador'; MoveWithTransporter es la acción
        que lo coge en nuestro modelo."""
        world = world_with_package_held(base_world, "libre1", "d1", "izq")
        r = PlanRunner(world)
        d1_mete = make_load(
            "d1", "libre1", "t1", duration=5.0, cmd_id="d1_mete"
        )
        d2_vuela_con_t1 = make_move_with_transporter(
            "d2", "t1", "casa1", duration=10.0, cmd_id="d2_vuela"
        )
        plan = Plan(scheduled=(
            ScheduledCommand(command=d1_mete, start_time=0.0),
            ScheduledCommand(command=d2_vuela_con_t1, start_time=2.0),
        ))
        result = r.execute(plan)
        assert not result.succeeded
        assert result.failures[0].kind == "concurrency"
        assert result.failures[0].scheduled.command is d2_vuela_con_t1

    def test_caso_legal_dos_loads_en_transportadores_distintos(
        self, base_world
    ):
        """Cada dron carga en un transportador distinto. Legal."""
        world = world_with_package_held(base_world, "libre1", "d1", "izq")
        world = world_with_package_held(world, "libre2", "d2", "izq")
        r = PlanRunner(world)
        d1_en_t1 = make_load(
            "d1", "libre1", "t1", duration=5.0, cmd_id="d1_t1"
        )
        d2_en_t2 = make_load(
            "d2", "libre2", "t2", duration=5.0, cmd_id="d2_t2"
        )
        plan = Plan(scheduled=(
            ScheduledCommand(command=d1_en_t1, start_time=0.0),
            ScheduledCommand(command=d2_en_t2, start_time=0.0),
        ))
        result = r.execute(plan)
        assert result.succeeded


# ===========================================================================
# REGLA 4: "Una persona solo puede recibir una entrega de un dron al
# mismo tiempo."
# ===========================================================================
class TestRegla4_UnaPersonaUnaEntrega:
    """Cita literal del PDF Parte 3:
        'Una persona solo puede recibir una entrega de un dron al
         mismo tiempo.'

    Esta regla es la única donde la persona aparece como recurso
    exclusivo. En el dominio bob necesita dos contenidos (comida y
    medicina), lo que permite construir escenarios con dos entregas
    distintas a la misma persona.
    """

    def test_caso_canonico_dos_drones_entregando_a_bob(self, base_world):
        """d1 entrega medicina, d2 entrega comida, ambos a bob,
        simultáneamente. La regla 4 lo prohíbe: bob no puede recibir
        dos entregas a la vez aunque sean cosas distintas."""
        # Ambos drones en casa2 con paquetes distintos sostenidos.
        world = world_with_drone_at(base_world, "d1", "casa2")
        world = world_with_drone_at(world, "d2", "casa2")
        world = world_with_package_held(world, "libre1", "d1", "izq")
        world = world_with_package_held(world, "libre2", "d2", "izq")

        r = PlanRunner(world)
        d1_entrega_medicina = make_deliver(
            "d1", "libre1", "bob",
            duration=5.0, cmd_id="d1_medicina",
        )
        d2_entrega_comida = make_deliver(
            "d2", "libre2", "bob",
            duration=5.0, cmd_id="d2_comida",
        )
        plan = Plan(scheduled=(
            ScheduledCommand(
                command=d1_entrega_medicina, start_time=0.0
            ),
            ScheduledCommand(
                command=d2_entrega_comida, start_time=2.0
            ),
        ))
        result = r.execute(plan)
        assert not result.succeeded
        assert result.failures[0].kind == "concurrency"
        assert "bob" in result.failures[0].reason

    def test_caso_legal_entregas_secuenciales_a_la_misma_persona(
        self, base_world
    ):
        """Dos entregas a bob, una tras otra (intervalos disjuntos).
        Bob recibe los dos contenidos."""
        # d1 entrega medicina [0, 5); luego d2 entrega comida [5, 10).
        world = world_with_drone_at(base_world, "d1", "casa2")
        world = world_with_drone_at(world, "d2", "casa2")
        world = world_with_package_held(world, "libre1", "d1", "izq")
        world = world_with_package_held(world, "libre2", "d2", "izq")

        r = PlanRunner(world)
        d1_medicina = make_deliver(
            "d1", "libre1", "bob",
            duration=5.0, cmd_id="d1_m",
        )
        d2_comida = make_deliver(
            "d2", "libre2", "bob",
            duration=5.0, cmd_id="d2_c",
        )
        plan = Plan(scheduled=(
            ScheduledCommand(command=d1_medicina, start_time=0.0),
            ScheduledCommand(command=d2_comida, start_time=5.0),
        ))
        result = r.execute(plan)
        assert result.succeeded
        assert r.current_world.persons["bob"].needs == ()

    def test_caso_legal_entregas_simultaneas_a_personas_distintas(
        self, base_world
    ):
        """d1 entrega a ana, d2 entrega a bob. Personas distintas,
        entrega simultánea legal."""
        world = world_with_drone_at(base_world, "d1", "casa1")
        world = world_with_drone_at(world, "d2", "casa2")
        world = world_with_package_held(world, "libre1", "d1", "izq")
        # libre2 es comida; bob necesita comida. ✓
        world = world_with_package_held(world, "libre2", "d2", "izq")

        r = PlanRunner(world)
        a = make_deliver(
            "d1", "libre1", "ana", duration=5.0, cmd_id="a"
        )
        b = make_deliver(
            "d2", "libre2", "bob", duration=5.0, cmd_id="b"
        )
        plan = Plan(scheduled=(
            ScheduledCommand(command=a, start_time=0.0),
            ScheduledCommand(command=b, start_time=0.0),
        ))
        result = r.execute(plan)
        assert result.succeeded


# ===========================================================================
# REGLA 5: "Todas las acciones que no sean de vuelo tendrán una
# duración de 5 segundos."
# ===========================================================================
class TestRegla5_DuracionesPDF:
    """Cita literal del PDF Parte 3:
        'Todas las acciones que no sean de vuelo tendrán una duración
         de 5 segundos.'

    Adicionalmente:
        'Las acciones con costes deben ser sustituidas por durative
         actions. La duración de las acciones de vuelo seguirá
         dependiendo de la función fly-cost.'

    IMPORTANTE: el runner NO inventa duraciones. La traducción 'acción
    no-vuelo -> duración=5.0' y 'acción de vuelo -> duración=fly-cost'
    es responsabilidad del FACADE (Sesión E), que al parsear un .pddl
    parte 3 construye los Commands con duration ya poblado. Estos
    tests verifican que el runner HONRA esos valores: NO modifica
    cmd.duration, lo respeta tal cual se lo pasen.
    """

    def test_runner_respeta_duration_5_en_pickup(self, base_world):
        """Construimos un PickUp con duration=5.0 (regla del PDF) y
        verificamos que el end_time del snapshot end coincide."""
        r = PlanRunner(base_world)
        recoger = make_pickup(
            "d1", "izq", "libre1", duration=5.0, cmd_id="r5"
        )
        plan = Plan(scheduled=(
            ScheduledCommand(command=recoger, start_time=0.0),
        ))
        r.execute(plan)
        # snapshot 0 = inicial; 1 = start (t=0); 2 = end (t=5).
        assert r.history.at(2).timestamp == 5.0

    def test_runner_respeta_duration_5_en_deliver(self, base_world):
        world = world_with_drone_at(base_world, "d1", "casa1")
        world = world_with_package_held(world, "libre1", "d1", "izq")
        r = PlanRunner(world)
        entregar = make_deliver(
            "d1", "libre1", "ana",
            duration=5.0, cmd_id="e5",
        )
        plan = Plan(scheduled=(
            ScheduledCommand(command=entregar, start_time=10.0),
        ))
        r.execute(plan)
        # end de un Deliver en t=10 con duración 5 → t=15.
        assert r.history.at(2).timestamp == 15.0

    def test_runner_respeta_duration_5_en_load_unload(self, base_world):
        """Verifica los otros dos no-vuelo de duración 5.0."""
        world = world_with_package_held(base_world, "libre1", "d1", "izq")
        r = PlanRunner(world)
        meter = make_load(
            "d1", "libre1", "t1",
            duration=5.0, cmd_id="meter",
        )
        plan = Plan(scheduled=(
            ScheduledCommand(command=meter, start_time=0.0),
        ))
        r.execute(plan)
        assert r.history.at(2).timestamp == 5.0


class TestRegla5_VueloConFlyCost:
    """Cita literal del PDF Parte 3:
        'La duración de las acciones de vuelo seguirá dependiendo de
         la función fly-cost.'

    En base_world:
        fly-cost(deposito, casa1) = 10.0
        fly-cost(deposito, casa2) = 20.0
        fly-cost(casa1, casa2)    = 15.0

    El facade traducirá un .pddl parte 3 mapeando cada acción de vuelo
    a Move con duration = fly_cost(origen, destino). Verificamos que
    el runner honra ese mapeo y que las dos funciones (total_cost vía
    fly-cost, total_time vía duration) coexisten correctamente.
    """

    def test_vuelo_corto_y_largo_simultaneos(self, base_world):
        """d1 vuela deposito->casa1 (fly-cost=10, duration=10).
        d2 vuela deposito->casa2 (fly-cost=20, duration=20).
        Ambos en paralelo desde t=0. Makespan = 20."""
        r = PlanRunner(base_world)
        v1 = make_move("d1", "casa1", duration=10.0, cmd_id="v1")
        v2 = make_move("d2", "casa2", duration=20.0, cmd_id="v2")
        plan = Plan(scheduled=(
            ScheduledCommand(command=v1, start_time=0.0),
            ScheduledCommand(command=v2, start_time=0.0),
        ))
        result = r.execute(plan)
        assert result.succeeded
        assert result.makespan == 20.0
        # total_cost = 10 + 20 = 30 (suma de fly-costs).
        assert result.final_metrics.total_cost == 30.0
        # total_time = 20 (makespan, no suma — bug fix de Sesión B).
        assert result.final_metrics.total_time == 20.0

    def test_total_time_distinto_de_total_cost(self, base_world):
        """Decisión PDDL parte 3: total_time != total_cost. Ambas
        métricas coexisten. Documentación viva."""
        r = PlanRunner(base_world)
        v = make_move("d1", "casa2", duration=20.0, cmd_id="v")
        plan = Plan(scheduled=(
            ScheduledCommand(command=v, start_time=0.0),
        ))
        result = r.execute(plan)
        # total_cost = 20 (fly-cost); total_time = 20 (makespan).
        # En este caso coinciden, pero ojo: el TIPO es distinto. Una
        # es "cost" del PDDL parte 2 (acumulativa), otra es "time" del
        # PDDL parte 3 (makespan).
        assert result.final_metrics.total_cost == 20.0
        assert result.final_metrics.total_time == 20.0


# ===========================================================================
# Escenario integrador: plan parte 3 multi-drone con duraciones reales
# ===========================================================================
class TestEscenarioIntegradorParte3:
    """Escenario completo derivado del PDF parte 3: dos drones, dos
    transportadores, varias entregas. Verifica que las cinco reglas
    operan a la vez sin conflictos y que la métrica makespan refleja
    correctamente la paralelización."""

    def test_dos_misiones_paralelas_a_dos_destinos(self, base_world):
        """
        Plan completo:
            d1: pickup libre1 [0,5), move a casa1 [5,15),
                deliver libre1 a ana [15,20).
            d2: pickup libre2 [0,5), move a casa2 [5,25),
                deliver libre2 a bob [25,30).
        Recursos disjuntos en todo momento. Makespan = 30.
        """
        r = PlanRunner(base_world)
        # d1: medicina -> ana (necesita medicina).
        p1 = make_pickup(
            "d1", "izq", "libre1", duration=5.0, cmd_id="p1"
        )
        mv1 = make_move("d1", "casa1", duration=10.0, cmd_id="mv1")
        de1 = make_deliver(
            "d1", "libre1", "ana", duration=5.0, cmd_id="de1"
        )
        # d2: comida -> bob (necesita comida y medicina).
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
        # Métricas:
        assert result.makespan == 30.0
        assert result.final_metrics.total_time == 30.0
        # fly-costs: 10 (mv1) + 20 (mv2) = 30.
        assert result.final_metrics.total_cost == 30.0
        # 6 acciones exitosas.
        assert result.final_metrics.action_count == 6
        # 0 fallos.
        assert result.final_metrics.failed_commands == 0
        # Necesidades:
        assert r.current_world.persons["ana"].needs == ()
        assert Content(id="comida") not in (
            r.current_world.persons["bob"].needs
        )
        # Ambos drones IDLE al final (todas las acciones terminaron).
        assert r.current_world.drones["d1"].state == DroneState.IDLE
        assert r.current_world.drones["d2"].state == DroneState.IDLE

    def test_plan_compartiendo_transportador_secuencialmente(
        self, base_world
    ):
        """Patrón realista del PDF parte 2 extendido al parte 3:
        un solo transportador, dos drones que lo usan
        secuencialmente. d1 carga libre1 en t1 [0,5),
        d1 vuela con t1 a casa1 [5,15), d2 (que estaba esperando en
        casa1) descarga libre1 de t1 [15,20).
        """
        # d2 ya está en casa1 esperando.
        world = world_with_drone_at(base_world, "d2", "casa1")
        r = PlanRunner(world)
        # d1: pickup libre1, load en t1, mover t1 a casa1.
        p1 = make_pickup(
            "d1", "izq", "libre1", duration=5.0, cmd_id="p1"
        )
        l1 = make_load(
            "d1", "libre1", "t1", duration=5.0, cmd_id="l1"
        )
        mv1 = make_move_with_transporter(
            "d1", "t1", "casa1", duration=10.0, cmd_id="mv1"
        )
        # d2: descarga libre1 de t1 (en casa1).
        u1 = make_unload(
            "d2", "izq", "libre1", "t1",
            duration=5.0, cmd_id="u1",
        )
        plan = Plan(scheduled=(
            ScheduledCommand(command=p1, start_time=0.0),    # [0, 5)
            ScheduledCommand(command=l1, start_time=5.0),    # [5, 10)
            ScheduledCommand(command=mv1, start_time=10.0),  # [10, 20)
            ScheduledCommand(command=u1, start_time=20.0),   # [20, 25)
        ))
        result = r.execute(plan)
        assert result.succeeded
        # libre1 acaba en el brazo izq de d2.
        assert r.current_world.packages["libre1"].at == HeldByArm(
            drone_id="d2", arm_id="izq"
        )
        # Makespan = 25.
        assert result.makespan == 25.0
