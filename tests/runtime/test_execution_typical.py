"""Tests de "ejecución tipo": planes completos a través de PlanRunner.

Estos tests son end-to-end a nivel de runtime: construyen un World, un
Plan durativo y lo ejecutan con PlanRunner.execute(), verificando el
RESULTADO de una ejecución realista (no la mecánica de un Command
aislado, que ya cubren tests/commands/ y los demás tests de runtime/).

La arquitectura modela el espacio como un grafo de localizaciones
discretas, y la "ejecución tipo" se expresa como World + Plan + PlanRunner.

Categorías:

1. Ejecución tipo (un drone): recoger → mover → entregar, sin fallos.
2. Movimientos concurrentes: varios drones en paralelo sobre recursos
   disjuntos (makespan = máximo, no suma).
3. Uso de transporter: ciclo completo recoger → cargar → mover-con-
   transporter → descargar → entregar.
4. Combinado: logística con transporter de un drone EN PARALELO con la
   entrega directa de otro drone.

Reutiliza el `base_world` y las factorías make_* de tests/runtime/conftest.py.
"""
from __future__ import annotations

from droneplan_viz.domain import DroneState
from droneplan_viz.runtime import Plan, PlanRunner, ScheduledCommand

from tests.runtime.conftest import (
    make_deliver,
    make_load,
    make_move,
    make_move_with_transporter,
    make_pickup,
    make_unload,
)


def _plan(*pairs):
    """Construye un Plan a partir de pares (command, start_time)."""
    return Plan(scheduled=tuple(
        ScheduledCommand(command=cmd, start_time=t) for cmd, t in pairs
    ))


# ===========================================================================
# 1. Ejecución tipo: un drone recoge, mueve y entrega
# ===========================================================================
class TestEjecucionTipo:
    """d1 (en deposito) recoge libre1 (medicina), vuela a casa1 y la
    entrega a ana, que la necesitaba. Es la "ejecución tipo" feliz."""

    def test_entrega_simple_tiene_exito(self, base_world):
        plan = _plan(
            (make_pickup("d1", "izq", "libre1", duration=5.0, cmd_id="pk"), 0.0),
            (make_move("d1", "casa1", duration=8.0, cmd_id="mv"), 5.0),
            (make_deliver("d1", "libre1", "ana", duration=5.0, cmd_id="dl"), 13.0),
        )
        result = PlanRunner(base_world).execute(plan)

        assert result.succeeded
        assert result.failures == ()
        assert result.makespan == 18.0          # último end_time (13 + 5)
        assert result.history_length > 1         # se generaron snapshots

    def test_estado_final_refleja_la_entrega(self, base_world):
        plan = _plan(
            (make_pickup("d1", "izq", "libre1", duration=5.0, cmd_id="pk"), 0.0),
            (make_move("d1", "casa1", duration=8.0, cmd_id="mv"), 5.0),
            (make_deliver("d1", "libre1", "ana", duration=5.0, cmd_id="dl"), 13.0),
        )
        fw = PlanRunner(base_world).execute(plan).final_world

        # ana recibió la medicina que necesitaba.
        assert any(c.id == "medicina" for c in fw.persons["ana"].has_received)
        # El drone terminó en casa1 y vuelve a IDLE (no quedó MOVING/ERROR).
        assert fw.drones["d1"].position == "casa1"
        assert fw.drones["d1"].state == DroneState.IDLE
        # El paquete acabó depositado en casa1 (donde está ana).
        assert fw.packages["libre1"].at.loc_id == "casa1"

    def test_metricas_sin_fallos(self, base_world):
        plan = _plan(
            (make_pickup("d1", "izq", "libre1", duration=5.0, cmd_id="pk"), 0.0),
            (make_move("d1", "casa1", duration=8.0, cmd_id="mv"), 5.0),
            (make_deliver("d1", "libre1", "ana", duration=5.0, cmd_id="dl"), 13.0),
        )
        result = PlanRunner(base_world).execute(plan)
        assert result.final_metrics.failed_commands == 0


# ===========================================================================
# 2. Movimientos concurrentes: dos/tres drones en paralelo
# ===========================================================================
class TestMovimientosConcurrentes:
    """Drones distintos volando a la vez sobre recursos disjuntos: legal.
    El makespan refleja el PARALELISMO (máximo de duraciones), no la suma.
    """

    def test_dos_drones_se_mueven_en_paralelo(self, base_world):
        # d1: deposito → casa1 (10s).  d2: deposito → casa2 (20s).  Ambos en t=0.
        plan = _plan(
            (make_move("d1", "casa1", duration=10.0, cmd_id="m1"), 0.0),
            (make_move("d2", "casa2", duration=20.0, cmd_id="m2"), 0.0),
        )
        result = PlanRunner(base_world).execute(plan)

        assert result.succeeded
        # En paralelo: makespan = max(10, 20) = 20, NO 30 (que sería secuencial).
        assert result.makespan == 20.0
        assert result.final_world.drones["d1"].position == "casa1"
        assert result.final_world.drones["d2"].position == "casa2"

    def test_concurrencia_no_es_suma_de_duraciones(self, base_world):
        plan = _plan(
            (make_move("d1", "casa1", duration=10.0, cmd_id="m1"), 0.0),
            (make_move("d2", "casa2", duration=20.0, cmd_id="m2"), 0.0),
        )
        result = PlanRunner(base_world).execute(plan)
        # La prueba dura del paralelismo: makespan < 10 + 20.
        assert result.makespan < 30.0

    def test_historial_captura_ambos_movimientos(self, base_world):
        plan = _plan(
            (make_move("d1", "casa1", duration=10.0, cmd_id="m1"), 0.0),
            (make_move("d2", "casa2", duration=20.0, cmd_id="m2"), 0.0),
        )
        result = PlanRunner(base_world).execute(plan)
        # Cada Move durativo deja un snapshot 'start' y otro 'end'. Dos
        # Moves → al menos 4 snapshots además del inicial.
        assert result.history_length >= 5

    def test_tres_drones_misma_concurrencia(self, base_world):
        # Reutilizamos base_world (d1, d2). Añadir un tercer drone exigiría
        # otro World; en su lugar verificamos que d1 y d2 pueden encadenar
        # un segundo tramo concurrente tras el primero (ida y vuelta).
        plan = _plan(
            (make_move("d1", "casa1", duration=10.0, cmd_id="m1a"), 0.0),
            (make_move("d2", "casa2", duration=10.0, cmd_id="m2a"), 0.0),
            (make_move("d1", "deposito", duration=10.0, cmd_id="m1b"), 10.0),
            (make_move("d2", "deposito", duration=10.0, cmd_id="m2b"), 10.0),
        )
        result = PlanRunner(base_world).execute(plan)
        assert result.succeeded
        assert result.makespan == 20.0  # dos tramos paralelos de 10s
        assert result.final_world.drones["d1"].position == "deposito"
        assert result.final_world.drones["d2"].position == "deposito"


# ===========================================================================
# 3. Uso de transporter: ciclo completo
# ===========================================================================
class TestUsoDeTransporter:
    """Ciclo de transporte: recoger → cargar en t1 → mover-con-transporter
    → descargar → entregar. Carga DOS paquetes para ejercitar la capacidad
    del transporter y deja uno dentro tras descargar el otro.

    Recordatorio de semántica (verificado contra el Validador): cargar un
    paquete en un transporter exige que el drone lo SOSTENGA antes
    (PickUp → LoadIntoTransporter); por eso el ciclo intercala recogidas.
    Todo el ciclo es del MISMO drone sobre el MISMO transporter, así que
    los Commands van estrictamente secuenciados (no podrían solaparse sin
    chocar por concurrencia de recursos).
    """

    def _plan_transporte(self):
        return _plan(
            (make_pickup("d1", "izq", "libre1", duration=2.0, cmd_id="pk1"), 0.0),
            (make_load("d1", "libre1", "t1", duration=2.0, cmd_id="ld1"), 2.0),
            (make_pickup("d1", "izq", "libre2", duration=2.0, cmd_id="pk2"), 4.0),
            (make_load("d1", "libre2", "t1", duration=2.0, cmd_id="ld2"), 6.0),
            (make_move_with_transporter("d1", "t1", "casa1", duration=10.0, cmd_id="mt"), 8.0),
            (make_unload("d1", "izq", "libre1", "t1", duration=2.0, cmd_id="ul1"), 18.0),
            (make_deliver("d1", "libre1", "ana", duration=2.0, cmd_id="dl"), 20.0),
        )

    def test_ciclo_completo_tiene_exito(self, base_world):
        result = PlanRunner(base_world).execute(self._plan_transporte())
        assert result.succeeded
        assert result.failures == ()
        assert result.makespan == 22.0

    def test_transporter_y_paquetes_acaban_donde_toca(self, base_world):
        fw = PlanRunner(base_world).execute(self._plan_transporte()).final_world
        # El transporter viajó con el drone a casa1.
        assert fw.transporters["t1"].position == "casa1"
        # libre1 se descargó y se entregó (acaba en casa1, con ana).
        assert fw.packages["libre1"].at.loc_id == "casa1"
        assert any(c.id == "medicina" for c in fw.persons["ana"].has_received)
        # libre2 sigue DENTRO del transporter (no se descargó).
        in_t1 = [p.id for p in fw.packages_in_transporter("t1")]
        assert in_t1 == ["libre2"]

    def test_drone_acaba_idle_en_destino(self, base_world):
        fw = PlanRunner(base_world).execute(self._plan_transporte()).final_world
        assert fw.drones["d1"].position == "casa1"
        assert fw.drones["d1"].state == DroneState.IDLE


# ===========================================================================
# 4. Combinado: logística con transporter EN PARALELO con entrega directa
# ===========================================================================
class TestTransporteYConcurrencia:
    """El caso "complejo" pedido: un drone hace logística con transporter
    mientras OTRO drone entrega por su cuenta, simultáneamente, sobre
    recursos disjuntos. Verifica que concurrencia y transporte conviven.
    """

    def test_d1_transporta_mientras_d2_entrega(self, base_world):
        # d1: recoge libre1, la carga en t1, arrastra t1 a casa1, descarga
        #     y entrega a ana (medicina). Secuencial dentro de d1/t1.
        # d2 (en paralelo): vuela a casa1, recoge libre3 (medicina, ya en
        #     casa1), vuela a casa2 y la entrega a bob.
        # Recursos disjuntos (d1/t1/libre1 vs d2/libre3) → sin colisión.
        plan = _plan(
            # --- pista de d1 (transporter) ---
            (make_pickup("d1", "izq", "libre1", duration=2.0, cmd_id="pk1"), 0.0),
            (make_load("d1", "libre1", "t1", duration=2.0, cmd_id="ld1"), 2.0),
            (make_move_with_transporter("d1", "t1", "casa1", duration=10.0, cmd_id="mt"), 4.0),
            (make_unload("d1", "izq", "libre1", "t1", duration=2.0, cmd_id="ul1"), 14.0),
            (make_deliver("d1", "libre1", "ana", duration=2.0, cmd_id="dl1"), 16.0),
            # --- pista de d2 (entrega directa), arrancando a la vez ---
            (make_move("d2", "casa1", duration=6.0, cmd_id="m2a"), 0.0),
            (make_pickup("d2", "izq", "libre3", duration=2.0, cmd_id="pk2"), 6.0),
            (make_move("d2", "casa2", duration=8.0, cmd_id="m2b"), 8.0),
            (make_deliver("d2", "libre3", "bob", duration=2.0, cmd_id="dl2"), 16.0),
        )
        result = PlanRunner(base_world).execute(plan)
        fw = result.final_world

        assert result.succeeded, [(f.kind, f.reason) for f in result.failures]
        # Ambas entregas ocurrieron.
        assert any(c.id == "medicina" for c in fw.persons["ana"].has_received)
        assert any(c.id == "medicina" for c in fw.persons["bob"].has_received)
        # Posiciones finales coherentes con cada pista.
        assert fw.drones["d1"].position == "casa1"
        assert fw.drones["d2"].position == "casa2"
        assert fw.transporters["t1"].position == "casa1"
