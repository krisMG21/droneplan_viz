"""DronePlanViz: la superficie pública principal de la librería.

Orquesta los dos builders (world, agents) sobre un `_FacadeState`
compartido y expone las acciones del plan. El flujo del alumno:

    from droneplan_viz import DronePlanViz

    viz = DronePlanViz()
    viz.world.location("deposito")
    viz.world.location("casa1")
    viz.world.content("comida")
    viz.world.person("p1", at="casa1", necesita=["comida"])
    viz.world.package("caja1", contiene="comida", at="deposito")
    viz.agents.drone("dron1", at="deposito")

    viz.recoger("dron1", caja="caja1", brazo="izq")
    viz.mover("dron1", a="casa1")
    viz.entregar("dron1", caja="caja1", a="p1")

    viz.run()            # ejecuta y abre la app interactiva

Diseño:

  - Identificación dual (decisión 2): toda referencia acepta str (el id) o
    el objeto devuelto por un builder. Lo resuelve references.resolve_id.

  - Bilingüe por alias: cada acción tiene nombre canónico en castellano y
    alias inglés (mover/move, recoger/grab, entregar/deliver,
    poner_en/load_into, sacar_de/unload_from). Mismo callable, dos nombres.

  - Simetría de brazos: recoger y sacar_de OCUPAN un brazo concreto -> lo
    llevan en `brazo=`. entregar y poner_en LIBERAN el brazo que sea -> no
    lo llevan. Es la misma semántica que los Commands del dominio.

  - Las acciones NO ejecutan: construyen un Command y lo ENCOLAN en orden.
    La validación de referencias es eager y localizada (el error salta en
    la línea de la acción). Los concerns globales del plan (homogeneidad
    temporal e identidad de los Commands) se resuelven en build().

  - Tiempos, regla "todo o nada" sobre `inicio`:
      * Ninguna acción con `inicio=`  -> modo SECUENCIAL: la fachada calcula
        timestamps encadenados sin solape (start_0 = 0, start_{i+1} =
        start_i + max(dur_i, GAP)). Una acción sin `duracion=` recibe una
        duración por defecto > 0 (_DEFAULT_DURATION), así que los planes FF
        (partes 1-2) dan tiempos 0, 1, 2, … pero CADA acción dura lo
        suficiente para animarse (vuelo + coreografía); no degeneran en
        saltos instantáneos. NO se usa Plan.sequential, que solapa
        duraciones reales.
      * Todas con `inicio=`           -> modo TEMPORAL: ScheduledCommands con
        esos tiempos absolutos (parte 3 OPTIC, concurrencia inter-agente).
      * Mezcla                        -> MixedTimingError (didáctico).
    `duracion` es independiente de `inicio`: dar duracion sin inicio es
    legítimo (un plan FF atemporal que quiere verse con duraciones reales
    en la animación) y NO viola "todo o nada". Omitir duracion tampoco la
    viola: simplemente se usa la duración por defecto.

  - command_id deterministas: si la acción trae `id=`, se usa verbatim (es
    la etiqueta del paso PDDL); si no, build() genera "{verbo}_{posición}"
    (recoger_1, mover_2, …). Legible en el panel de Fallos del HUD y
    estable entre construcciones (hace pasar test_funcion_es_pura sin
    tocarlo). Un id repetido -> DuplicateIdError.

  - La acción "soltar" (dejar una caja en el suelo sin entregarla a una
    persona) NO se expone en esta sesión: requiere un Command "Drop" y su
    handler que aún no existen en commands/. En los .pddl de la asignatura
    el patrón es siempre recoger -> mover -> entregar, sin soltar
    intermedio, así que la omisión no bloquea ningún ejemplo real. Añadir
    en una futura iteración de commands si algún plan lo requiere.

  - build()/simular() son headless (sin pygame); run() abre la app
    reutilizando la de la el runtime (import lazy, ver run()).
"""
from __future__ import annotations

import colorsys
from dataclasses import replace
from typing import TYPE_CHECKING

from droneplan_viz.commands import (
    Deliver,
    LoadIntoTransporter,
    Move,
    MoveWithTransporter,
    PickUp,
    UnloadFromTransporter,
)
from droneplan_viz.commands.base import Command
from droneplan_viz.domain.world import World
from droneplan_viz.facade.agent_builder import AgentBuilder
from droneplan_viz.facade.errors import (
    DuplicateIdError,
    FacadeError,
    MixedTimingError,
    UnknownArmError,
    WrongReferenceKindError,
)
from droneplan_viz.facade.references import (
    resolve_drone_id,
    resolve_id,
    resolve_location_id,
    resolve_package_id,
    resolve_person_id,
    resolve_transporter_id,
)
from droneplan_viz.facade.state import _FacadeState, _QueuedAction
from droneplan_viz.facade.world_builder import WorldBuilder
from droneplan_viz.runtime import Plan, PlanRunner, ScheduledCommand

if TYPE_CHECKING:  # solo para anotaciones; evita importar el runtime de salida
    from droneplan_viz.runtime import RunResult


#: Duración (segundos de plan) que se asigna a una acción cuando el alumno
#: NO pasa `duracion=` (planes FF / atemporales). Es > 0 a propósito: una
#: duración de 0 haría que el runtime emitiera un único snapshot por acción
#: y el render no tendría ningún tramo que interpolar -> el dron "saltaría"
#: entre localizaciones sin animación de vuelo ni coreografía de
#: interacción. Con una duración por defecto > 0, el runner emite los dos
#: snapshots (inicio/fin) que el render necesita para animar. No se ofrece
#: forma de desactivar la animación: un plan siempre se ve moverse.
_DEFAULT_DURATION = 1.0

#: Separación temporal mínima entre dos acciones consecutivas en modo
#: secuencial. Actúa de suelo para duraciones explícitas muy pequeñas; con
#: la duración por defecto (== este valor) las acciones quedan contiguas y
#: producen tiempos de inicio 0, 1, 2, … sin solaparse.
_SEQUENTIAL_GAP = 1.0


#: Prefijo de verbo para el command_id determinista, por tipo de Command.
_VERB_BY_TYPE: dict[type, str] = {
    Move: "mover",
    MoveWithTransporter: "mover",
    PickUp: "recoger",
    Deliver: "entregar",
    LoadIntoTransporter: "poner_en",
    UnloadFromTransporter: "sacar_de",
}


def _coerce_time(value: float, *, nombre: str) -> float:
    """Valida y normaliza un tiempo (inicio o duracion) a float >= 0."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise FacadeError(f"{nombre} debe ser un número; recibí {value!r}")
    v = float(value)
    if v < 0:
        raise FacadeError(f"{nombre} no puede ser negativo; recibí {v}")
    return v


#: Semilla por defecto de la paleta de color por contenido. Fija → la paleta
#: es REPRODUCIBLE (mismo color por contenido en cada ejecución y en tests).
#: colores_contenido(seed=...) permite otra semilla si se quiere otro reparto.
_DEFAULT_PALETTE_SEED = 1234

#: Conjugado áureo (√5−1)/2 ≈ 0.618: fracción de vuelta del círculo de tono que
#: se avanza por cada contenido (≈137.5°). Es la secuencia de baja discrepancia
#: ÓPTIMA sobre el círculo: tonos consecutivos lo más separados posible y que
#: NUNCA se agrupan, por muchos contenidos que haya (a diferencia de tonos
#: aleatorios, que colisionan por azar ya con 3-4 tipos). Esto materializa la
#: idea "primarios → secundarios → mezclas" de forma determinista y óptima.
_GOLDEN_RATIO_CONJUGATE = 0.6180339887498949

#: Escalones (valor, saturación) que se alternan por índice como SEGUNDO eje
#: distintivo: aunque el tono ya separa, variar brillo/saturación entre
#: contenidos consecutivos los hace aún más inconfundibles y cubre con holgura
#: el caso de muchos tipos (la idea de "repetir bajando el brillo"). Todos
#: quedan vivos y legibles como outline sobre el fondo oscuro.
_VIVID_TIERS: tuple[tuple[float, float], ...] = (
    (1.00, 0.85),
    (0.88, 0.68),
    (0.96, 0.98),
    (0.80, 0.78),
)


def _spaced_vivid_color(index: int, phase: float = 0.0) -> tuple[int, int, int]:
    """Color RGB vivo y DETERMINISTA por índice, máximamente separado de sus vecinos.

    El tono avanza por el ángulo áureo (secuencia de baja discrepancia),
    desplazado por `phase`; el brillo y la saturación recorren `_VIVID_TIERS`
    de forma cíclica como segundo eje distintivo. Sin RNG: el mismo índice da
    siempre el mismo color, y contenidos consecutivos son inconfundibles.

    Args:
        index: posición (0, 1, 2, …) del contenido en el orden estable.
        phase: desplazamiento global del tono (0..1); la semilla lo usa para
            obtener otro reparto igual de separado.
    """
    h = (phase + index * _GOLDEN_RATIO_CONJUGATE) % 1.0
    v, s = _VIVID_TIERS[index % len(_VIVID_TIERS)]
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (round(r * 255), round(g * 255), round(b * 255))


class DronePlanViz:
    """Fachada ergonómica de droneplan_viz.

    Atributos públicos:
        world: WorldBuilder para declarar localizaciones, contenidos,
            personas, paquetes y costes.
        agents: AgentBuilder para declarar drones y transportadores.

    Los métodos de acción (mover, recoger, entregar, poner_en, sacar_de y
    sus alias ingleses) encolan Commands. build()/simular()/run() los
    materializan.
    """

    __slots__ = ("_state", "world", "agents", "_declared_ids")

    def __init__(self) -> None:
        self._state = _FacadeState()
        self.world = WorldBuilder(self._state)
        self.agents = AgentBuilder(self._state)
        #: ids explícitos ya usados por `id=`, para detectar duplicados
        #: pronto (en la propia llamada de la acción).
        self._declared_ids: set[str] = set()

    # =================================================================
    # Helpers internos
    # =================================================================
    def _enqueue(
        self,
        command: Command,
        *,
        inicio: float | None,
        duracion: float | None,
        declared_id: str | None,
    ) -> None:
        """Registra un Command en la cola, normalizando tiempos e id."""
        start_time = (
            None if inicio is None else _coerce_time(inicio, nombre="inicio")
        )
        duration = (
            _DEFAULT_DURATION
            if duracion is None
            else _coerce_time(duracion, nombre="duracion")
        )
        if declared_id is not None:
            if not isinstance(declared_id, str) or not declared_id.strip():
                raise FacadeError(
                    f"el id de una acción debe ser un texto no vacío; "
                    f"recibí {declared_id!r}"
                )
            if declared_id in self._declared_ids:
                raise DuplicateIdError(
                    declared_id, categoria="una acción"
                )
            self._declared_ids.add(declared_id)
        # La duración viaja en el Command (para la animación y el makespan);
        # la reconstruimos con replace para inyectarla sin depender del
        # orden de campos de cada dataclass.
        command = replace(command, duration=duration)
        self._state.queued.append(
            _QueuedAction(
                command=command,
                start_time=start_time,
                duration=duration,
                declared_id=declared_id,
            )
        )

    def _resolve_brazo(self, brazo: object, drone_id: str) -> str:
        """Resuelve un brazo a su nombre y exige que EXISTA en el dron.

        Solo valida existencia (construcción), no ocupación (física, del
        runner). Un dron explorador (arms=()) produce el mensaje específico
        de UnknownArmError.
        """
        arm_id = resolve_id(brazo)
        drone = self._state.drones[drone_id]
        disponibles = tuple(a.id for a in drone.arms)
        if arm_id not in disponibles:
            raise UnknownArmError(
                arm_id, drone_id, brazos_disponibles=disponibles
            )
        return arm_id

    def _resolve_destino_localizacion(
        self, a: object, *, metodo: str
    ) -> str:
        """Resuelve `a` como localización destino, con guarda cruzada.

        Si `a` es en realidad una persona declarada, lo dice con precisión
        (WrongReferenceKindError) en vez de un genérico "no existe".
        """
        a_id = resolve_id(a)
        if a_id not in self._state.locations and a_id in self._state.persons:
            raise WrongReferenceKindError(
                a_id,
                metodo=metodo,
                parametro="a",
                esperado="una localización",
                recibido="la persona",
            )
        return resolve_location_id(
            a, self._state.locations, contexto=f"al encolar la acción {metodo}"
        )

    def _resolve_destino_persona(self, a: object, *, metodo: str) -> str:
        """Resuelve `a` como persona destinataria, con guarda cruzada."""
        a_id = resolve_id(a)
        if a_id not in self._state.persons and a_id in self._state.locations:
            raise WrongReferenceKindError(
                a_id,
                metodo=metodo,
                parametro="a",
                esperado="una persona",
                recibido="la localización",
            )
        return resolve_person_id(
            a, self._state.persons, contexto=f"al encolar la acción {metodo}"
        )

    # =================================================================
    # Acciones del plan
    # =================================================================
    def mover(
        self,
        dron: object,
        a: object,
        con: object | None = None,
        *,
        inicio: float | None = None,
        duracion: float | None = None,
        id: str | None = None,
    ) -> None:
        """Acción 'volar': el dron se desplaza a otra localización.

        Args:
            dron: dron que vuela (id o referencia).
            a: localización DESTINO (id o referencia). Debe estar declarada.
            con: transportador opcional a arrastrar (id o referencia). Si se
                da, la acción es 'mover-transportador'.
            inicio: instante absoluto de inicio (modo temporal). Ver regla
                "todo o nada" en el docstring del módulo.
            duracion: duración de la acción (default 0.0).
            id: identificador explícito de la acción (la etiqueta del paso
                PDDL). Debe ser único en el escenario.

        Alias inglés: move.
        """
        contexto = "al encolar la acción mover"
        drone_id = resolve_drone_id(
            dron, self._state.drones, contexto=contexto
        )
        dest_id = self._resolve_destino_localizacion(a, metodo="mover")
        if con is None:
            command: Command = Move(drone_id=drone_id, destination_id=dest_id)
        else:
            transporter_id = resolve_transporter_id(
                con, self._state.transporters, contexto=contexto
            )
            command = MoveWithTransporter(
                drone_id=drone_id,
                transporter_id=transporter_id,
                destination_id=dest_id,
            )
        self._enqueue(command, inicio=inicio, duracion=duracion, declared_id=id)

    def recoger(
        self,
        dron: object,
        caja: object,
        brazo: object,
        *,
        inicio: float | None = None,
        duracion: float | None = None,
        id: str | None = None,
    ) -> None:
        """Acción 'recoger': el dron toma una caja del suelo con un brazo.

        recoger OCUPA un brazo concreto, por eso lleva `brazo=` (simétrico
        con sacar_de; entregar y poner_en lo liberan y no lo llevan).

        Alias inglés: grab.
        """
        contexto = "al encolar la acción recoger"
        drone_id = resolve_drone_id(
            dron, self._state.drones, contexto=contexto
        )
        package_id = resolve_package_id(
            caja, self._state.packages, contexto=contexto
        )
        arm_id = self._resolve_brazo(brazo, drone_id)
        command = PickUp(
            drone_id=drone_id, arm_id=arm_id, package_id=package_id
        )
        self._enqueue(command, inicio=inicio, duracion=duracion, declared_id=id)

    def entregar(
        self,
        dron: object,
        caja: object,
        a: object,
        *,
        inicio: float | None = None,
        duracion: float | None = None,
        id: str | None = None,
    ) -> None:
        """Acción 'entregar': el dron da una caja sostenida a una persona.

        entregar LIBERA el brazo que sostenía la caja (el dominio deriva
        cuál), por eso NO lleva `brazo=`.

        Args:
            a: PERSONA destinataria (id o referencia). Debe estar declarada.

        Alias inglés: deliver.
        """
        contexto = "al encolar la acción entregar"
        drone_id = resolve_drone_id(
            dron, self._state.drones, contexto=contexto
        )
        package_id = resolve_package_id(
            caja, self._state.packages, contexto=contexto
        )
        person_id = self._resolve_destino_persona(a, metodo="entregar")
        command = Deliver(
            drone_id=drone_id, package_id=package_id, person_id=person_id
        )
        self._enqueue(command, inicio=inicio, duracion=duracion, declared_id=id)

    def poner_en(
        self,
        dron: object,
        caja: object,
        transportador: object,
        *,
        inicio: float | None = None,
        duracion: float | None = None,
        id: str | None = None,
    ) -> None:
        """Acción 'poner-caja-en-transportador'.

        poner_en LIBERA el brazo que sostenía la caja, por eso NO lleva
        `brazo=`.

        Alias inglés: load_into.
        """
        contexto = "al encolar la acción poner_en"
        drone_id = resolve_drone_id(
            dron, self._state.drones, contexto=contexto
        )
        package_id = resolve_package_id(
            caja, self._state.packages, contexto=contexto
        )
        transporter_id = resolve_transporter_id(
            transportador, self._state.transporters, contexto=contexto
        )
        command = LoadIntoTransporter(
            drone_id=drone_id,
            package_id=package_id,
            transporter_id=transporter_id,
        )
        self._enqueue(command, inicio=inicio, duracion=duracion, declared_id=id)

    def sacar_de(
        self,
        dron: object,
        caja: object,
        transportador: object,
        brazo: object,
        *,
        inicio: float | None = None,
        duracion: float | None = None,
        id: str | None = None,
    ) -> None:
        """Acción 'coger-caja-del-transportador'.

        sacar_de OCUPA un brazo concreto (la caja acaba sostenida por ese
        brazo), por eso lleva `brazo=`, simétrico con recoger.

        Alias inglés: unload_from.
        """
        contexto = "al encolar la acción sacar_de"
        drone_id = resolve_drone_id(
            dron, self._state.drones, contexto=contexto
        )
        package_id = resolve_package_id(
            caja, self._state.packages, contexto=contexto
        )
        transporter_id = resolve_transporter_id(
            transportador, self._state.transporters, contexto=contexto
        )
        arm_id = self._resolve_brazo(brazo, drone_id)
        command = UnloadFromTransporter(
            drone_id=drone_id,
            arm_id=arm_id,
            package_id=package_id,
            transporter_id=transporter_id,
        )
        self._enqueue(command, inicio=inicio, duracion=duracion, declared_id=id)

    # Alias en inglés: mismo callable, dos nombres (decisión 1).
    move = mover
    grab = recoger
    deliver = entregar
    load_into = poner_en
    unload_from = sacar_de

    # =================================================================
    # Materialización: build / simular / run
    # =================================================================
    def build(self) -> tuple[World, Plan]:
        """Ensambla el escenario en un (World, Plan) del dominio.

        Headless: no abre ventana ni toca pygame. Es el punto que los tests
        usan para validar la construcción.

        Raises:
            MixedTimingError: si el plan mezcla acciones con y sin `inicio=`.
            DuplicateIdError: si dos acciones acaban con el mismo command_id.
            PlanValidationError: si el plan resultante viola una invariante
                estructural del runtime (no debería ocurrir si la fachada
                hizo su trabajo).
        """
        return self._build_world(), self._build_plan()

    def _build_world(self) -> World:
        """Envuelve el estado acumulado en un World inmutable.

        World copia y congela los dicts internamente (MappingProxyType), así
        que pasarle los del estado es seguro: mutaciones posteriores del
        builder no afectan a un World ya construido.
        """
        return World(
            locations=self._state.locations,
            drones=self._state.drones,
            transporters=self._state.transporters,
            packages=self._state.packages,
            persons=self._state.persons,
            contents=self._state.contents,
            costs=self._state.costs,
        )

    def _assign_command_ids(self) -> list[str]:
        """Calcula el command_id final de cada acción encolada.

        Regla: si la acción trae declared_id, se usa verbatim; si no, se
        genera "{verbo}_{posición}" (1-indexado). Verifica unicidad global
        del resultado (cubre la colisión declared-vs-automático).
        """
        final_ids: list[str] = []
        vistos: set[str] = set()
        for i, qa in enumerate(self._state.queued, start=1):
            if qa.declared_id is not None:
                fid = qa.declared_id
            else:
                verbo = _VERB_BY_TYPE[type(qa.command)]
                fid = f"{verbo}_{i}"
            if fid in vistos:
                raise DuplicateIdError(fid, categoria="una acción")
            vistos.add(fid)
            final_ids.append(fid)
        return final_ids

    def _build_plan(self) -> Plan:
        """Convierte las acciones encoladas en un Plan, aplicando la regla
        de tiempos "todo o nada"."""
        queued = self._state.queued
        final_ids = self._assign_command_ids()
        if not queued:
            return Plan(scheduled=())

        con_inicio = [qa.start_time is not None for qa in queued]
        self._check_timing_homogeneo(con_inicio, final_ids)

        scheduled: list[ScheduledCommand] = []
        if all(con_inicio):
            # Modo temporal: tiempos absolutos tal cual los dio el alumno.
            for qa, fid in zip(queued, final_ids):
                assert qa.start_time is not None
                cmd = replace(qa.command, command_id=fid)
                scheduled.append(
                    ScheduledCommand(command=cmd, start_time=qa.start_time)
                )
        else:
            # Modo secuencial: timestamps encadenados sin solape.
            t = 0.0
            for qa, fid in zip(queued, final_ids):
                cmd = replace(qa.command, command_id=fid)
                scheduled.append(ScheduledCommand(command=cmd, start_time=t))
                t += max(qa.duration, _SEQUENTIAL_GAP)
        return Plan(scheduled=tuple(scheduled))

    @staticmethod
    def _check_timing_homogeneo(
        con_inicio: list[bool], final_ids: list[str]
    ) -> None:
        """Verifica que todas las acciones o ninguna llevan `inicio=`."""
        if not con_inicio:
            return
        primero = con_inicio[0]
        for i, tiene in enumerate(con_inicio):
            if tiene != primero:
                # `falta_inicio` es True cuando esta acción NO lleva inicio
                # mientras las anteriores sí (el caso típico: media salida
                # FF pegada tras media OPTIC).
                raise MixedTimingError(final_ids[i], falta_inicio=not tiene)

    def simular(self) -> "RunResult":
        """Ejecuta el plan sobre el world, headless, y devuelve el RunResult.

        Equivale a build() + PlanRunner.execute(). Útil para depurar una
        traducción manual del .pddl ("¿falló mi plan? ¿qué falló?") sin
        abrir ventana, y para tests que aseveran el resultado de ejecución
        sin tocar pygame.display.

        Alias inglés: simulate.
        """
        world, plan = self.build()
        return PlanRunner(world).execute(plan)

    simulate = simular

    def colores_contenido(
        self, *, seed: int = _DEFAULT_PALETTE_SEED
    ) -> dict[str, tuple[int, int, int]]:
        """Paleta de color por tipo de contenido, asignada al crear el escenario.

        Asigna a cada content declarado un color VIVO y DETERMINISTA, recorriendo
        los contenidos por id ORDENADO (así no depende del orden de declaración) y
        repartiendo los tonos por el ÁNGULO ÁUREO: contenidos consecutivos quedan
        lo más separados posible en el círculo de color y no se agrupan por muchos
        que haya (a diferencia del reparto aleatorio anterior, que daba colores
        parecidos ya con pocos tipos). El brillo/saturación alterna por escalones
        como segundo eje distintivo. Las claves van en minúsculas (color_for_content
        normaliza a minúsculas al consultar).

        `seed` desplaza el tono de arranque: misma semilla → mismo reparto
        (reproducible en tests); otra semilla → otro reparto igual de separado.

        La app inyecta esta paleta en el Theme (content_colors) para pintar el
        outline de las cajas según su tipo. Útil también para tests o para
        mostrar la leyenda de colores.

        Alias inglés: content_colors.
        """
        phase = (seed * _GOLDEN_RATIO_CONJUGATE) % 1.0
        return {
            cid.lower(): _spaced_vivid_color(i, phase)
            for i, cid in enumerate(sorted(self._state.contents))
        }

    content_colors = colores_contenido

    def run(self, **app_kwargs: object) -> int:
        """Ejecuta el plan y abre la app interactiva de la el runtime.

        Reutiliza DroneplanVizApp tal cual: no reimplementa loop, HUD ni
        eventos. Pasa el (world, plan) construido al vuelo por el camino de
        entrada `scenario=`.

        El import de DroneplanVizApp es LAZY (dentro del método) a propósito:
        mantiene la fachada importable y testeable en headless sin el extra
        `[app]` (pygame_gui) y evita el import circular
        droneplan_viz -> facade -> app -> droneplan_viz. No subir este import
        al top del módulo.

        Args:
            **app_kwargs: se reenvían a DroneplanVizApp (p. ej. window_size).

        Returns:
            El código de salida de la app (0).
        """
        world, plan = self.build()
        from droneplan_viz_app.app import DroneplanVizApp

        # Paleta de color por contenido (semilla fija → reproducible). Se inyecta
        # en el theme de la app para el outline de las cajas. Si el llamante ya
        # pasó content_colors en app_kwargs, su valor tiene precedencia.
        app_kwargs.setdefault("content_colors", self.colores_contenido())
        app = DroneplanVizApp(scenario=(world, plan), **app_kwargs)  # type: ignore[arg-type]
        return app.run()
