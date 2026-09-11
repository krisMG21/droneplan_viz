"""Excepciones de CONSTRUCCIÓN de la fachada.

Esta es la taxonomía de errores que la fachada lanza cuando el alumno
declara o referencia mal el escenario: una localización que no existe, un
brazo que el dron no tiene, un id duplicado, un plan con tiempos
inconsistentes, etc. Son errores DIDÁCTICOS: su valor está en el mensaje,
que siempre cita la llamada concreta que arregla el problema.

Frontera explícita con la validación física (decisión 8 del diseño):

    - La fachada valida CONSTRUCCIÓN: que las referencias apunten a algo
      declarado, que los ids sean únicos, que el plan no mezcle modos
      temporales. Todo eso se detecta ANTES de ejecutar, mientras el
      alumno teclea el escenario, y produce las excepciones de este
      módulo.

    - La fachada NO valida FÍSICA: que un brazo esté libre, que el dron y
      el paquete estén co-localizados, que el transportador tenga
      capacidad. Eso lo sigue comprobando el PlanRunner durante la
      ejecución (drone a ERROR, RunResult.failures), con sus mensajes
      explicativos. La fachada no lo reimplementa.

    El caso límite de UnknownArmError ilustra la frontera: comprobamos que
    el brazo EXISTE en el dron (construcción), pero no que esté LIBRE
    (física, del runner). Un dron explorador con arms=[] que intenta
    recoger una caja cae aquí con un mensaje claro; un dron con el brazo
    ocupado lo rechaza el runner, no la fachada.

Diseño de las clases:

    - Todas heredan de FacadeError, así un consumidor puede capturar
      cualquier error de construcción con un solo `except FacadeError`.
    - Cada clase recibe DATOS ESTRUCTURADOS (el id ofensor, el dron, los
      brazos disponibles…) y construye el mensaje en su __init__. Los
      datos quedan accesibles como atributos para que los tests aserten
      sobre ellos sin tener que hacer match de substrings frágiles.
    - El módulo no importa nada del dominio ni de PyGame: es Python puro,
      seguro de importar en cualquier entorno (headless incluido) y sin
      riesgo de import circular cuando droneplan_viz/__init__.py lo
      reexporte.

Sobre el énfasis "ANTES": la construcción de la fachada es eager y exige
orden topológico (primero infraestructura, luego entidades, luego
acciones), igual que el alumno lee la sección :init de su .pddl. Por eso
los mensajes de "no declarado" no dicen solo "no existe", sino "declárala
ANTES de referenciarla aquí": le indican CUÁNDO debe declararla, no solo
QUE falta.
"""
from __future__ import annotations


class FacadeError(Exception):
    """Base de todos los errores de construcción de la fachada.

    Capturar `FacadeError` atrapa cualquier problema detectado mientras se
    declara el escenario (referencias rotas, ids duplicados, plan mal
    temporizado). NO atrapa los fallos físicos del plan: esos viven en
    RunResult.failures tras ejecutar con el PlanRunner, no se lanzan como
    excepción.
    """


# ---------------------------------------------------------------------------
# Referencias a entidades no declaradas
#
# Una familia por tipo de entidad. Cada mensaje cita el método del builder
# que la declara, para que el alumno sepa exactamente qué teclear. El
# parámetro `contexto` opcional permite a quien lanza el error precisar
# DÓNDE se usó la referencia rota (p. ej. "al declarar el dron 'dron1'"),
# sustituyendo el genérico "aquí" por algo concreto.
# ---------------------------------------------------------------------------


class UnknownLocationError(FacadeError):
    """Se referenció una localización que no se había declarado.

    Atributos:
        loc_id: id de la localización inexistente.
        contexto: descripción opcional de dónde se usó la referencia.
    """

    def __init__(self, loc_id: str, *, contexto: str | None = None) -> None:
        self.loc_id = loc_id
        self.contexto = contexto
        donde = contexto if contexto is not None else "aquí"
        super().__init__(
            f"la localización '{loc_id}' no existe en este momento; "
            f"declárala con viz.world.location('{loc_id}') ANTES de "
            f"referenciarla {donde}"
        )


class UnknownContentError(FacadeError):
    """Se referenció un contenido (categoría de carga) no declarado.

    Atributos:
        content_id: id del contenido inexistente.
        contexto: descripción opcional de dónde se usó la referencia.
    """

    def __init__(self, content_id: str, *, contexto: str | None = None) -> None:
        self.content_id = content_id
        self.contexto = contexto
        donde = contexto if contexto is not None else "aquí"
        super().__init__(
            f"el contenido '{content_id}' no existe en este momento; "
            f"decláralo con viz.world.content('{content_id}') ANTES de "
            f"referenciarlo {donde}"
        )


class UnknownPersonError(FacadeError):
    """Se referenció una persona no declarada.

    Atributos:
        person_id: id de la persona inexistente.
        contexto: descripción opcional de dónde se usó la referencia.
    """

    def __init__(self, person_id: str, *, contexto: str | None = None) -> None:
        self.person_id = person_id
        self.contexto = contexto
        donde = contexto if contexto is not None else "aquí"
        super().__init__(
            f"la persona '{person_id}' no existe en este momento; "
            f"declárala con viz.world.person('{person_id}', at=...) ANTES "
            f"de referenciarla {donde}"
        )


class UnknownPackageError(FacadeError):
    """Se referenció un paquete (caja) no declarado.

    Atributos:
        package_id: id del paquete inexistente.
        contexto: descripción opcional de dónde se usó la referencia.
    """

    def __init__(self, package_id: str, *, contexto: str | None = None) -> None:
        self.package_id = package_id
        self.contexto = contexto
        donde = contexto if contexto is not None else "aquí"
        super().__init__(
            f"la caja '{package_id}' no existe en este momento; "
            f"declárala con viz.world.package('{package_id}', contiene=..., "
            f"at=...) ANTES de referenciarla {donde}"
        )


class UnknownDroneError(FacadeError):
    """Se referenció un dron no declarado.

    Atributos:
        drone_id: id del dron inexistente.
        contexto: descripción opcional de dónde se usó la referencia.
    """

    def __init__(self, drone_id: str, *, contexto: str | None = None) -> None:
        self.drone_id = drone_id
        self.contexto = contexto
        donde = contexto if contexto is not None else "aquí"
        super().__init__(
            f"el dron '{drone_id}' no existe en este momento; "
            f"decláralo con viz.agents.drone('{drone_id}', at=...) ANTES de "
            f"referenciarlo {donde}"
        )


class UnknownTransporterError(FacadeError):
    """Se referenció un transportador no declarado.

    Atributos:
        transporter_id: id del transportador inexistente.
        contexto: descripción opcional de dónde se usó la referencia.
    """

    def __init__(
        self, transporter_id: str, *, contexto: str | None = None
    ) -> None:
        self.transporter_id = transporter_id
        self.contexto = contexto
        donde = contexto if contexto is not None else "aquí"
        super().__init__(
            f"el transportador '{transporter_id}' no existe en este "
            f"momento; decláralo con viz.agents.transporter("
            f"'{transporter_id}', capacidad=..., at=...) ANTES de "
            f"referenciarlo {donde}"
        )


# ---------------------------------------------------------------------------
# Brazo inexistente en un dron
#
# Solo valida EXISTENCIA del brazo en el dron, no su ocupación (eso es
# física del runner). Distingue dos casos para que el mensaje sea útil:
#   - el dron tiene brazos, pero ninguno se llama así  -> lístalos;
#   - el dron no tiene brazos (explorador, arms=[])     -> explícalo y
#     sugiere declararlo con brazos.
# ---------------------------------------------------------------------------


class UnknownArmError(FacadeError):
    """Se usó un brazo que el dron indicado no tiene.

    Atributos:
        arm_id: nombre del brazo solicitado.
        drone_id: dron sobre el que se pidió el brazo.
        brazos_disponibles: tupla con los nombres de los brazos que el
            dron sí tiene. Vacía si es un dron explorador (arms=[]).
    """

    def __init__(
        self,
        arm_id: str,
        drone_id: str,
        *,
        brazos_disponibles: tuple[str, ...] = (),
    ) -> None:
        self.arm_id = arm_id
        self.drone_id = drone_id
        self.brazos_disponibles = tuple(brazos_disponibles)
        if self.brazos_disponibles:
            disponibles = ", ".join(f"'{b}'" for b in self.brazos_disponibles)
            mensaje = (
                f"el dron '{drone_id}' no tiene un brazo '{arm_id}'; "
                f"sus brazos son {disponibles}"
            )
        else:
            mensaje = (
                f"el dron '{drone_id}' no tiene brazos (es un dron "
                f"explorador) y no puede recoger ni sacar cajas; decláralo "
                f"con brazos: viz.agents.drone('{drone_id}', at=..., "
                f"arms=['izq', 'der'])"
            )
        super().__init__(mensaje)


# ---------------------------------------------------------------------------
# Referencia del tipo equivocado (guarda del solape de `a`)
#
# `a` es destino-localización en mover y persona en entregar. Si el alumno
# cruza los cables (mover hacia una persona, o entregar a una localización)
# y el id SÍ existe pero como entidad de otra clase, este error lo dice con
# precisión en vez de un genérico "no existe".
# ---------------------------------------------------------------------------


class WrongReferenceKindError(FacadeError):
    """Se pasó una entidad declarada, pero de un tipo distinto al esperado.

    Atributos:
        ref_id: id de la entidad recibida.
        metodo: método de la fachada donde ocurrió (p. ej. "mover").
        parametro: nombre del parámetro implicado (p. ej. "a").
        esperado: descripción del tipo esperado (p. ej. "una localización").
        recibido: descripción del tipo recibido (p. ej. "la persona").
    """

    def __init__(
        self,
        ref_id: str,
        *,
        metodo: str,
        parametro: str,
        esperado: str,
        recibido: str,
    ) -> None:
        self.ref_id = ref_id
        self.metodo = metodo
        self.parametro = parametro
        self.esperado = esperado
        self.recibido = recibido
        super().__init__(
            f"'{metodo}' espera {esperado} en '{parametro}'; recibió "
            f"{recibido} '{ref_id}'"
        )


# ---------------------------------------------------------------------------
# Identificador duplicado
#
# Cubre dos casos con la misma semántica ("este id ya está usado"):
#   - declarar dos veces una entidad con el mismo id;
#   - pasar id= explícito a dos acciones distintas.
# El campo `categoria` da el matiz en el mensaje.
# ---------------------------------------------------------------------------


class DuplicateIdError(FacadeError):
    """Se intentó usar un identificador que ya estaba ocupado.

    Atributos:
        id_repetido: el identificador duplicado.
        categoria: descripción de qué clase de elemento lo usa, en
            singular y con artículo (p. ej. "una localización",
            "un dron", "una acción"). Aparece tal cual en el
            mensaje.
    """

    def __init__(self, id_repetido: str, *, categoria: str) -> None:
        self.id_repetido = id_repetido
        self.categoria = categoria
        super().__init__(
            f"ya existe {categoria} con id '{id_repetido}'; los "
            f"identificadores deben ser únicos en el escenario"
        )


# ---------------------------------------------------------------------------
# Plan mixto temporal/secuencial
#
# La regla "todo o nada" sobre `inicio`: o todas las acciones llevan
# inicio= (plan temporal, parte 3 OPTIC) o ninguna lo lleva (plan
# secuencial, partes 1-2 FF). Mezclar suele venir de copiar media salida
# FF y media OPTIC, así que el mensaje nombra la acción culpable y la
# regla.
# ---------------------------------------------------------------------------


class MixedTimingError(FacadeError):
    """El plan mezcla acciones con `inicio=` y acciones sin él.

    Atributos:
        accion_id: id de la primera acción que rompe la homogeneidad.
        falta_inicio: True si la acción culpable NO lleva inicio= mientras
            las anteriores sí; False en el caso simétrico.
    """

    def __init__(self, accion_id: str, *, falta_inicio: bool) -> None:
        self.accion_id = accion_id
        self.falta_inicio = falta_inicio
        if falta_inicio:
            detalle = (
                f"la acción '{accion_id}' no lleva 'inicio=' pero las "
                f"acciones anteriores sí"
            )
        else:
            detalle = (
                f"la acción '{accion_id}' lleva 'inicio=' pero las acciones "
                f"anteriores no"
            )
        super().__init__(
            f"detectado plan mixto temporal-secuencial: {detalle}. Define "
            f"'inicio=' en todas las acciones o en ninguna"
        )


__all__ = [
    "FacadeError",
    "UnknownLocationError",
    "UnknownContentError",
    "UnknownPersonError",
    "UnknownPackageError",
    "UnknownDroneError",
    "UnknownTransporterError",
    "UnknownArmError",
    "WrongReferenceKindError",
    "DuplicateIdError",
    "MixedTimingError",
]
