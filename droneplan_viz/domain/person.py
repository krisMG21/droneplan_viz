"""Person: entidad receptora de contenidos en una localización del grafo.

Una persona vive en una Location del mundo y tiene dos colecciones de
contenidos: lo que aún necesita y lo que ya ha recibido. Ambas se
mantienen como tuplas inmutables.

Modelado fiel al PDDL:

    (necesita ?p ?co)  →  Person.needs: tuple[Content, ...]
    (posee ?p ?co)     →  Person.has_received: tuple[Content, ...]

En el dominio, la acción "entregar" produce dos efectos sobre la persona:
añade un content a `posee` y quita el mismo de `necesita`. Nuestro
método receive() refleja exactamente eso.

La meta del plan se cumple cuando todas las personas tienen needs vacío.
Esto se calculará en MetricsTracker (o desde la facade), no aquí.

Sobre duplicados en needs: el PDDL canónico usa hechos booleanos sin
multiplicidad. Aun así, modelamos needs como tupla (no como set) por dos
razones: coherencia con has_received, que sí admite duplicados, y
extensibilidad a variantes futuras del dominio que sí cuenten
ocurrencias. receive() retira una sola ocurrencia, no todas, para
respetar esa semántica.

Como en el resto del dominio, las relaciones se modelan por id
(loc_id de la Location) y la entidad es inmutable: receive() devuelve
una nueva Person sin mutar la original.
"""

from dataclasses import dataclass, field, replace

from droneplan_viz.domain.content import Content


@dataclass(frozen=True, slots=True)
class Person:
    """Receptor de contenidos.

    Atributos:
        id: identificador simbólico único en el mundo.
        position: id de la Location en la que se encuentra. Las personas
            no se mueven en este dominio.
        needs: contenidos que aún le faltan. Se decrementa con receive().
            Cuando queda vacío, la meta sobre esta persona está cumplida.
        has_received: contenidos ya recibidos, en orden de recepción.
            Acumulativo. Admite duplicados.
    """

    id: str
    position: str
    needs: tuple[Content, ...] = field(default=())
    has_received: tuple[Content, ...] = field(default=())

    def receive(self, content: Content) -> "Person":
        """Registra la recepción de un content.

        Efectos paralelos al PDDL:
            - Se añade el content al final de has_received.
            - Si el content estaba en needs, se retira UNA ocurrencia
              (la primera). Si no estaba, needs queda igual; esta
              situación corresponde a una entrega no demandada, que el
              Validator habrá rechazado antes de llegar aquí (regla
              "necesita" como precondition de la acción entregar).

        Devuelve una nueva Person, no muta self. Garantía para Memento:
        los snapshots previos quedan intactos.
        """
        new_received = self.has_received + (content,)
        if content in self.needs:
            i = self.needs.index(content)
            new_needs = self.needs[:i] + self.needs[i + 1:]
        else:
            new_needs = self.needs
        return replace(self, needs=new_needs, has_received=new_received)
