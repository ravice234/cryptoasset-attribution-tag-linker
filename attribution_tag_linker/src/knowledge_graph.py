import attribution_tag_linker.src.utils as utils
import yaml
import json
from yamlinclude import YamlIncludeConstructor

CONFIG = "config.yaml"

class Actors():

    def __init__(self):
        self.config = yaml.safe_load(open(CONFIG))
        self.entity_store = {}
        self.labelEntityMap = {}
        self.uri = self.config["actor_db"]
        self._loadActorInstanceFile()

    def _loadActorInstanceFile(self):

        YamlIncludeConstructor.add_to_loader_class(loader_class=utils.UniqueKeyLoader, base_dir=None)
        contents = yaml.load(open(self.uri, "r"), utils.UniqueKeyLoader)

        self.conceptActorMap = {}
        idx = 0
        for actor in contents["actors"]:
            actor_instance = ActorInstance.from_contents(actor)
            actor_instance.idx = idx
            idx += 1
            self.entity_store.update({actor_instance.id: actor_instance})
            
            for concept_id in actor_instance.categories:
                if concept_id in self.conceptActorMap:
                    actor_set = self.conceptActorMap[concept_id]
                    actor_set.update([actor_instance.id])
                else:
                    actor_set = set([actor_instance.id])
                self.conceptActorMap.update({concept_id: actor_set})

            self.labelEntityMap.update({actor_instance.label: actor_instance.id})

    def getEntityByLabel(self, label):
        try:return self.labelEntityMap[label]
        except KeyError:return None     

    def getEntityById(self, id):
        try:return self.entity_store[id]
        except KeyError:return None 
                
    def getActorsByConcept(self, id):
        try:return self.conceptActorMap[id]
        except KeyError:return set()

    def __str__(self): return str(self.contents)
                   
class ActorInstance():
    def __init__(self, contents):
        self.contents = contents
        self.id = self.contents.get("id", None)
        self._label = self.contents.get("label", None)
        altLabels = self.contents.get("altLabels", [])
        categories = self.contents.get("categories", [])
        self.altLabels = set(altLabels)
        self.categories = set(categories)

    @staticmethod
    def from_contents(contents): return ActorInstance(contents)
    
    @property
    def label(self):return self._label

    @property
    def explicit_fields(self): return {k: v for k, v in self.contents.items()}  

    @property
    def all_fields(self): return {**self.explicit_fields}

    def to_json(self):
        actor = self.all_fields
        return json.dumps(actor, indent=4, sort_keys=True, default=str)
    
    def __str__(self): return str(self.all_fields)



class Taxonomy:

    def __init__(self, key="actor"):

        self.key = key
        config = yaml.safe_load(open(CONFIG))
        self.uri = config["taxonomy_db"]
        self.root_concept = None
        self.leaves = set()
        self.concepts = {}
        self.labels = {}
        self._load_from_local()    

    def _load_from_local(self):
        
        with open(self.uri) as f:
            schema_data = yaml.safe_load(f)

            for key, value in schema_data.items():
                
                id = value["id"]
                label = value.get("prefLabel", id)
                description = value.get("description", None)
                altLabels = value.get("altLabels", None)
                parents = value.get("broader", None)
                children = value.get("narrower", None)
                concept = ConceptNode(self, id, label, description, altLabels=altLabels, parents=parents, children=children)
                self.concepts.update({id: concept})

                if id == self.key:
                    self.root_concept = concept

                self.labels.update({id: id})
                self.labels.update({label: id})

                if children is None:
                    self.leaves.update([concept])

                if altLabels is not None:
                    for altLabel in altLabels:
                        self.labels.update({altLabel: id})

    def getConceptByLabel(self, label):
        try:return self.labels[label]
        except KeyError:return None
        
    def addAltLabel(self, concept_id, label):
        concept = self.concepts[concept_id]
        self.labels.update({label: concept.id})
        concept.altLabels.update([label])

    def strChildren(self, concept, child, str_children="", indentation="\t"):
        indentation += "\t"
        old_indentation = indentation
        if concept in self.leaves:
            str_children += indentation + str(concept)
        else:
            for child in concept.childConcepts:
                str_children += indentation + str(concept) + "<broader>" + "\n"
                str_children = self.strChildren(self, concept, child, str_children, indentation)
                indentation = old_indentation
        
        return str_children
    
    def getRelatedConcepts(self, concept_id):
        concept = self.concepts[concept_id]
        children = self._listChildren(concept)
        parents = self._listParents(concept)

        return children + parents
  
    def _listChildren(self, concept=None, children=None):
        if children is None:
            children = []
        if concept is None:
            concept = self.root_concept
        for child in concept.childConcepts:
            self._listChildren(child, children)
            children.append(child)
        
        return children

    def _listParents(self, concept=None, parents=None):
        if parents is None:
            parents = []
        if concept is None:
            concept = self.root_concept
        for parent in concept.parentConcepts:
            parents.append(parent)
            self._listParents(parent, parents)

        return parents
    
    def __str__(self):
        s = "Taxonomy: ", + self.key + self.uri + "{\n" + self.strChildren(self.root_concept) + "}"
        return s

class ConceptNode:

    def __init__(self, taxonomy, id, label, description, altLabels=None, parents=None, children=None):

        self.taxonomy = taxonomy
        self.id = id
        self.label = label
        self.description = description
        self.altLabels = set()
        self.parents = set()
        self.children = set()

        if altLabels is not None:
            self.altLabels.update(altLabels)

        if parents is not None:
            self.parents.update(parents)

        if children is not None:
            self.children.update(children)

    @property
    def childConcepts(self):

        children = []
        for child_id in self.children:
            child = self.taxonomy.concepts[child_id]
            children.append(child)

        return children
    
    @property
    def parentConcepts(self):

        parents = []
        for parent_id in self.parents:
            parent = self.taxonomy.concepts[parent_id]
            parents.append(parent)

        return parents

    def __str__(self):
        return self.name
    
    def __eq__(self, other):
        if isinstance (other, ConceptNode):
            return self.id == other.id
        else:
            return self.id == other

    def __hash__(self):
        return hash(self.id)

    def __str__(self):
        s = [str(self.id),str(self.label)]
        return "{" + " | ".join(s) + "}"

    def __repr__(self):
        return str(self)