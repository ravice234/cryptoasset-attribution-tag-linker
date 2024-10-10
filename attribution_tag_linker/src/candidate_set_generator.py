import numpy as np
from nltk.util import ngrams
from rank_bm25 import BM25Okapi
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import attribution_tag_linker.src.utils as utils
import operator

class CandidateSetGenerator:

    def __init__(self, filter, blocker, entities, taxonomy=None):    
        try:
            self.filter = getattr(self, filter)
        except:
            self.filter = None
        try:
            self.blocker = getattr(self, blocker)
        except:
            self.blocker = None
        self.entities = entities
        self.taxonomy = taxonomy

    def generate_candidate_sets(self, tags, k):
        filtered_records = self.filtering(tags)     
        return self.blocking(filtered_records, k) 
        
    def filtering(self, tags):
        if self.filter and self.taxonomy:
            return self.filter(tags)
        else:
            return [(tag[0], list(self.entities.entity_store.keys())) for tag in tags]
        
    def blocking(self, filtered_records, k):
        if self.blocker:
            return self.blocker(filtered_records, k)
        else:
            return filtered_records
        
    def link(self, tags, t):   
        candidates = self.generate_candidate_sets(tags, 1)
        results = {}
        for tag_label, candidate, score in candidates:
            if (score[0]>=t):
                results[tag_label] = candidate[0]
            else:
                results[tag_label] = '-1'    
                                                         
        return results
        
    def evaluate(self, candidate_sets, y):
        matches = 0
        results = [] 
        # Exclude records from evaluation that have ground truth '-1'
        no_actors = operator.countOf(y.values(), '-1')
        if (len(candidate_sets) - no_actors) <= 0:
            return 0

        for tag_label, candidates, _ in candidate_sets:
            found = 0
            for candidate in candidates:
                if y[tag_label] == candidate:
                    found = 1
                    matches += 1
            results.append((found, tag_label, candidates, y[tag_label]))
                    
        recall = matches / (len(candidate_sets) - no_actors)

        return recall, results

    def evaluate_links(self, results, y):   
        keys = list(y.keys())
        y_true = [y[key] for key in keys]
        y_pred = [results[key] for key in keys]
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        return precision, recall, f1, accuracy
    
    def filter_same_concept(self, tags):
        filtered_records = []
        for tag_label, concept_ids in tags:
            if len(concept_ids) > 0:
                candidate_entities = []
                for concept_id in concept_ids:
                    candidate_entities += self.entities.getActorsByConcept(concept_id)
                filtered_records.append((tag_label, candidate_entities)) 
            else:
                filtered_records.append((tag_label, list(self.entities.entity_store.keys()))) 
        return filtered_records        

    def filter_related_concept(self, tags):
        filtered_records = []
        for tag_label, concept_ids in tags:
            if len(concept_ids) > 0:
                candidate_entities = []
                related_concepts = set()
                for concept_id in concept_ids:
                    related_concepts.update([concept_id])
                    related_concepts.update(self.taxonomy.getRelatedConcepts(concept_id))
                for concept_id in related_concepts: 
                    candidate_entities += self.entities.getActorsByConcept(concept_id)
                candidate_entities = list(set(candidate_entities))
                filtered_records.append((tag_label, candidate_entities))
            else:
                filtered_records.append((tag_label, list(self.entities.entity_store.keys())))
        return filtered_records
        
    def blocker_overlap_3(self, filtered_records, k):
        candidate_sets = []
        
        for tag_label, filtered_entities in filtered_records:
            if len(filtered_entities) <= k:
                candidate_sets.append((tag_label, filtered_entities))
                continue
            
            candidate_entities = []
            scores = np.zeros(len(filtered_entities))
            fa = np.array(filtered_entities)

            for i, candidate_entity in enumerate(filtered_entities):
                entity_label = self.entities.getEntityById(candidate_entity).label
                scores[i] += self._ngram_overlap_similarity(tag_label, entity_label)
            
            sorted_ix = np.argsort(scores)[::-1]
            candidate_entities = fa[sorted_ix][:k]
            candidate_scores = scores[sorted_ix][:k]

            candidate_sets.append((tag_label, candidate_entities.tolist(), candidate_scores.tolist())) 

        return candidate_sets

    def blocker_bm25_3(self, filtered_records, k):
        k = k
        idx_map = {}
        idx = 0
        corpus = []
        for id, entity in self.entities.entity_store.items():
            idx_map[id] = idx
            corpus.append(utils.preprocess_string(entity.label))
            idx += 1

        tokenized_corpus = [list(ngrams(doc, n=3)) for doc in corpus]

        bm25 = BM25Okapi(tokenized_corpus)

        candidate_sets = []

        for tag_label, filtered_entities in filtered_records:
        
            if len(filtered_entities) <= k:
                candidate_sets.append((tag_label, filtered_entities))
                continue

            fa = np.array(filtered_entities)
            candidate_entities = []
            doc_ids = [idx_map[id] for id in filtered_entities]
            tokenized_query = ngrams(utils.preprocess_string(tag_label), n=3)
            scores = np.array(bm25.get_batch_scores(tokenized_query, doc_ids))
            sorted_ix = np.argsort(scores)[::-1]
            candidate_entities = fa[sorted_ix][:k]
            candidate_scores = scores[sorted_ix][:k]
            candidate_sets.append((tag_label, candidate_entities.tolist(), candidate_scores.tolist()))       

        return candidate_sets
    
    def _ngram_overlap_similarity(self, s1, s2, n=3):
        A = set(ngrams(utils.preprocess_string(s1), n=n))
        B = set(ngrams(utils.preprocess_string(s2), n=n))
        if len(A) == 0 or len(B) == 0:
            return 0
        return len(A.intersection(B)) / min(len(A), len(B))   