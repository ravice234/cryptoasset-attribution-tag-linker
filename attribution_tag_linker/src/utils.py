try:
    from yaml import CSafeLoader as SafeLoader
except ImportError:
    from yaml import SafeLoader as SafeLoader

import string
import pandas as pd
import json
import ast

class UniqueKeyLoader(SafeLoader):
    def construct_mapping(self, node, deep=False):
        mapping = set()
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in mapping:
                raise Exception(f"Duplicate {key!r} key found in YAML.")
            mapping.add(key)
        return super().construct_mapping(node, deep)

def preprocess_string(s):
    return s.strip().lower().translate(str.maketrans('', '', string.punctuation))

def load_df(filename):
    df = pd.read_csv(filename)

    if 'category' in df.keys():
        df['category'] = df['category'].apply(lambda x: ast.literal_eval(x))
    else:
        df['category'] = [[] for _ in range(len(df))]

    tags = list(df[['label', 'category']].itertuples(index=False, name=None))

    ground_truth = {}    
    for _, row in df.iterrows():
        label = row['label']
        actor_list = row['actor']
        ground_truth[label] = actor_list

    return tags, ground_truth

def load_matching_records(filename, actors):

    data = json.load(open(filename))
    ground_truth = {}
    matching_records = []

    for label, candidate_actors, y in data:
        if y == -1:
            ground_truth[label] = '-1'
        else:
            ground_truth[label] = candidate_actors[y]

        candidates = []
        for candidate_id in candidate_actors:
            candidate_label = actors.getEntityById(candidate_id).label
            candidates.append((candidate_label, candidate_id))
        matching_records.append((label, candidates))
    
    return matching_records, ground_truth

def build_matching_records(candidates, actors):

    matching_records = []

    for label, candidate_actors, _ in candidates:

        actor_labels = []
        for candidate_id in candidate_actors:
            candidate_label = actors.getEntityById(candidate_id).label
            actor_labels.append((candidate_label, candidate_id))
        matching_records.append((label, actor_labels))
    
    return matching_records