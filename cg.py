import time
from argparse import ArgumentParser
import yaml

# setting path
import sys
sys.path.append('attribution_tag_linker/src')

import attribution_tag_linker.src.utils as utils
from candidate_set_generator import CandidateSetGenerator
from knowledge_graph import Actors, Taxonomy

CONFIG = "config.yaml"

def main():

    parser = ArgumentParser()
    parser.add_argument('-f', '--filename', type=str, required=True, help='The name of the file')
    parser.add_argument('-ft', '--filter',  type=str, required=True, help='The name of the filter')
    parser.add_argument('-b', '--blocker',  type=str, required=True, help='The name of the blocker')
    parser.add_argument('-k', '--k',  type=int, required=True, help='The name of the blocker')
    args = parser.parse_args()

    filename = args.filename
    vfilter = "filter_"+args.filter
    blocker = "blocker_"+args.blocker
    k = args.k

    actors = Actors()
    taxonomy = Taxonomy()

    config = yaml.safe_load(open(CONFIG))
    out_path = config["cg_out_path"]
    X, y = utils.load_df(filename)

    cg = CandidateSetGenerator(vfilter, blocker, actors, taxonomy)

    start_time = time.time()
    candidate_sets = cg.generate_candidate_sets(X, k)
    end_time = time.time()
    elapsed_time = end_time - start_time

    recall, results = cg.evaluate(candidate_sets, y)

    summary = f"Filter: {vfilter}\nBlocker: {blocker}\nK: {k}\nTime: {elapsed_time} seconds\nRecall: {recall}\n"
    print('\n'+summary+'\n')

    with open(f"{out_path}logs/candidates_k{k}_{vfilter}_{blocker}_run-{end_time}_log.txt", "w") as f:
        f.write(summary+"\n")
        for record in results:
            f.write(str(record) + '\n')

    with open(f"{out_path}benchmark.csv",'a') as f:
        f.write(f"{vfilter};{blocker};{k};{elapsed_time};{recall}\n")

if __name__ == "__main__":
    main()
