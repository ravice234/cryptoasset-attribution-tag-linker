import time
from argparse import ArgumentParser
import yaml

# setting path
import sys
sys.path.append('attribution_tag_linker/src')

import attribution_tag_linker.src.utils as utils
from knowledge_graph import Actors, Taxonomy
from candidate_set_generator import CandidateSetGenerator
from llm_entity_linker import LLMEntityLinker, OpenAIEntityLinker

# The repo https://github.com/ruc-datalab/Unicorn need to be downloaded in the specified (config) location.
# Note: The original code in the repository might not run without (simpel) changes see: https://github.com/ruc-datalab/Unicorn/issues/3
try:
    from unicornlinker import UnicornLinker
except ImportError: 
    pass

CONFIG = "config.yaml"

def main():

    start_time = time.time()

    parser = ArgumentParser()
    parser.add_argument('-f', '--filename', type=str, required=True, help='The name of the file')
    parser.add_argument('-ft', '--filter', type=str, required=True, help='The name of the filter')
    parser.add_argument('-b', '--blocker', type=str, required=True, help='The name of the blocker')
    parser.add_argument('-k', '--k',  type=int, required=True, help='The name of the blocker')
    parser.add_argument('-m', '--model', type=str, required=True, help='The name of the filter')
    parser.add_argument('-t', '--template', type=int, required=False, default=1, help='The name of the filter')
    parser.add_argument('-s', '--shots', type=int, required=False, default=0, help='The name of the filter')
    args = parser.parse_args()

    args = parser.parse_args()

    filename = args.filename
    vfilter = "filter_"+args.filter
    blocker = "blocker_"+args.blocker
    k = args.k
    model = args.model
    template = args.template
    shots = args.shots

    actors = Actors()
    taxonomy = Taxonomy()

    config = yaml.safe_load(open(CONFIG))
    out_path = config["e2e_out_path"]
    open_ai_models = config["open_ai_models"]
    unicorn_models = config["unicorn_models"]

    tags, y = utils.load_df(filename)

    cg = CandidateSetGenerator(vfilter, blocker, actors, taxonomy)

    if model == args.blocker:
        results = cg.link(tags, 15.7238)
        X = [(k, v) for k, v in results.items()]
        precision, recall, f1, accuracy = cg.evaluate_links(results, y)
        cg_recall = recall
        cost = (0, 0, 0)  
    else:
        candidate_sets = cg.generate_candidate_sets(tags, k)
        cg_recall, _ = cg.evaluate(candidate_sets, y)
        if model in open_ai_models:
            el = OpenAIEntityLinker(model, template, shots)
        elif model in unicorn_models:
            el = UnicornLinker(model)
        else:
            el = LLMEntityLinker(model, template, shots)

        X = utils.build_matching_records(candidate_sets, actors)   
        results, cost = el.link(X)

        precision, recall, f1, accuracy = el.evaluate(results, y)
    
    end_time = time.time()
    elapsed_time = end_time - start_time  

    if model in open_ai_models:
        cost_s = f"Input Token: {cost[0]}\nOutput Token: {cost[1]}\nPrice: ${cost[2]}\n"
    else:
        cost_s = f"Loading Time: {cost[0]}\nInference Time: {cost[1]}\n{cost[2]} Prompts/s\n"  
    summary = f"File: {filename}\nShots: {shots}\nTime: {elapsed_time} seconds\nFilter: {vfilter}\nBlocker: {blocker}\nK: {k}\nCG Recall {cg_recall}\nModel: {model}\n" + cost_s + f"Recall: {recall}\nPrecision: {precision}\nF1: {f1}\nAccuracy: {accuracy}\n"
    print("\n"+summary+"\n")

    # Logfile with summary and all predictions
    with open(f"{out_path}logs/{utils.preprocess_string(filename)}_{model}_{shots}-shot_run-{end_time}_log.txt", "w") as f:
        f.write(summary+"\n")            
        for tag_label, candidates in X:                  
            f.write(f"{str(results[tag_label]==y[tag_label])};{tag_label};{candidates};{y[tag_label]};{results[tag_label]}\n")
    # Benchmark file
    with open(f"{out_path}{utils.preprocess_string(filename)}_benchmark.csv",'a') as f:
        f.write(f"{filename};{vfilter};{blocker};{k};{cg_recall};{model};{shots};{template};{elapsed_time};{cost[0]};{cost[1]};{cost[2]};{recall};{precision};{f1};{accuracy}\n")

if __name__ == "__main__":
    main()