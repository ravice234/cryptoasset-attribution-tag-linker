import time
import yaml
from argparse import ArgumentParser

# setting path
import sys
sys.path.append('attribution_tag_linker/src')

import attribution_tag_linker.src.utils as utils
from knowledge_graph import Actors
from llm_entity_linker import LLMEntityLinker, OpenAIEntityLinker

# The repo https://github.com/ruc-datalab/Unicorn need to be downloaded in the specified (config) location.
# Note: The original code in the repository might not run without (simpel) changes see: https://github.com/ruc-datalab/Unicorn/issues/3
try:
    from unicornlinker import UnicornLinker
except ImportError: 
    pass

CONFIG = "config.yaml"

def main():

    parser = ArgumentParser()
    parser.add_argument('-f', '--filename', type=str, required=True, help='The name of the file')
    parser.add_argument('-m', '--model',  type=str, required=True, help='The name of the filter')
    parser.add_argument('-t', '--template',  type=int, required=False, default=1, help='The name of the filter')
    parser.add_argument('-s', '--shots',  type=int, required=False, default=0, help='The name of the filter')
    args = parser.parse_args()

    filename = args.filename
    model = args.model
    template = args.template
    shots = args.shots

    actors = Actors()

    config = yaml.safe_load(open(CONFIG))
    out_path = config["el_out_path"]
    open_ai_models = config["open_ai_models"]
    unicorn_models = config["unicorn_models"]

    X, y = utils.load_matching_records(filename, actors)

    if model in open_ai_models:
        el = OpenAIEntityLinker(model, template, shots)
    elif model in unicorn_models:
        el = UnicornLinker(model)
    else:
        el = LLMEntityLinker(model, template, shots)

    start_time = time.time()
    results, cost = el.link(X)
    end_time = time.time()
    elapsed_time = end_time - start_time

    precision, recall, f1, accuracy = el.evaluate(results, y)

    if model in open_ai_models:
        cost_s = f"Input Token: {cost[0]}\nOutput Token: {cost[1]}\nPrice: ${cost[2]}\n"
    else:
        cost_s = f"Loading Time: {cost[0]}\nInference Time: {cost[1]}\n{cost[2]} Prompts/s\n"
    summary = f"Model: {model}\nTime: {elapsed_time} seconds" + cost_s + f"Recall: {recall}\nPrecision: {precision}\nF1: {f1}\nAccuracy: {accuracy}\n"
    
    print("\n")
    print(summary)
    print("\n")

    # Logfile with summary and all predictions 
    with open(f"{out_path}logs/linking_{model}_T{template}_S{shots}_run-{end_time}_log.txt", "w") as f:
        f.write(summary+"\n")
        for tag_label, candidates in X:                  
            f.write(f"{str(results[tag_label]==y[tag_label])};{tag_label};{candidates};{y[tag_label]};{results[tag_label]}\n")
    # Benchmark file
    with open(f"{out_path}benchmark.csv",'a') as f:
        f.write(f"{model};{shots};{template};{elapsed_time};{cost[0]};{cost[1]};{cost[2]};{recall};{precision};{f1};{accuracy}\n")

if __name__ == "__main__":
    main()