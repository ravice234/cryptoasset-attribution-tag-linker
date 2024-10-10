from transformers import DebertaTokenizer
import torch
import yaml
import random
import argparse

import sys
CONFIG = "config.yaml"
config = yaml.safe_load(open(CONFIG))
unicorn_path = config["unicorn_path"]
sys.path.append(unicorn_path)
sys.path.append('attribution_tag_linker/src')
import attribution_tag_linker.src.utils as utils
from candidate_set_generator import CandidateGenerator
from knowledge_graph import Actors, Taxonomy
from unicorn.model.encoder import DebertaBaseEncoder
from unicorn.model.matcher import Classifier, MOEClassifier
from unicorn.model import moe
from unicorn.trainer import pretrain
from unicorn.utils.utils import init_model
from unicorn.dataprocess import predata
from unicorn.utils import param

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)


def process_input(file):

    actors = Actors()
    taxonomy = Taxonomy()
    vfilter = "related_concept"
    blocker = "bm25_3"
    tags, y = utils.load_df(file)
    cg = CandidateGenerator(vfilter, blocker, actors, taxonomy)
    candidates = cg.generate_candidate_sets(tags, 1)

    data = []
    
    for tag_label, candidate_actors, _ in candidates:
        for candidate_id in candidate_actors:
            candidate_label = actors.getEntityById(candidate_id).label
            record = []
            record.append(tag_label)
            record.append(candidate_label)
            record.append(int(y[tag_label] == candidate_id))
            data.append(record)

    return data

def main():

    args = argparse.Namespace(batch_size=96, c_learning_rate=3e-06, ckpt='UnicornPlus', clip_value=0.01, dataset_path='', 
                                expertsnum=6, load=True, load_balance=0, log_step=10, max_grad_norm=1.0, max_seq_length=128, 
                                model='deberta_base', modelname='UnicornPlusFT2', num_cls=5, num_data=1000, num_k=2, num_tasks=2, pre_epochs=10, 
                                pre_log_step=10, pretrain=False, resample=0, scale=20, seed=42, shuffle=0, size_output=768, 
                                test_metrics='f1', train_seed=42, units=768, wmoe=1)
    
    param.model_root = unicorn_path + "/checkpoint"
        
    # argument setting
    print("=== Argument Setting ===")
    print("encoder: " + str(args.model))
    print("max_seq_length: " + str(args.max_seq_length))
    print("batch_size: " + str(args.batch_size))
    set_seed(args.train_seed)

    if args.model == 'deberta_base':
        tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
        encoder = DebertaBaseEncoder()
            
    wmoe = args.wmoe
    if wmoe:
        classifiers = MOEClassifier(args.units) 
    else:
        classifiers = Classifier()
            
    if wmoe:
        exp = args.expertsnum
        moelayer = moe.MoEModule(args.size_output,args.units,exp,load_balance=args.load_balance)
    
    if args.load:
        encoder = init_model(args, encoder, restore=args.ckpt+"_"+param.encoder_path)
        classifiers = init_model(args, classifiers, restore=args.ckpt+"_"+param.cls_path)
        if wmoe:
            moelayer = init_model(args, moelayer, restore=args.ckpt+"_"+param.moe_path)
    else:
        encoder = init_model(args, encoder)
        classifiers = init_model(args, classifiers)
        if wmoe:
            moelayer = init_model(args, moelayer)

    train_sets = []
    train_sets.append(process_input("data/train2.csv"))
    valid_sets = []
    valid_sets.append(process_input("data/validation2.csv"))
    train_metrics = ['f1' for i in range(0, len(train_sets))]

    train_data_loaders = []
    valid_data_loaders = []

    for i in range(len(train_sets)):
        fea = predata.convert_examples_to_features([ [x[0]+" [SEP] "+x[1]] for x in train_sets[i] ], [int(x[2]) for x in train_sets[i]], args.max_seq_length, tokenizer)
        print("train fea: ",len(fea))
        train_data_loaders.append(predata.convert_fea_to_tensor(fea, args.batch_size, do_train=0))
    for i in range(len(valid_sets)):
        fea = predata.convert_examples_to_features([ [x[0]+" [SEP] "+x[1]] for x in valid_sets[i] ], [int(x[2]) for x in valid_sets[i]], args.max_seq_length, tokenizer)
        print("valid fea: ",len(fea))
        valid_data_loaders.append(predata.convert_fea_to_tensor(fea, args.batch_size, do_train=0))
    
    print("train datasets num: ",len(train_data_loaders[0]))
    print("valid datasets num: ",len(valid_data_loaders[0]))
    encoder, moelayer, classifiers = pretrain.train_moe(args, encoder, moelayer, classifiers, train_data_loaders, valid_data_loaders, train_metrics)
            
if __name__ == '__main__':
    main()
