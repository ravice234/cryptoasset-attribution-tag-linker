import torch
import torch.nn as nn
import random
import argparse
import yaml
import time
from transformers import DebertaTokenizer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import sys
CONFIG = "config.yaml"
config = yaml.safe_load(open(CONFIG))
unicorn_path = config["unicorn_path"]
sys.path.append(unicorn_path)

from unicorn.model.encoder import DebertaBaseEncoder
from unicorn.model.matcher import MOEClassifier
from unicorn.model.moe import MoEModule
from unicorn.utils.utils import init_model, make_cuda
from unicorn.dataprocess import predata
from unicorn.utils import param

class UnicornLinker:

    def __init__(self, model):
 
        self.model = model

        args = argparse.Namespace(batch_size=96, c_learning_rate=3e-06, ckpt=model, clip_value=0.01, dataset_path='', 
                                  expertsnum=6, load=True, load_balance=0, log_step=10, max_grad_norm=1.0, max_seq_length=128, 
                                  model='deberta_base', modelname='', num_cls=5, num_data=1000, num_k=2, num_tasks=2, pre_epochs=10, 
                                  pre_log_step=10, pretrain=False, resample=0, scale=20, seed=42, shuffle=0, size_output=768, 
                                  test_metrics='f1', train_seed=42, units=768, wmoe=1)
        
        param.model_root = unicorn_path + "/checkpoint"

        random.seed(args.train_seed)
        torch.manual_seed(args.train_seed)
        if torch.cuda.device_count() > 0:
            torch.cuda.manual_seed_all(args.train_seed)

        encoder = DebertaBaseEncoder()
        classifiers = MOEClassifier(args.units) 
        moelayer = MoEModule(args.size_output,args.units,args.expertsnum,load_balance=args.load_balance)

        start_time = time.perf_counter()
        self.encoder = init_model(args, encoder, restore=args.ckpt+"_"+param.encoder_path)
        self.classifiers = init_model(args, classifiers, restore=args.ckpt+"_"+param.cls_path)
        self.moelayer = init_model(args, moelayer, restore=args.ckpt+"_"+param.moe_path)
        self.tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
        self.args = args
        end_time = time.perf_counter()
        self.loading_time = end_time - start_time

    def link(self, matching_records):

        args = self.args
        test_set = self._process_input(matching_records)
        features = predata.convert_examples_to_features([[x[0]+" [SEP] "+x[1]] for x in test_set[0]], None, args.max_seq_length, self.tokenizer, task_ids=test_set[1])
        test_data_loader = predata.convert_fea_to_tensor(features, args.batch_size, do_train=0)
        
        start_time = time.perf_counter()
        pairwise_predictions = self._predict(test_data_loader)
        end_time = time.perf_counter()
        inference_time = end_time - start_time 

        results = self._process_predictions(pairwise_predictions, matching_records)

        return results, (self.loading_time, inference_time, len(matching_records)/inference_time)

    def evaluate(self, results, y):

        keys = list(y.keys())
        y_true = [y[key] for key in keys]
        y_pred = [results[key] for key in keys]

        # Calculate precision, recall, and F1-score
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)

        return precision, recall, f1, accuracy
    
    def _process_input(self, matching_records):
        
        data = []

        # We give each of the k-candidates per tag label the same task-id. 
        # We use the task-id later to group them and find the most confident prediction per tag label
        task_ids = []

        for task_id, (tag_label, candidates) in enumerate(matching_records):
            for candidate_label, _ in candidates:
                record = []
                record.append(tag_label)
                record.append(candidate_label)
                data.append(record)
                task_ids.append(task_id)

        return (data, task_ids)
    
    def _predict(self,data_loader,index=-1):

        args = self.args
        encoder = self.encoder
        moelayer = self.moelayer
        classifier = self.classifiers

        # set eval state for Dropout and BN layers
        encoder.eval()
        moelayer.eval()
        classifier.eval()

        # init loss and accuracy
        loss = 0
        # set loss function
        criterion = nn.CrossEntropyLoss()
        # evaluate network
        predictions = []
        averagegateweight = torch.Tensor([0 for _ in range(args.expertsnum)]).cuda()
        for (reviews, mask,segment, labels,exm_id,task_id) in data_loader:
            reviews = make_cuda(reviews)
            mask = make_cuda(mask)
            segment = make_cuda(segment)
            labels = make_cuda(labels)
            truelen = torch.sum(mask, dim=1)
            
            with torch.no_grad():
                if args.model in ['distilbert','distilroberta']:
                    feat = encoder(reviews,mask)
                else:
                    feat = encoder(reviews, mask, segment)
                if index==-1:
                    if args.load_balance:
                        moeoutput,balanceloss,_,gateweights = moelayer(feat)
                        averagegateweight += gateweights
                    else:
                        moeoutput,gateweights = moelayer(feat)
                        averagegateweight += gateweights
                else:
                    moeoutput,_ = moelayer(feat,gate_idx=index)
                preds = classifier(moeoutput)

            loss += criterion(preds, labels).item()

            pred_cls = preds.max(1)[1]
            probabilities = nn.functional.softmax(preds, dim=1)

            for ix in range(len(pred_cls)):
                predictions.append((task_id[ix].item(), pred_cls[ix].item(), probabilities[ix][0].item(), probabilities[ix][1].item()))

        return predictions

    def _process_predictions(self, predictions, matching_records):

        grouped_data = {}
        for task_id, cls, prob0, prob1 in predictions:
            if task_id not in grouped_data:
                grouped_data[task_id] = []
            grouped_data[task_id].append((cls, prob1))

        results = {}
        for task_id in sorted(grouped_data.keys()):
            max_prob = -1
            max_index = -1
            for relative_index, (cls, prob1) in enumerate(grouped_data[task_id]):
                if cls == 1 and prob1 > max_prob:
                    max_prob = prob1
                    max_index = relative_index

            if max_index == -1:
                results[matching_records[task_id][0]] = '-1'
            else:
                # [task_id] current record, [0]/[1] tag_label/candidate list, [max_index] matched_candidate, [0]/[1] candidate label / candidate_id
                results[matching_records[task_id][0]] = matching_records[task_id][1][max_index][1]
        
        return results