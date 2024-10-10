import os
import templates
import yaml
import time
import attribution_tag_linker.src.utils as utils
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate, ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_community.llms import VLLM
from langchain_openai import ChatOpenAI
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

CONFIG = "config.yaml"

class LLMEntityLinker:

    def __init__(self, model_name, template, shots=0): 
        self.model_name = model_name
        self.template = templates.TEMPLATES[template]
        self.shots = shots
        config_file = yaml.safe_load(open(CONFIG))
        self.config= config_file[model_name]
        self._load_model()

    def link(self, X):
        prompts = self.build_prompts(X)
        self.results, cost = self._batch(prompts, X)
        return self.results, cost
    
    def build_prompts(self, X):
        prompt_template = self._prepare_template()
        prompts = []
        for record in X:
            match_string = f"Attribution Tag Label: {record[0]}"
            for i, (actor_label, _) in enumerate(record[1]):
                match_string += f"\n[{i}] {actor_label}"            
            match_string += f"\n[{i+1}] None of the entities above\n" 
            text_prompt = prompt_template.format(input_string=match_string)
            prompts.append(text_prompt)
        return prompts
    
    def evaluate(self, results, y):
        keys = list(y.keys())
        y_true = [y[key] for key in keys]
        y_pred = [results[key] for key in keys]
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        return precision, recall, f1, accuracy
    
    def _load_model(self):
        model_path = self.config["path"]
        temperature = self.config["temperature"]
        max_new_tokens = self.config["max_new_tokens"]
        vllm_kwargs = self.config["vllm_kwargs"]
        start_time = time.perf_counter()
        self.model = VLLM(model=model_path, temperature=temperature, max_new_tokens=max_new_tokens, vllm_kwargs=vllm_kwargs)
        end_time = time.perf_counter()
        self.loading_time = end_time - start_time
    
    def _prepare_template(self):
        # Using ChatPromptTemplate
        if self.config["chat_template"]:
            # Zero-Shot
            if self.shots == 0:
                if self.template[0] != '': 
                    # Template contians system message
                    chat_template = ChatPromptTemplate.from_messages([("system", self.template[0]), 
                                                                      ("user", self.template[1]),
                                                                      ("assistant", '')])
                else:
                    # Template without system message
                    chat_template = ChatPromptTemplate.from_messages([("user", self.template[1]),
                                                                      ("assistant", '')])
            # Few-Shot
            else:
                example_prompt = ChatPromptTemplate.from_messages([("user", self.template[1]),("assistant", "{output}")])
                few_shot_prompt = FewShotChatMessagePromptTemplate(example_prompt=example_prompt,examples=templates.EXAMPLES[:self.shots])
                # Template contians system message
                if self.template[0] != '':
                    chat_template = ChatPromptTemplate.from_messages([("system", self.template[0]),
                                                                      few_shot_prompt,
                                                                      ("user", self.template[1]),
                                                                      ("assistant", '')])
                # Template without system message
                else:
                    chat_template = ChatPromptTemplate.from_messages([few_shot_prompt,
                                                                      ("user", self.template[1]),
                                                                      ("assistant", '')])
            return chat_template
        
        # Using regular PromptTemplate
        else:           
            s_inst = self.config["start_instruction"]
            e_inst = self.config["end_instruction"]
            # Zero-Shot
            if self.shots == 0:
                template = f"{self.template[0]}\n\n{s_inst}{self.template[1]}{e_inst}"
                prompt_template = PromptTemplate(template=template, input_variables=['input_string'])
            # Few-Shot
            else:    
                example_template = f"{s_inst}\n\n{self.template[1]}\n\n{e_inst}\n\n"+"""{output}\n\n"""
                example_prompt = PromptTemplate(template=example_template, input_variables=['input_string', 'output'])
                prompt_template = FewShotPromptTemplate(prefix=f"{self.template[0]}\n\n",
                                                        example_prompt=example_prompt,
                                                        examples=templates.EXAMPLES[:self.shots],
                                                        suffix=f"{s_inst}{self.template[1]}{e_inst}",
                                                        input_variables=['input_string'])
            return prompt_template

    def _batch(self, prompts, X):
        n_prompts = len(prompts)
        start_time = time.perf_counter()
        predictions = self.model.batch(prompts)
        end_time = time.perf_counter()
        inference_time = end_time - start_time
        prompt_throuput = n_prompts / inference_time

        results = {}

        for ix, prediction in enumerate(predictions):
            pred_ix = utils.preprocess_string(prediction)               
            if pred_ix in ['0', '1', '2', '3', '4']:
                # X[ix][0] -> attribution tag label of current record
                # X[ix][1] -> candidate list of current record
                # [pred_ix][1] -> actor id of predicted candidate
                results[X[ix][0]] = X[ix][1][int(pred_ix)][1]
            else:
                results[X[ix][0]] = '-1'

        return results, (self.loading_time, inference_time, prompt_throuput)
    

class OpenAIEntityLinker(LLMEntityLinker):

    def __init__(self, model, template, shots=0):
        self.api_key = os.environ.get("OPENAI_API_KEY_ATL")
        super().__init__(model, template, shots)
    
    def _load_model(self):
        temperature = self.config["temperature"]
        max_tokens = self.config["max_tokens"]
        self.model = ChatOpenAI(model_name=self.model_name, temperature=temperature, openai_api_key=self.api_key, max_tokens=max_tokens)

    def _batch(self, prompts, X):
        token_count_input = 0
        token_count_output = 0
        results = {}
        config = self.config["config"]

        preds = self.model.batch(prompts, config=config)

        for ix, pred in enumerate(preds):
            token_count_input += pred.response_metadata['token_usage']['prompt_tokens']
            token_count_output += pred.response_metadata['token_usage']['completion_tokens']
            pred_ix = utils.preprocess_string(pred.content)
                
            if pred_ix in ['0', '1', '2', '3', '4']:
                # X[ix][0] -> attribution tag label of current record
                # X[ix][1] -> candidate list of current record
                # [pred_ix][1] -> actor id of predicted candidate
                results[X[ix][0]] = X[ix][1][int(pred_ix)][1]
            else:
                results[X[ix][0]] = '-1'

        # Prices per 1000 I/O tokens in $USD
        prices = self.config["prices"]
        dollar_cost = (token_count_input/1000)*prices[0] + (token_count_output/1000)*prices[1]

        return results, (token_count_input, token_count_output, dollar_cost)