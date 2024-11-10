import re
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from langchain_ollama import OllamaLLM
import prompt


MODELS = ['mistral:7b','yi:6b','qwen:14b'] # loaded by ollama

prompt_type = 'base'
data_name = 'MMLU' # choose from [MMLU,CosmosQA,Hellaswag,Halu_dialogue,Halu_summarization]
EVAL_PROMPT = prompt.base_prompt

def token_allocation(res):
    # get token_allocation from LLM's response
    dc = {}
    matches = re.findall(r'(A|B|C|D|E|F): (\d+)', res)
    if matches:
        for match in matches:
            letter, value = match
            dc[letter] = value
    else:
        matches = re.findall(r'("A"|"B"|"C"|"D"|"E"|"F"): (\d+)', res)
        if matches:
            for match in matches:
                letter, value = match
                dc[letter[1]] = value
        else: dc = {'A':None, 'B':None, 'C':None, 'D':None, 'E':None, 'F':None}
    return dc

labeled_data_path = 'D:/ollama_fact/data/'
file = labeled_data_path + data_name + 'json'
with open(file, 'r', encoding='utf-8') as data:
    ls_total_data = json.load(data)
    data.close()

for model in MODELS:
    model_storge = model.split(':')[0]+'_'+model.split(':')[1]
    LLM_file = f'D:/ollama_fact/data/black_box_data/LLM_train_res/{data_name}_{model_storge}_{prompt_type}.jsonl'

    with open(LLM_file, 'a') as jsonl_file:
        for dc_data in tqdm(ls_total_data, desc="Processing"):
            dc_result = {}
            dc_result['question']=dc_data['question']
            dc_result['choices']=dc_data['choices']
            dc_result['answer']=dc_data['answer']
            dc_result['model']=model
            input = f'''
                    {EVAL_PROMPT}
                    question:
                    {dc_data['question']}

                    options:
                    "A": {dc_data['choices']["A"]}, "B": {dc_data['choices']["B"]}, "C": {dc_data['choices']["C"]}, "D": {dc_data['choices']["D"]}, "E": {dc_data['choices']["E"]}, "F": {dc_data['choices']["F"]}

                    token_allocation:
                    '''
            res = OllamaLLM(model=model).invoke(input)
            dc_token = token_allocation(res=res)
            dc_result['A']=dc_token.get("A")
            dc_result['B']=dc_token.get("B")
            dc_result['C']=dc_token.get("C")
            dc_result['D']=dc_token.get("D")
            dc_result['E']=dc_token.get("E")
            dc_result['F']=dc_token.get("F")
            dc_result['res']=res
            jsonl_file.write(json.dumps(dc_result) + '\n')