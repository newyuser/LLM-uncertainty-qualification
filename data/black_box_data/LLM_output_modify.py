import re
import os
import json
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from langchain_ollama import OllamaLLM


def jsonl_to_df(file_name):
    file_path = f'D:/ollama_fact/data/black_box_data/LLM_train_res/{file_name}'
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            try: 
                data.append(json.loads(line))
            except json.decoder.JSONDecodeError as e:
                print(line)
    df = pd.DataFrame(data)
    return df

folder_path = 'D:/ollama_fact/data/black_box_data/LLM_train_res'
data_name = 'MMLU'
jsonl_files = glob.glob(os.path.join(folder_path, f'{data_name}*.jsonl'))
file_names = [os.path.basename(file) for file in jsonl_files]
df_list = [jsonl_to_df(file) for file in file_names]
df = pd.concat(df_list, ignore_index=True)
miss = df[df[['A', 'B', 'C', 'D']].isna().all(axis=1)]

def token_allocation(res):
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
        else: 
            matches = re.findall(r"('A'|'B'|'C'|'D'|'E'|'F'): (\d+)", res)
            if matches:
                for match in matches:
                    letter, value = match
                    dc[letter[1]] = value
            else: dc = {'A':None, 'B':None, 'C':None, 'D':None, 'E':None, 'F':None}
    return dc

EVAL_PROMPT = '''
You will get the output of the LLM, which should assign tokens to the options, but the format is not uniform.

Then you will be given options, please assign tokens to each option according to the options using the following format
{
  'A': number of tokens,
  'B': number of tokens,
  'C': number of tokens,
  'D': number of tokens,
  'E': number of tokens,
  'F': number of tokens
}

Restrictions:
The key of the output dict can only be 'A', 'B', 'C', 'D', 'E', 'F' and the value can only be numbers. Only out jsonl dict without any other things.

Example 1:
output:{"APP": 15,"PS1": 12,"PS2": 10,"APOE": 60,"I don't know": 3,"None of the above": 0}
options:{'A': 'APP', 'B': 'PS1', 'C': 'PS2', 'D': 'APOE', 'E': "I don't know", 'F': 'None of the above'}
your output:{'A': 15, 'B': 12, 'C': 10, 'D': 60, 'E': 3, 'F': 0}
Example 2:
output:{"A": [50],"C": [40],"B": [0], "D": [0],"E": [10],"F": [0]}
your output:{'A': 50, 'B': 40, 'C': 0, 'D': 0, 'E': 10, 'F': 0}
'''
 
for idx, row in tqdm(miss.iterrows(), desc="Processing"):
    input = f'''
            {EVAL_PROMPT}
            
            output:
            {row['res']}

            options:
            "A": {row['choices']["A"]}, "B": {row['choices']["B"]}, "C": {row['choices']["C"]}, "D": {row['choices']["D"]}, "E": {row['choices']["E"]}, "F": {row['choices']["F"]}
            
            modified output:
            '''
    res = OllamaLLM(model='yi:6b').invoke(input)
    dc_token = token_allocation(res=res)
    df.loc[idx, 'A']=dc_token.get('A')
    df.loc[idx, 'B']=dc_token.get('B')
    df.loc[idx, 'C']=dc_token.get('C')
    df.loc[idx, 'D']=dc_token.get('D')
    df.loc[idx, 'E']=dc_token.get('E')
    df.loc[idx, 'F']=dc_token.get('F')
    df.loc[idx, 'modified_res']=res

jsonl_lines = df.to_json(orient='records', lines=True)

with open(f'D:/LLM-uncertainty-qualification/data/black_box_data/LLM_train_res/{file}.jsonl'', 'w') as f:
    f.write(jsonl_lines)

