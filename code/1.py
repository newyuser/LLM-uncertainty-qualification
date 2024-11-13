import pickle 
import json
import os
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import argparse


def get_SPU(args):
    """
    0~5:softmax score for option
    6:variance
    7~12:prediction(one hot)
    13:uncertainty
    14:type
    """
    SPU = {}
    for model in args.model:
        SPU[model] = 1
    return SPU

def plot_accuracy(results, title, save):
    plt.figure()
    xs, ys, zs = results[:, 0], results[:, 1], results[:, 2]
    plt.plot(xs, ys, '-x')
    # zip joins x and y coordinates in pairs
    for x, y, z in zip(xs,ys, zs):
        label = "{:.0f}%".format(100 * z)
        plt.annotate(label, # this is the text
                     (x,y), # these are the coordinates to position the label
                     textcoords="offset points", # how to position the text
                     xytext=(0,-10), # distance from text to points (x,y)
                     ha='center') 
    plt.xlabel('uncertainty threshold')
    plt.ylabel('accuracy')
    if save:
        file = f'D:/LLM-uncertainty-qualification/figure/logists_token_model/{title}'
        plt.savefig(file + '.png')
        plt.savefig(file + '.pdf')
    plt.title(title)
    plt.show()

def uncertainty_acc_result(title, PUT_data, cal = False, plot_results = True, save = save):
    results = []
    for i in range(10):
        threshold = (i + 1) / 10
        p = PUT_data[:, -3]
        u = PUT_data[:, -2]
        t = PUT_data[:, -1]
        if cal: 
            # bucket algorithm
            quantiles = np.percentile(u, np.arange(10, 101, 10))
            new_u = np.zeros(len(u))
            for i, q in enumerate(quantiles):
                if i == 0:
                    new_u[u <= q] = i * 0.1
                else:
                    new_u[(u > quantiles[i - 1]) & (u <= q)] = i * 0.1
            mask = new_u <= threshold 
        else: mask = u <= threshold 
        acc = (p == t)[mask].sum() / max(1, sum(mask))
        perc_data = mask.mean()
        results.append([threshold, acc, perc_data])
    results = np.array(results)
    results = results[results[:,2]>0]
    if plot_results:
        plot_accuracy(results, title, save)
    return results

def black_box_method(args):
    file_path = args.tokens_data_dir + args.data_name + '.jsonl'
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            try: 
                data.append(json.loads(line))
            except json.decoder.JSONDecodeError as e:
                print(line)
    df = pd.DataFrame(data)
    models = df['model'].unique()
    Token = {}
    for model in models:
        df_model = df[df['model']==model]
        truth = np.array(df_model['answer'].map(args.mapping))
        token = np.array(df_model[['A', 'B', 'C', 'D', 'E', 'F']])
        model = model.split(':')[0]+'_'+model.split(':')[1]
        Token[model] = np.column_stack((token / 100, OneHotEncoder(sparse_output=False).fit_transform(np.array(subject).reshape(-1, 1))))
        pred = np.argmax(token[:, :4], axis=1)
        uncertany = 1 - np.max(token[:, :4], axis=1) / 100
        put = np.column_stack((pred,uncertany,truth))
        uncertainty_acc_result(f'token {model}', put, cal = False, plot_results = True, save = True)

def which_box_method(args):
choice = {
      "A": "Wrong, Wrong",
      "B": "Wrong, Not wrong",
      "C": "Not wrong, Wrong",
      "D": "Not wrong, Not wrong",
      "E": "I don't know",
      "F": "None of the above"
    }
ls_id = []
ls_answer = []
labeled_data_path = 'D:/LLM-uncertainty-qualification/data/'
MMLU_file = labeled_data_path + 'MMLU.json'
with open(MMLU_file, 'r', encoding='utf-8') as data:
    ls_total_data = json.load(data)
for i in range(len(ls_total_data)):
    if ls_total_data[i]['choices'] != choice: 
        ls_id.append(i)
        ls_answer.append(args.mapping[ls_total_data[i]['answer']])
logits_truth = np.array(ls_answer)
def softmax_2d(array):
    max_per_row = np.max(array, axis=1, keepdims=True)
    exp_array = np.exp(array - max_per_row)
    sum_per_row = np.sum(exp_array, axis=1, keepdims=True)
    softmax_result = exp_array / sum_per_row
    return softmax_result

Logits = {}
models = [model.split(':')[0]+'_'+model.split(':')[1] for model in models]
model_map = dict(zip(args.sub_list, models))
for model in args.sub_list:
    logits_file = os.path.join(args.logits_data_dir, model+"_mmlu_10k_base_icl1.pkl")
    with open(logits_file, 'rb') as f:
        logits_data = pickle.load(f)
        logits_w_data = [logits_data[i] for i in ls_id]
    model = model_map[model]
    Logits[model] = softmax_2d(np.array([item['logits_options'] for _, item in enumerate(logits_w_data)]))
    logits_pred = np.argmax(Logits[model], axis=1)
    logits_uncertainty = 1 - np.max(Logits[model][:, :4], axis=1)
    logits_put = np.column_stack((logits_pred,logits_uncertainty,logits_truth))
    uncertainty_acc_result(f'logits {model}', logits_put, cal = False, plot_results = True, save = True)
    Logits[model] = np.column_stack((Logits[model], OneHotEncoder(sparse_output=False).fit_transform(np.array(subject).reshape(-1, 1))))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mother_list", type=list, default=['Llama-2-7b-hf','Qwen-1_8B','falcon-7b','internlm-7b','Yi-34B','Qwen-7B','Qwen-14B','deepseek-llm-7b-base','Yi-6B','Qwen-72B','mpt-7b','deepseek-llm-67b-base','Llama-2-13b-hf','Llama-2-70b-hf','falcon-40b','Mistral-7B-v0.1'])
    parser.add_argument("--sub_list", type=list, default=['Qwen-14B','Yi-6B','Mistral-7B-v0.1'])
    parser.add_argument("--datasets_dir", type=str, default='D:/LLM-uncertainty-qualification/data/')
    parser.add_argument("--tokens_data_dir", type=str, default='D:/LLM-uncertainty-qualification/data/black_box_data/')
    parser.add_argument("--logits_data_dir", type=str, default='D:/LLM-uncertainty-qualification/data/white_box_data/')
    parser.add_argument("--data_name", type=str, default='MMLU_w')
    parser.add_argument("--mapping", type=dict, default={'A': 0,'B': 1,'C': 2,'D': 3})
    parser.add_argument("--tasks", type=dict, default={'mmlu_10k':'QA', 'cosmosqa_10k':'RC', 'hellaswag_10k':'CI', 'halu_dialogue':'DRS', 'halu_summarization':'DS'})

    args = parser.parse_args()

    main(args)

