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

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="Qwen-72B")
parser.add_argument("--raw_data_dir", type=str, default="data",
                    help="Directory where raw data are stored.")
parser.add_argument("--logits_data_dir", type=str, default="outputs_base",
                    help="Directory where logits data are stored.")
parser.add_argument("--data_names", nargs='*', 
                    default=['mmlu_10k', 'cosmosqa_10k', 'hellaswag_10k', 'halu_dialogue', 'halu_summarization'], 
                    help='List of datasets to be evaluated. If empty, all datasets are evaluated.')
parser.add_argument("--prompt_methods", nargs='*', 
                    default=['base', 'shared', 'task'], 
                    help='List of prompting methods. If empty, all methods are evaluated.')
parser.add_argument("--icl_methods", nargs='*', 
                    default=['icl1'], 
                    help='Select from icl1, icl0, icl0_cot.')
parser.add_argument("--cal_ratio", type=float, default=0.5,
                    help="The ratio of data to be used as the calibration data.")
parser.add_argument("--alpha", type=float, default=0.1,
                    help="The error rate parameter.")
args = parser.parse_args()

print(get_SPU(args))

