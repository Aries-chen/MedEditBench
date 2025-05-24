import argparse
import json
from tqdm import tqdm
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    
from utils import *
from consctruct_qa_utils import *

import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(
    description="Generate locality and new QA pairs based on references"
)
parser.add_argument(
    "--model_id",
    type=str,
    required=True,
    choices=[
        "Llama-3.1-8B-Instruct",
        "Llama-3.2-3B-Instruct",
    ],
    help="Model ID"
)
parser.add_argument(
    "--data_id",
    type=str,
    required=True,
    choices=["MedExQA", "MedMCQA"],
    help="Data ID"
)
parser.add_argument(
    "--subset_num",
    type=str,
    default="all",
    help="subset num, default all"
)
args = parser.parse_args()

model_id = args.model_id
data_id = args.data_id
subset_num = args.subset_num


output_path = f'../data/2_medical_edit_data/{data_id}/{model_id}/sub{subset_num}'
os.makedirs(output_path, exist_ok=True)


source_path_gen = output_path +"/cot_qa.json"
data = get_json_data(source_path_gen)

model = get_original_model(model_id)
print(f'Using model: {model.config.name_or_path}')

new_qa_list = []
for d in tqdm(data):
    print(f'\nProcessing ID: {d["id"]}')
    ori_q = d["question"]
    ori_a = d["gt_ans"]
    d['llm_cot'] = " ".join(d['llm_cot'])
    d['cot'] = clean_string(" ".join(d["cot"]))
    reference = clean_string(d['reference'])
    d['reference'] = reference

    subj = generate_sub_by_ds(ori_q, ori_a, reference) 
    if subj is None:
        print("Failed to generate subject word")
        continue
    print(f'ori_q: {ori_q}\nsubj: {subj}')
    d['subj_word'] = subj
    qa_pair = get_qa_pair(ori_q, ori_a, reference, model)  
    
    d['loc_qa'] = qa_pair.get('loc_qa')
    d['new_qa'] = qa_pair.get('new_qa')
        
    new_qa_list.append(d)

print(f'len(new_qa_list): {len(new_qa_list)}')


from consctruct_qa_utils import shuffle_data

edit_data_path = output_path+"/edit_data_all.json"
with open(edit_data_path, "w") as f:
    json.dump(new_qa_list, f, ensure_ascii=False, indent=4)
