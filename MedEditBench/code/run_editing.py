import argparse
import sys
import numpy as np
import copy
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import json
import torch
import warnings
from tqdm import tqdm

from utils import *
from evaluate_utils import *
sys.path.append('../../')
warnings.filterwarnings("ignore")


def edit_process(data, edit_method, edit_target_type,  edit_output_path, edit_system_prompt, shuffle=False, model_id="Llama-3.2-3B-Instruct"):
    hparams = get_hparams(edit_method, model_id)
    ori_model, tokenizer = get_model(model_id, edit_method)
    edit_results_list = []
    ori_success = 0
    nq_success = 0
    loc_success = 0
    count_new = 0
    count_loc = 0

    for d in tqdm(data):
        base = copy.deepcopy(ori_model)
        print(f'id: {d["id"]}\n')
        q = d["question"]
        llm_answer = d["llm_answer"]
        subj_word = d["subj_word"]
        ori_gt_ans = d["gta"]

        edit_target = edit_by_target(d, edit_target_type)
        
        new_q, new_a = None, None
        loc_q, loc_a = None, None
        
        if d.get('new_qa'):
            new_qa = d['new_qa']
            new_q = new_qa.get('q')
            new_a = new_qa.get('a')
            count_new += 1
        if d.get('loc_qa'):
            loc_qa = d['loc_qa']
            loc_q = loc_qa.get('q')
            loc_a = loc_qa.get('a')
            count_loc += 1

        edited_model = get_edit_model_fast(hparams, q, subj_word, edit_target, llm_answer, model=base, tok=tokenizer)
        print(f'\nid: {d["id"]}, Edit method: {edit_method}, Edit target type: {edit_target_type}, Model ID: {model_id}')

        if shuffle:
            s_q = d["shuffle_qa"]["s_q"]
            s_a = d["shuffle_qa"]["s_a"]
            print(f'shuffle mode, test by s_qa: \n{s_q},{s_a}')
            edited_ori_exp = text_generate(edited_model, f'Question: {s_q}', sys_prompt=edit_system_prompt, model_id=model_id)
            edited_ori_ans, ori_edit_status = edit_status(edited_ori_exp, s_a)
        else:
            edited_ori_exp = text_generate(edited_model, f'Question: {q}', sys_prompt=edit_system_prompt, model_id=model_id)
            edited_ori_ans, ori_edit_status = edit_status(edited_ori_exp, ori_gt_ans)
        
        print(f'Edit by {edit_method}, edit_target_type: {edit_target_type}: {edit_target}\n')
        print(f"edited_ori_exp: \n{edited_ori_exp}")
        
        if ori_edit_status:
            ori_success += 1
        
        if new_q and new_a:
            print(f'Test the Edited Model by New QA: \n{new_q}, {new_a}')
            edited_nq_exp = text_generate(edited_model, f'Question: {new_q}', sys_prompt=edit_system_prompt, model_id=model_id)
            print(f'edited_nq_exp: \n{edited_nq_exp}')
            edited_nq_ans, nq_edit_status = edit_status(edited_nq_exp, new_a)
            if nq_edit_status:
                nq_success += 1
        else:
            edited_nq_exp = None
            nq_edit_status = None

        if loc_q and loc_a:
            print(f'Test the Edited Model by Locality QA: \n{loc_q}, {loc_a}')
            edited_loc_exp = text_generate(edited_model, f'Question: {loc_q}', sys_prompt=edit_system_prompt, model_id=model_id)
            print(f'edited_loc_exp: \n{edited_loc_exp}')
            edited_loc_ans, loc_edit_status = edit_status(edited_loc_exp, loc_a)
            if loc_edit_status:
                loc_success += 1
        else:
            edited_loc_exp = None
            loc_edit_status = None

        result_entry = {
            "id": d["id"],
            "question": q,
            "ori_gt_ans": ori_gt_ans,
            "llm_answer": llm_answer,
            f"{edit_target_type}": edit_target,
            "edited_ori_exp": edited_ori_exp,
            "edited_ori_ans": edited_ori_ans,
            "ori_q_success": ori_edit_status,
        }
        if new_q and new_a:
            result_entry.update({
                "new_q": new_q,
                "new_a": new_a,
                "new_exp": d["new_qa"]['exp'],  
                "edited_nq_exp": edited_nq_exp,
                "edited_nq_ans": edited_nq_ans,
                "nq_success": nq_edit_status
            })
        if loc_q and loc_a:
            result_entry.update({
                "loc_q": loc_q,
                "loc_a": loc_a,
                "loc_exp": d["loc_qa"]['exp'],
                "edited_loc_exp": edited_loc_exp,
                "edited_loc_ans": edited_loc_ans,
                "loc_success": loc_edit_status
            })

        if shuffle:
            result_entry["shuffle_qa"] = d["shuffle_qa"]

        if edit_target_type in ["sgr", "re"]:
            pred_cot = extract_cot(edited_ori_exp)
            if not pred_cot.strip():
                pred_cot = edited_ori_exp  
            rougeL_score = compute_rougeL(pred_cot, edit_target)
            bleu_score = compute_bleu_score(pred_cot, edit_target)
            print(f'ROUGE-L Score: {rougeL_score}, BLEU Score: {bleu_score},')
            result_entry["rougeL_score"] = rougeL_score
            result_entry["bleu_score"] = bleu_score 

        edit_results_list.append(result_entry)


    ori_success_rate = ori_success / len(data)
    nq_success_rate = nq_success / count_new if count_new > 0 else 0
    loc_success_rate = loc_success / count_loc if count_loc > 0 else 0
    avg_success_rate = np.mean([ori_success_rate, nq_success_rate, loc_success_rate])

    print(f'Edit method: {edit_method}, Edit target type: {edit_target_type}, Shuffle: {shuffle}, Model ID: {model_id}')
    print(f'ori_success_rate: {ori_success_rate:.3f}, nq_success_rate: {nq_success_rate:.3f}, loc_success_rate: {loc_success_rate:.3f}')
    print(f'avg_success_rate: {avg_success_rate:.3f}')

    if edit_target_type in ["sgr", "re"]:
        rougeL_score_mean = np.mean([d["rougeL_score"] for d in edit_results_list if "rougeL_score" in d])
        bleu_score_mean = np.mean([d["bleu_score"] for d in edit_results_list if "bleu_score" in d])
        print(f'ROUGE-L Score Mean: {rougeL_score_mean:.3f}, BLEU Score Mean: {bleu_score_mean:.3f}')
        semantic_sim_mean = np.mean([rougeL_score_mean, bleu_score_mean])
        print(f'Semantic Similarity Mean: {semantic_sim_mean:.3f}')
        
    os.makedirs(edit_output_path, exist_ok=True)
    with open(os.path.join(edit_output_path, f"edit_by_{edit_target_type}.json"), "w") as f:
        json.dump(edit_results_list, f, indent=4, ensure_ascii=False)

    result_scores = [
        f'Edit method: {edit_method}, Edit target type: {edit_target_type}, Shuffle: {shuffle}, Model ID: {model_id}',
        f'ori_success_rate: {ori_success_rate:.3f}',
        f'nq_success_rate:  {nq_success_rate:.3f}',
        f'loc_success_rate: {loc_success_rate:.3f}',
        f'avg_success_rate: {avg_success_rate:.3f}'
    ]
    if edit_target_type in ["sgr", "re"]:
        result_scores.extend([
            f'rougeL_score_mean: {rougeL_score_mean:.3f}',
            f'bleu_score_mean: {bleu_score_mean:.3f}',
            f'semantic_sim_mean: {semantic_sim_mean:.3f}'
        ])
    with open(os.path.join(edit_output_path, f"edit_by_{edit_target_type}_scores.txt"), "w") as f:
        for line in result_scores:
            f.write(line + "\n")
        
    return edit_results_list

    
def main():
    parser = argparse.ArgumentParser(description="Medical Data Editing Script")
    
    parser.add_argument('--gpu', type=int, default=3, help='GPU card to use (default: 3)')
    parser.add_argument('--data_id', type=str, required=True, choices=['MedExQA', 'MedMCQA'], help='Dataset ID')
    parser.add_argument('--model_id', type=str, required=True, 
                      choices=['Qwen2.5-7B-Instruct', 'Llama-3.1-8B-Instruct', 'Llama-3.2-3B-Instruct'], 
                      help='Model ID')
    parser.add_argument('--edit_method', type=str, required=True, 
                      choices=['rome', 'lora', 'grace', 'memit', 'memit_are', 'ft', 'alphaedit'], 
                      help='Editing method')
    parser.add_argument('--edit_target_type', type=str, default='gkgd', 
                      help='Editing target type (default: gkgd)')
    parser.add_argument('--shuffle', default=False, action='store_true',
                      help='Shuffle original Q choices if set')
    parser.add_argument('--test', default=False, action='store_true', 
                      help='Test mode (use first data entry)')
    parser.add_argument('--exp_name', type=str, default='test',
                      help='Custom experiment name for output path suffix')

    args = parser.parse_args()
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)


    edit_data_path = (
        f"../data/2_medical_edit_data/{args.data_id}/"
        f"{args.model_id}/"
        f"edit_data.json"
    )
    
    data = get_json_data(edit_data_path)
    print(f'\nData loaded from {edit_data_path}, total {len(data)} samples.\n')
    if args.test:
        data = data[:1]

    experiment_suffix = f"_{args.exp_name}" if args.exp_name else ""
    if args.test:  
        experiment_suffix += "_test"


    edit_output_path = (
        f'../data/3_edit_output/{args.data_id}/'
        f'{args.model_id}/{args.subset}/'
        f'{args.edit_method}_{args.edit_data_type}_sf{args.shuffle}'
        f'{experiment_suffix}'
    )
    if args.gold:
        edit_output_path += "_gold"

    edit_process(
        data,
        args.edit_method,
        args.edit_target_type,
        edit_output_path,
        shuffle=args.shuffle,
        model_id=args.model_id
    )

if __name__ == "__main__":
    main()

