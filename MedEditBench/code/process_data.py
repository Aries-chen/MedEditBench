import pandas as pd
from utils import *
from tqdm import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"


model_id = "Llama-3.1-8B-Instruct"
# model_id = "Llama-3.2-3B-Instruct"

data_id = "MedMCQA" 
# data_id = "MedExQA" 
subset_num = "all"
batch_size = 50  

print(f'model_id: {model_id}, data_id: {data_id}, subset_num: {subset_num}, batch_size: {batch_size}')

all_data = pd.read_csv(f"../data/1_processed_data/{data_id}/all.csv")

basic_df = all_data[:] if subset_num == "all" else all_data[: subset_num]  
print(f'len(basic_df): {len(basic_df)}')


model = get_original_model(model_id)
all_samples = []
for i in range(len(basic_df)):
    sample = basic_df.iloc[i]
    q_with_exp = f"Question: {sample['question']} Context: {sample['exp']}"
    all_samples.append((sample, q_with_exp))

ds_list = []
for start in tqdm(range(0, len(all_samples), batch_size)):
    end = min(start + batch_size, len(all_samples))
    batch_samples = all_samples[start:end]
    
    samples, queries = zip(*batch_samples)
    icl_judges = text_generate_batch(model, queries, model_id=model_id, sys_prompt=icl_system_prompt)
    
    for sample, q_with_exp, icl_judge in zip(samples, queries, icl_judges):
        print(f'id: {sample["id"]}')
        print(f'icl_judge: {icl_judge}')

        icl_ans, is_valid = edit_status(icl_judge, sample["answer"])
        ds_list.append([
            sample["id"],
            sample["question"],
            sample["answer"],
            sample["exp"],
            sample["subject"],
            icl_judge,
            icl_ans,
            is_valid
        ])

ds_df = pd.DataFrame(ds_list, columns=["id", "question", "answer", "exp", "subject", "icl_judge", "icl_ans", "is_valid"])


valid_exp_df = ds_df[ds_df["is_valid"] == True]
print(f'len(ds_df): {len(ds_df)}')
print(f'len(valid_exp_df): {len(valid_exp_df)}')
save_dir = f"../data/1_processed_data/{data_id}/{model_id}"
os.makedirs(save_dir, exist_ok=True)
valid_exp_df.to_csv(f"xxx/{save_dir}/icl_processed_data_sub{subset_num}.csv", index=False)
