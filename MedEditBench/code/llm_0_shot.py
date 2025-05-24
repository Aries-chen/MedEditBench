import argparse
import os
import json
from tqdm import tqdm
import warnings

parser = argparse.ArgumentParser(
    description="Run LLM QA generation for different models and datasets"
)
parser.add_argument(
    "--model_id",
    type=str,
    required=True,
    choices=[
        "Llama-3.1-8B-Instruct",
        "Llama-3.2-3B-Instruct",
    ]
)
parser.add_argument(
    "--data_id",
    type=str,
    required=True,
    choices=["MedExQA", "MedMCQA"]
)
parser.add_argument(
    "--subset_num",
    type=str,
    default="all"
)
args = parser.parse_args()

model_id = args.model_id
data_id = args.data_id
subset_num = args.subset_num

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
warnings.filterwarnings("ignore")

from utils import (
    get_original_model,
    get_processed_data,
    icl_system_prompt,
    text_generate_batch,
    get_json_data,
    is_ans_equal
)

output_path = f'../data/2_medical_edit_data/{data_id}/{model_id}/sub{subset_num}'
os.makedirs(output_path, exist_ok=True)
llm_output_path = os.path.join(output_path, "llm_qa.json")

ori_model = get_original_model(model_id)
data = get_processed_data(data_id, model_id, subset_num=subset_num)

batch_size = 50
qa = []
for start in tqdm(range(0, len(data), batch_size)):
    end = min(start + batch_size, len(data))
    batch_data = data.iloc[start:end]
    queries = [f"Question: {d}" for d in batch_data['question']]
    batch_res = text_generate_batch(
        ori_model,
        queries,
        sys_prompt=icl_system_prompt,
        temperature=0.0,
        max_new_tokens=512,
        model_id=model_id
    )
    for i, (idx, row) in enumerate(batch_data.iterrows()):
        sample = {
            "id": str(row['id']),
            "question": row['question'],
            "llm_response": batch_res[i],
            "re": row['exp'],
            "gta": row['answer']
        }
        qa.append(sample)
        if i == 0:
            print(f'id: {row["id"]}')
            print(f'input_text: {row["question"]}')
            print(f'res: {batch_res[i]}')

with open(llm_output_path, "w") as f:
    json.dump(qa, f, indent=4)
print(f"Saved {len(qa)} samples to {llm_output_path}")

raw = get_json_data(llm_output_path)

filter_data = []
for d in raw:
    res = d["llm_response"]
    gta = d["gta"]
    if "[STEP]" not in res or "[ANSWER]" not in res:
        continue
    ans = res.split("[ANSWER]")[-1].strip()
    if is_ans_equal(gta, ans):
        continue
    steps = [s.strip() for s in res.split("[ANSWER]")[0].split("[STEP]") if s.strip()]
    filter_data.append({
        "id": d["id"],
        "question": d["question"],
        "llm_answer": ans,
        "llm_cot": steps,
        "re": d["re"],
        "gta": gta
    })

llm_qa_output_path = os.path.join(output_path, "llm_qa_filtered.json")

with open(llm_qa_output_path, "w") as f:
    json.dump(filter_data, f, indent=4)
print(f'Filtered: {len(filter_data)} samples saved to {llm_qa_output_path}')
