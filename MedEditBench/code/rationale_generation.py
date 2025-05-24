import argparse
import os
import json
from tqdm import tqdm
import warnings

parser = argparse.ArgumentParser(
    description="Generate self-generated-rationale (sgr) explanations"
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

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
warnings.filterwarnings("ignore")

from utils import (
    get_original_model,
    get_json_data,
    text_generate_cot_batch,
    cot_prompt,
    cot_1shot_q,
    cot_1shot_a,
    cot_1shot_r,
    cot_1shot_exp,
    validate_cot_response
)

output_path = f'../data/2_medical_edit_data/{data_id}/{model_id}/sub{subset_num}'
os.makedirs(output_path, exist_ok=True)

llm_source_path = os.path.join(output_path, "llm_qa_filtered.json")
cot_output_path = os.path.join(output_path, "cot_qa.json")

ori_model = get_original_model(model_id)
data = get_json_data(llm_source_path)
print(f"Loaded {len(data)} filtered QA samples from {llm_source_path}")

batch_size = 20

for start in tqdm(range(0, len(data), batch_size), desc="Processing batches"):
    end = min(start + batch_size, len(data))
    batch_data = data[start:end]

    queries = []
    for d in batch_data:
        q = d["question"]
        a = d["gta"]
        r = d["re"]
        queries.append(f"Question: {q}\nAnswer: {a}\nre: {r}")

    batch_responses = text_generate_cot_batch(
        ori_model,
        queries,
        sys_prompt=cot_prompt,
        temperature=0.0,
        model_id=model_id,
        one_shot_q=cot_1shot_q,
        one_shot_a=cot_1shot_a,
        one_shot_r=cot_1shot_r,
        one_shot_exp=cot_1shot_exp
    )

    for idx_in_batch, response in enumerate(batch_responses):
        global_idx = start + idx_in_batch
        d = data[global_idx]
        print(f'id: {d["id"]}')
        print(f'gta: {d["gta"]}')
        print(f'response: {response}')

        valid, res, ans = validate_cot_response(response, d["gta"])
        if valid:
            steps = [s.strip() for s in res.split("[ANSWER]")[0].split("[STEP]") if s.strip()]
            data[global_idx]["sgr"] = steps
        else:
            print("Invalid response generated.")
            data[global_idx]["sgr"] = None

data_filtered = [d for d in data if d["sgr"] is not None]
print(f'Before filter: {len(data)}, after filter: {len(data_filtered)}')

with open(cot_output_path, "w") as f:
    json.dump(data_filtered, f, indent=4)

print(f"Saved {len(data_filtered)} sgr samples to {cot_output_path}")
print(data_filtered[0])
