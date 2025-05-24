import __main__
import sys
sys.path.append('../../')
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import warnings
warnings.filterwarnings("ignore")

import json
import re
from tqdm import tqdm
import pandas as pd

import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer

from easyeditor import BaseEditor, BaseEditor_external



post_edit_system_prompt = '''Please answer the following question using a chain of thought. 
Use the token [STEP] to start each thought step, and the token [ANSWER] to indicate the final answer option (A, B, C, or D).

Template:
[STEP] ...
[STEP] ...
...
[ANSWER] ...'''


icl_system_prompt = '''Given the following question and context, use a chain of thought to answer.
Use the token [STEP] to start each thought step, and the token [ANSWER] to indicate the final answer option (A, B, C, or D).

Template:
[STEP] ...
[STEP] ...
...
[ANSWER] ...'''


cot_prompt = '''You have access to a reference that may assist in answering the question and in ultimately deriving the correct answer. Then focus on answering the following question using a chain of thought. 
Use the token [STEP] to start each thought step, and the token [ANSWER] to indicate the final answer option (A, B, C, or D). 

Template:
[STEP] ...
[STEP] ...
...
[ANSWER] ...'''

cot_1shot_q = "Which component of a PET scanner is used to collect the position information of annihilation radiation? A: Coincidence circuit B: Scintillator C: Collimator D: Photomultiplier tube"
cot_1shot_a = "A: Coincidence circuit"
cot_1shot_r = "In PET imaging, the detection of annihilation radiation, which occurs due to the interaction between positrons and electrons, is essential for image creation. A device that collects the position information of the annihilation radiation is required for this purpose. This device is the coincidence circuit. The coincidence circuit measures the time at which the annihilation radiation is detected by a photomultiplier tube, and simultaneously by another photomultiplier tube, to accurately determine the position of the annihilation radiation. Therefore, the coincidence circuit plays a crucial role in PET imaging."

cot_1shot_exp = '''
[STEP] According to the reference, the coincidence circuit is responsible for measuring the time at which the annihilation radiation is detected by two photomultiplier tubes, allowing for the accurate determination of the position of the annihilation radiation.
[STEP] The coincidence circuit is specifically designed to collect the position information of annihilation radiation, making it the correct answer.
[STEP] The scintillator is a component that converts the annihilation radiation into visible light, but it does not collect position information.
[STEP] The collimator is a component that helps to focus the annihilation radiation onto the scintillator, but it does not collect position information.
[STEP] The photomultiplier tube is a component that detects the visible light produced by the scintillator, but it does not collect position information.
[ANSWER] A'''


def text_generate_cot_batch(model, queries, sys_prompt=cot_prompt, temperature=0.2, max_new_tokens=512, model_id="Llama-3.2-3B-Instruct",
                        one_shot_q=cot_1shot_q, one_shot_a=cot_1shot_a, one_shot_r=cot_1shot_r, one_shot_exp=cot_1shot_exp):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = AutoTokenizer.from_pretrained("your path", trust_remote_code=True)

    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'
    tokenizer.truncation_side = 'left'

    batch_messages = [
        [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f'Question: {one_shot_q}\nAnswer: {one_shot_a}\nReference: {one_shot_r}'},
            {"role": "assistant", "content": one_shot_exp},
            {"role": "user", "content": query}
        ] 
        for query in queries  
    ]

    all_input_ids = []
    for msgs in batch_messages:
        input_ids = tokenizer.apply_chat_template(
            msgs, 
            tokenize=True, 
            add_generation_prompt=True, 
            return_tensors="pt"
        ).squeeze(0)
        all_input_ids.append(input_ids)
    model_inputs = tokenizer.pad({"input_ids": all_input_ids}, padding=True, return_tensors="pt").to(device)
    model.to(device)

    generated_ids = model.generate(
        **model_inputs, 
        temperature=temperature,
        do_sample=False,
        num_beams=1,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    responses = []
    for i in range(len(queries)):
        template_length = len(model_inputs['input_ids'][i])
        trimmed_ids = generated_ids[i][template_length:]
        responses.append(tokenizer.decode(trimmed_ids, skip_special_tokens=True))
    
    return responses


def get_json_data(dataset_input_path):
    with open(dataset_input_path, "r") as f:
        data = json.load(f)
    return data


def get_original_model(model_id):
    if model_id == "Llama-3.2-3B-Instruct":
        model_path = "../../ckpt/Llama-3.2-3B-Instruct"
    elif model_id == "Llama-3.2-1B-Instruct":
        model_path = "../../ckpt/Llama-3.2-1B-Instruct"
    else:
        raise ValueError("model_id not found")

    ori_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32, device_map="auto")

    return ori_model


def text_generate_batch(model, queries, sys_prompt=icl_system_prompt, temperature=0.0, max_new_tokens=512, model_id="Llama-3.2-3B-Instruct"):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained("your path", trust_remote_code=True)

    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'
    tokenizer.truncation_side = 'left'

    batch_messages = [
        [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": query}
        ] 
        for query in queries  
    ]

    all_input_ids = []
    for msgs in batch_messages:
        input_ids = tokenizer.apply_chat_template(
            msgs, 
            tokenize=True, 
            add_generation_prompt=True, 
            return_tensors="pt"
        ).squeeze(0)
        all_input_ids.append(input_ids)
    model_inputs = tokenizer.pad({"input_ids": all_input_ids}, padding=True, return_tensors="pt").to(device)
    
    model.to(device)

    generated_ids = model.generate(
        **model_inputs, 
        temperature=temperature,
        do_sample=False,
        num_beams=1,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )


    responses = []
    for i in range(len(queries)):
        template_length = len(model_inputs['input_ids'][i])
        trimmed_ids = generated_ids[i][template_length:]
        responses.append(tokenizer.decode(trimmed_ids, skip_special_tokens=True))
    
    return responses



def text_generate(model, query, sys_prompt=post_edit_system_prompt, temperature=0.0, max_new_tokens=512, model_id="Llama-2-7b-chat-hf"):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    tok = get_tok(model_id)

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": query}
    ]
    text = tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tok.encode(text, return_tensors="pt").to(device)
    template_length = len(model_inputs[0])
    model.eval()
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=model_inputs,
            temperature=temperature, 
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
            do_sample=False,
            max_new_tokens=max_new_tokens
        )
    trimmed_generated_ids = generated_ids[0][template_length:]
    response = tok.decode(trimmed_generated_ids, skip_special_tokens=True)
    return response



new_q_sys_prompt = '''Strictly use the provided medical fact to create a standalone exam question that:

* Extends and applies the core knowledge from the original question (ori_q) to a new clinical scenario.
* Tests further application of the underlying medical principle by exploring nuanced aspects (e.g., treatment variations, diagnostic approaches) of the same fact.
* Ensures that the new clinical scenario is distinct from the original, yet its reasoning and answer are strictly derived from—and can be found within—the provided fact without external information.

Mandatory Requirements:
1. Identify 1-2 key elements from the provided fact that relate directly to the original question’s core focus.
2. Build the question around these elements in a novel clinical context (e.g., different patient demographics or clinical settings) that extends the original knowledge.
3. Ensure the answer is directly supported by the content in the provided fact.

Strict Prohibitions:
1. No recycled answer options or distractors from the original question.
2. No introduction of external knowledge not included in the provided fact.
3. No repetition of the original question's clinical presentation or distractor structure.

Output Format:
[new_q]
1. Clinical stem presenting a new context that builds on the original core knowledge.
2. 4 plausible answer options that incorporate novel distractors.
3. A clear differentiator from the original question’s focus.

[new_a]
Provide answer reasoning that validates the new fact-derived clinical application and ensure the correct answer can be found within the provided fact.

[new_exp]
1. Explicitly cite the key fact elements used.
2. Explain how the scenario extends the original question's core knowledge.
'''


locality_q_sys_prompt = '''Strictly use the provided medical fact to create a standalone exam question that:

* Tests a distinct area of medical knowledge that is different from the core focus of the original question (ori_q).
* Evaluates an alternative clinical application scenario, ensuring the underlying knowledge is strictly derived from the provided fact.
* Presents a fresh angle that does not overlap with the original question’s content.

Mandatory Requirements:
1. Extract 1-2 key elements from the provided fact that represent different or additional aspects not covered in ori_q.
2. Construct a question that leverages these elements to form an entirely separate clinical scenario (e.g., alternative diagnostic methods or treatment considerations) from the original question.
3. Ensure all reasoning and content are exclusively based on the provided fact.

Strict Prohibitions:
1. No reuse of answer logic or distractors from the original question.
2. No incorporation of external information beyond the provided fact.
3. No similarity in clinical presentation to the original question’s scenario.

Output Format:
[locality_q]
1. Clinical stem with a new context that is clearly distinct from the original question’s focus.
2. 4 plausible answer options with novel distractors.
3. A clear differentiation from the original question’s tested knowledge.

[locality_a]
Provide answer reasoning strictly based on the fact-derived knowledge.

[locality_exp]
1. Explicitly reference the key fact elements.
2. Contrast this scenario with the original question’s focus.
'''
    

# def generate_question_by_ds(q, a, fact_str, sys_prompt=new_q_sys_prompt):
#     client = Ark('your api key')


sub_prompt = '''
You are a clinical concept extractor. Given a medical Question and its Explanation, strictly follow these steps:  

1. **Identify Core Focus**: Considering the explanation, find the most short centered medical entity/action explicitly mentioned in the Question  
2. **Original Text Verification**: Ensure the subject_word exists verbatim (including uppercase/lowercase letter consistency) in the Question.
3. **Output Control**: Return ONLY the extracted subject_word without any specific choice identifiers (e.g., "A", "Option A", "(A)" or "Choice A")
'''

one_shot_q = "Question: The site of action of local anaesthetic in epidural anesthesia is \n  A: Spinal nerve root B: Spinal cord C: Epidural neural tissue D: Anterior root of spinal nerve \n Answer: "
one_shot_exp = "Blockade of these posterior nerve root fibers interrupts somatic and visceral sensation. Therefore, the site of action of local anesthetics in epidural anesthesia is at the spinal nerve root level."
one_shot_subj = "Spinal nerve root"


def get_subject_word(q, exp, model, llmtokenizer, system_prompt=sub_prompt,
                     one_shot_q=one_shot_q, one_shot_exp=one_shot_exp, one_shot_subj=one_shot_subj):
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f'{one_shot_q}\nExplanation: {one_shot_exp}'},
        {"role": "assistant", "content": one_shot_subj},
        {"role": "user", "content": f'{q}\nExplanation: {exp}'}
    ]

    retries = 0
    max_retries = 3
    while retries < max_retries:
        input_ids = llmtokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)

        outputs = model.generate(
            input_ids,
            do_sample=False,
            num_beams=1,
            max_new_tokens=32,
            temperature=0.9,  
            eos_token_id=llmtokenizer.eos_token_id, pad_token_id=llmtokenizer.eos_token_id
        )

        generated_ids = outputs[0][input_ids.shape[-1]:]  
        response = llmtokenizer.decode(generated_ids, skip_special_tokens=True)

        if response not in ['A', 'B', 'C', 'D']:
            if response in q:
                return response
            else: 
                print(f'Error: subj: {response} - not in q: {q}')
        else:
            print(f'Error: subj: {response} - is letter in choices')

        retries += 1

    return None


def get_edit_model(hparams, prompts, subject, target_new, ground_truth):
    editor=BaseEditor.from_hparams(hparams)
    _, edited_model, _ = editor.edit(
        prompts=prompts,
        subject=subject,
        target_new=target_new,
        ground_truth=ground_truth,
        sequential_edit=True, 
    )
    return edited_model


def get_edit_model_fast(hparams, prompts, subject, target_new, ground_truth, model, tok):
    editor = BaseEditor_external.from_hparams(hparams, model=model, tok=tok)
    _, edited_model, _ = editor.edit(
        prompts=prompts,
        subject=subject,
        target_new=target_new,
        ground_truth=ground_truth,
        sequential_edit=True, 
    )
    return edited_model


# helpful functions
def is_ans_equal(gt_ans, edited_ans):
    if gt_ans is None or edited_ans is None:
        return False

    if edited_ans[0].lower() not in ['a', 'b', 'c', 'd']:
        letter = find_final_option_letter(edited_ans)
        print(f'letter: {letter}')
        if letter:
            if gt_ans[0].lower() == letter.lower():
                return True
        else:
            return False
    else:
        if gt_ans[0].lower() == edited_ans[0].lower():
            return True
    return False


def get_ans_part(answer_text):
    answer_match = re.search(r'\[ANSWER\][\s]*([A-D])', answer_text)
    return answer_match.group(1) if answer_match else None


def clean_string(input_str):

    input_str = re.sub(r'\([A-D]\)', '', input_str)

    input_str = re.sub(r'Ans\. [a-d]\.', '', input_str)

    input_str = re.sub(r'Ans\. [A-D]\.', '', input_str)

    input_str = re.sub(r'Answer- [A-D]', '', input_str)

    input_str = re.sub(r'\([Oo]ption [A-D]\)', '', input_str)

    input_str = re.sub(r'option [A-D]', '', input_str, flags=re.IGNORECASE)
    
    input_str = re.sub(r'the correct answer is [A-D]', '', input_str, flags=re.IGNORECASE)

    input_str = re.sub(r' choice [A-D]', '', input_str, flags=re.IGNORECASE)

    input_str = re.sub(r'Ans\. is \'[a-d]\'', '', input_str)

    return input_str


def extract_answer_content(text):
    cleaned_text = re.sub(r'[A-D]:\s*', '', text)
    cleaned_text = re.sub(r'[A-D]\)\s*', '', cleaned_text)
    return cleaned_text.strip()


def edit_status(edit_output, ori_ans): 
    edit_ans = get_ans_part(edit_output)
    edit_result = is_ans_equal(ori_ans, edit_ans)
    return edit_ans, edit_result


def calculate_success_rates(edit_results_list):
    ori_success = 0
    nq_success = 0
    loc_success = 0

    for result in edit_results_list:
        if result["ori_q_success"]:
            ori_success += 1
        if result["nq_success"]:
            nq_success += 1
        if result["loc_success"]:
            loc_success += 1

    ori_success_rate = ori_success / len(edit_results_list)
    nq_success_rate = nq_success / len(edit_results_list)
    loc_success_rate = loc_success / len(edit_results_list)

    print(f'ori_success_rate: {ori_success_rate:.3f}, nq_success_rate: {nq_success_rate:.3f}, loc_success_rate: {loc_success_rate:.3f}')

    return ori_success_rate, nq_success_rate, loc_success_rate


def get_model(model_id, edit_method):
    global model_path

    if model_id == "Llama-3.1-8B-Instruct":
        model_path = "../../ckpt/Llama-3.1-8B-Instruct"
        llmtokenizer = AutoTokenizer.from_pretrained(model_path)
        llmtokenizer.pad_token_id = llmtokenizer.eos_token_id

    elif model_id == "Llama-3.2-3B-Instruct":
        model_path = "../../ckpt/Llama-3.2-3B-Instruct"
        llmtokenizer = AutoTokenizer.from_pretrained(model_path)
        llmtokenizer.pad_token_id = llmtokenizer.eos_token_id
    else:
        raise ValueError("model_id not found")

    model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map='auto')

    return model, llmtokenizer


def get_tok(model_id):
    if model_id == "Llama-3.1-8B-Instruct":
        model_path = "../../ckpt/Llama-3.1-8B-Instruct"
        llmtokenizer = AutoTokenizer.from_pretrained(model_path)
        llmtokenizer.pad_token_id = llmtokenizer.eos_token_id

    elif model_id == "Llama-3.2-3B-Instruct":
        model_path = "../../ckpt/Llama-3.2-3B-Instruct"
        llmtokenizer = AutoTokenizer.from_pretrained(model_path)
        llmtokenizer.pad_token_id = llmtokenizer.eos_token_id

    else:
        raise ValueError("model_id not found")
    
    return llmtokenizer

def get_llm_tok_num(text, tokenizer):
    outputs = tokenizer(text)
    token_count = len(outputs["input_ids"])
    return token_count


def remove_illegal_chars(data):
    data["question_ans"] = data["question"].astype(str) + " " + data["answer"].astype(str)
    new_data = data[~data["question_ans"].apply(is_illegal_chars)]
    new_data = new_data.drop(columns=["question_ans"])
    return new_data


def remove_exp_leakage(data):
    new_data = data["explanation"].apply(clean_string)
    return new_data


import re
from easyeditor import MEMITHyperParams, MEMITAREHyperParams, AlphaEditHyperParams, LoRAHyperParams, ROMEHyperParams, GraceHyperParams, FTHyperParams
from evaluate_utils import *
import warnings
warnings.filterwarnings("ignore")


def get_hparams(edit_method, model_id): 
    if model_id == "Llama-3.2-3B-Instruct":
        if edit_method == "memit":
            hparams = MEMITHyperParams.from_hparams('../../hparams/MEMIT/llama3.2-3b.yaml')
        elif edit_method == "memit_are":
            hparams=MEMITAREHyperParams.from_hparams('../../hparams/MEMIT_ARE/llama3.2-3b.yaml')
        elif edit_method == "alphaedit":
            hparams = AlphaEditHyperParams.from_hparams('../../hparams/AlphaEdit/llama3.2-3b.yaml')
        elif edit_method == "grace":
            hparams =  GraceHyperParams.from_hparams('../../hparams/GRACE/llama3.2-3b.yaml')
        elif edit_method == "lora":
            hparams =  LoRAHyperParams.from_hparams('../../hparams/LoRA/llama3.2-3b.yaml')
        elif edit_method == "rome":
            hparams =  ROMEHyperParams.from_hparams('../../hparams/ROME/llama3.2-3b.yaml')
        elif edit_method == "ft":
            hparams =  FTHyperParams.from_hparams('../../hparams/FT/llama3.2-3b.yaml')

        else:
            raise ValueError("hparams should be 'memit', 'lora', or 'rome'")

    elif model_id == "Llama-3.1-8B-Instruct":
        if edit_method == "memit":
            hparams = MEMITHyperParams.from_hparams('../../hparams/MEMIT/llama3-8b.yaml')
        elif edit_method == "memit_are":
            hparams=MEMITAREHyperParams.from_hparams('../../hparams/MEMIT_ARE/llama3-8b.yaml')
        elif edit_method == "alphaedit":
            hparams = AlphaEditHyperParams.from_hparams('../../hparams/AlphaEdit/llama3-8b.yaml')
        elif edit_method == "grace":
            hparams =  GraceHyperParams.from_hparams('../../hparams/GRACE/llama3-8b.yaml')
        elif edit_method == "lora":
            hparams =  LoRAHyperParams.from_hparams('../../hparams/LoRA/llama3-8b.yaml')
        elif edit_method == "rome":
            hparams =  ROMEHyperParams.from_hparams('../../hparams/ROME/llama3-8b.yaml')
        elif edit_method == "ft":
            hparams =  FTHyperParams.from_hparams('../../hparams/FT/llama3-8b.yaml')
        else:
            raise ValueError("hparams should be 'memit', 'lora', or 'rome'")

    else:
        raise ValueError("model_id should be 'Llama-3.2-3B-Instruct' or 'Llama-3.1-8B-Instruct'")

    print(f"The hparams of {edit_method} is {hparams}")
    return hparams


def edit_by_target(d, target_type):
    if target_type == "gta":
        target = extract_answer_content(d["gta"])
    elif target_type == "re":
        target = d["re"]
    elif target_type == "sgr":
        target = d['sgr']
    else:
        raise ValueError("target_type should be 'gta', 're' or 'sgr'")
    return target



if __name__ == "__main__":
    pass
