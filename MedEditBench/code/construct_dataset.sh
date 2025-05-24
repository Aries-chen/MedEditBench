#!/bin/bash
python process_data.py

python llm_0_shot.py \
  --model_id Llama-3.1-8B-Instruct \
  --data_id MedMCQA 

python rationale_generation.py \
  --model_id Llama-3.1-8B-Instruct \
  --data_id MedMCQA 

python construct_qa.py \
  --model_id Llama-3.1-8B-Instruct \
  --data_id MedMCQA 
