#!/bin/bash
python run_editing.py \
  --gpu 5 \
  --data_id MedMCQA \
  --model_id Llama-3.1-8B-Instruct \
  --edit_method memit \
  --edit_data_type both \
  --edit_target_type sgr \
  --shuffle 
