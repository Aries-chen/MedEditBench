# Beyond Memorization: A Rigorous Evaluation Framework for Medical Knowledge Editing

This repository contains the code and dataset for **MedEditBench**, as proposed in the paper  
**"Beyond Memorization: A Rigorous Evaluation Framework for Medical Knowledge Editing."**



## 🔧 Requirements

MedEditBench is based on the open-source [EasyEdit](https://github.com/zjunlp/EasyEdit) framework.

To build the environment, please run:

```bash
pip install -r requirements.txt
````



## 📚 Medical QA Datasets

We use the following datasets to construct `MedMCQA_edit` and `MedExQA_edit`.

* MedMCQA: [https://github.com/medmcqa/medmcqa](https://github.com/medmcqa/medmcqa)
* MedExQA: [https://github.com/knowlab/MedExQA](https://github.com/knowlab/MedExQA)

We have provided the medical knowledge editing datasets in `./MedEditBench/data/`.

You can also construct `MedMCQA_edit` and `MedExQA_edit` from scratch, please run:

```bash
cd ./MedEditBench/code/
bash construct_dataset.sh
```

## 🧠 Medical Knowledge Editing

To run the medical knowledge editing procedure, please execute:

```bash
cd ./MedEditBench/code/
bash run_editing.sh
```

For details of the editing and evaluation process, see:

```
./MedEditBench/code/run_editing.py
```
