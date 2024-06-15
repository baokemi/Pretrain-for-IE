# SKIE Project Guide

Welcome to the SKIE project's pretraining guide. This document will walk you through the necessary steps to preprocess your datasets and perform pretraining. Please follow the requirements.txt to install and use it. 


## 1. Preprocessing

### Step 1: Dataset Preprocessing

To preprocess your dataset for the model, ensure each sentence is converted into a JSON object containing sentence and tokens. Run the following script:
```
python preprocess_pretrained_data.py
```

### Step 2: Extract AMR Subgraphs

We use a transformer-based AMR parser 'Transition-based Parsing with Stack-Transformers' in our model. 

Run the script to process AMR subgraphs:
```
nohup python get_sub_counts.py --dataset {dataset} > {dataset}subcounts.log 2>&1 &
```
## 2. Pretraining

### Initial Pretraining
For the initial pretraining of the model, use the following command. This will save the model in a specified directory named '{dataset}_global_complete_graph_finetuned_bert_model'.
```
nohup python pretrain.py --dataset {dataset} > {dataset}pretrain.log 2>&1 &
```
### Continual Training on a New Dataset
To continue training on a new dataset using a pre-trained model:
```
nohup python pretrain.py --dataset {dataset} --bert_model_name {dataset}_global_complete_graph_finetuned_bert_model > {dataset}pretrain.log 2>&1 &
```

Due to privacy concerns, our pre-trained models will be made available after the paper is accepted.