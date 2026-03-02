# Research on Japanese and Chinese Machine Translation based on Knowledge Augmentation

This repository contains the source code and experimental setup used in this research.

## Explanation of Experiments

experiment_1: Baseline  
experiment_3_earliest_test: Raw Sum  
experiment_3_earliest_test_1: Raw Sum + Entity Only  
experiment_3_earliest_test_2: Raw Sum + Gate  
experiment_3_earliest_test_3: Raw Sum + Entity Only + Gate  
experiment_4_earliest_test: Concatenation  
experiment_5_earliest_test: Residual Sum  
experiment_5_earliest_test_1: Residual Sum + Entity Only   
experiment_5_earliest_test_2: Residual Sum + Gate  
experiment_5_earliest_test_3: Residual Sum + Entity Only + Gate  
experiment_6_earliest_test: Ablation experiment

---

## 1. Directory Structure

### `preprocess/`
Scripts and utilities for data preprocessing

### `experiment/`
Contains:
- All experimental scripts
- Model modifications from the original source code of *"End-to-end Entity-aware Neural Machine Translation (Xie et al., 2022)"* , link: https:// www. dropb ox. com/s/ 1owvy h6w0a hu8k4/ entity_ nmt. zip?dl=0
- Training configurations
- Evaluation procedures used in this research

### `conda_environment/`
Contains exported conda environment files (`.yml`) used for reproducibility.

---

## 2. Required External Resources

### 2.1 Pretrained Embeddings

Pretrained Wikipedia2Vec embeddings (Japanese, 300-dimensional):

- File: `jawiki_20180420_300d.pkl`
- Download link:
  https://wikipedia2vec.github.io/wikipedia2vec/pretrained/

---

### 2.2 Dataset

ASPEC Japanese–Chinese (ASPEC JC) corpus:

https://jipsti.jst.go.jp/aspec/

Please download the dataset from the official website.

---

## 3. Environment Setup

### System Configuration

- Python 3.10  
- CUDA 13.1  
- GPU: NVIDIA RTX PRO 6000 Blackwell (96GB VRAM)  
- PyTorch 2.9.1+cu128  

---

### Conda Environments

Most scripts can be executed using: ~/conda_environment/xyc.1.yml  

Exceptions:

- `~/preprocess/3_ner_tag.py`  
  → must be executed under: dp-old.yml  

- `~/experiment/experiment_x/train_and_eval.sh`  
→ must be executed under: fairseq12.yml  

---

## 4. Data Preparation

### 4.1 Dataset Processing

Process the ASPEC JC dataset by executing the scripts in the `preprocess/` directory in sequential order.

This will generate training-ready data.

---

### 4.2 Knowledge Graph Embedding Construction

1. Download: jawiki_20180420_300d.pkl  

2. Run: ~/preprocess/build_wiki2vec_kg_emb.py  

This will generate: kg_embed.ja.wiki2vec.pt

---

### 4.3 Applying Modified fairseq

The project includes a patch file: ~/experiment/experiment_x/my_fairseq_changes.patch

To reproduce the modified fairseq framework:  
cd /path/to/experiment/experiment_x  
git clone https://github.com/facebookresearch/fairseq.git  
cd fairseq  
git checkout 3d262bb25690e4eb2e7d3c1309b1e9c406ca4b99  
git apply my_fairseq_changes.patch  

---

## 5. Running Experiments

To reproduce experiments:

Run each `train_and_eval.sh` script inside the `experiment/` folder.  
Example:  
cd experiment/experiment_x  
bash train_and_eval.sh  