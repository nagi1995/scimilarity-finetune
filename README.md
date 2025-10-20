# SCimilarity Fine-Tuning

This repository provides tools and workflows for fine-tuning the SCimilarity model on custom single-cell RNA sequencing (scRNA-seq) datasets. SCimilarity is a unifying representation of single-cell expression profiles that quantifies similarity between expression states and generalizes to represent new studies without additional training

## ▶️ Sample Demo

https://github.com/user-attachments/assets/b3027f4a-355f-446a-bdf8-7352dc2ab24f

## Dataset Preparation

### 1. Downloading the Dataset

Dataset can be downloaded from [here](https://cellxgene.cziscience.com/collections/433700dc-e8a5-48b0-b5cd-beb22f3f88fe)

### 2. Data Preprocessing

- Clone this repository:

   ```bash
   git clone https://github.com/nagi1995/scimilarity-finetune.git
   cd scimilarity-finetune
   ```
- Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

Before fine-tuning, preprocess the dataset to match the input requirements of the SCimilarity model. This typically involves:

- Normalizing gene expression values.
- Ensuring gene identifiers are consistent with those used in the pre-trained SCimilarity model.
- Structuring the data in a format compatible with the model's input pipeline.
- [03_Preprocessing_and_Dataset_Preparation](notebooks/03_Preprocessing_and_Dataset_Preparation.ipynb) demonstrates the preprocessing steps


### 3. LoRA fine tuning

- [04_LoRA_training](notebooks/04_LoRA_training.ipynb) demonstrates the LoRA fine tuning steps involved



