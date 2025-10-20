# SCimilarity Fine-Tuning

This repository provides tools and workflows for fine-tuning the SCimilarity model on custom single-cell RNA sequencing (scRNA-seq) datasets. SCimilarity is a unifying representation of single-cell expression profiles that quantifies similarity between expression states and generalizes to represent new studies without additional training

---

## ▶️ Sample Demo

https://github.com/user-attachments/assets/b3027f4a-355f-446a-bdf8-7352dc2ab24f

---

## 1. Downloading the Dataset

Dataset can be downloaded from [here](https://cellxgene.cziscience.com/collections/433700dc-e8a5-48b0-b5cd-beb22f3f88fe)

## 2. Data Preprocessing

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


## 3. LoRA fine tuning

- [04_LoRA_training](notebooks/04_LoRA_training.ipynb) demonstrates the LoRA fine tuning steps involved

## 4. Model Evaluation

| model | accuracy | f1 score |
|----------|----------|----------|
| Baseline    |  27.40 %    | 0.1846 |
| LoRA fine tuning    |    38.40 %  | 0.2382 |

---

## Inference Service

To deploy the fine-tuned model as an inference service:

### 1. Build the Docker image:

   ```bash
   docker build -t scimilarity-be .
   ```

### 2. Run the Docker container:

   ```bash
   docker run -p 8000:8000 scimilarity-be
   ```

### 3. The inference service will be accessible at `http://localhost:8000`. You can send POST requests to this endpoint with gene expression data to obtain similarity predictions.

---
##  Frontend Usage

1. Edit `frontend/index.html` → set:

   ```js
   const BASE_URL = "http://localhost:8000";
   ```
2. Open the HTML file in a browser.
3. Upload the json file with gene expression to get the `cell_type` along with `confidence`.

---

## 🔐 Notes

* CORS is open (`*`) for simplicity — **tighten it in production**.

