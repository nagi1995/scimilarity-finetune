# SCimilarity Fine-Tuning

This repository provides tools and workflows for fine-tuning the SCimilarity model on custom single-cell RNA sequencing (scRNA-seq) datasets. SCimilarity is a unifying representation of single-cell expression profiles that quantifies similarity between expression states and generalizes to represent new studies without additional training

---

## ▶️ Sample Demo

https://github.com/user-attachments/assets/b3027f4a-355f-446a-bdf8-7352dc2ab24f

---

## Project Structure
```
scimilarity-finetune/
├── app/                    # FastAPI backend application
│   ├── model/             # Model wrapper
│   └── main.py           # API endpoints
├── artifacts/             # Model artifacts and preprocessing objects
├── data/                 # Dataset storage
│   ├── raw/             # Raw input data
│   ├── processed/       # Processed datasets
│   └── json_samples/    # Example JSON inputs
├── frontend/            # Web interface
├── notebooks/          # Jupyter notebooks for analysis
├── weights/           # Model weights and LoRA adapters
└── cfn_template.yml   # AWS CloudFormation template
└── Dockerfile         # Dockerfile
```


---
## 🚀 Getting Started

### 1. Environment Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/nagi1995/scimilarity-finetune.git
   cd scimilarity-finetune
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### 2. Data Preparation

1. Download the dataset from [CellXGene](https://cellxgene.cziscience.com/collections/433700dc-e8a5-48b0-b5cd-beb22f3f88fe)
2. Place the downloaded data in `data/raw/`
3. Follow preprocessing notebooks:
   - [01_Data_Exploration](notebooks/01_data_exploration.ipynb)
   - [02_Marker_Analysis](notebooks/02_marker_analysis.ipynb)
   - [03_Preprocessing_and_Dataset_Preparation](notebooks/03_Preprocessing_and_Dataset_Preparation.ipynb)
### 3. Model Training

Run the LoRA fine-tuning notebook:
- [04_LoRA_Training](notebooks/04_LoRA_training.ipynb)

### 4. Evaluation

Compare model performance:
- [05_Baseline_Evaluation](notebooks/05_Baseline_Evaluation.ipynb)
- [06_LoRA_Evaluation](notebooks/06_LoRA_Evaluation.ipynb)

## 📈 Performance

| model | accuracy | f1 score |
|----------|----------|----------|
| Baseline    |  27.40 %    | 0.1846 |
| LoRA fine tuning    |    38.40 %  | 0.2382 |

---

## Inference Service (Locally)

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
   const BASE_URL = "http://<backend-host>:8000";
   ```
2. Open the HTML file in a browser.
3. Upload the json file with gene expression to get the `cell_type` along with `confidence`.

---

## 🌩️ AWS Deployment

### Prerequisites
- AWS CLI configured
- An ECR repository with your backend image pushed (latest tag)
- Existing VPC and subnets

### 1. Install AWS CLI and configure credentials:
   ```bash
   aws configure
   ```

### 2. Deploy using CloudFormation:
   ```bash
   aws cloudformation deploy \
     --template-file cfn_template.yml \
     --stack-name scimilarity-stack \
     --capabilities CAPABILITY_NAMED_IAM \
     --parameter-overrides \
       MyIp=<your-ip-address>/32 \
       SubnetIds='<subnet-ids>' \
       EcrRepoName=scimilarity-be \
       ContainerPort=8000
   ```

---

## 🔐 Notes

* CORS is open (`*`) for simplicity — **tighten it in production**.

