from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import numpy as np
import joblib
import os
import json

from app.utils import load_artifacts
from app.model.wrapper import load_lora_encoder

# Load artifacts once at startup
ARTIFACTS_DIR = "artifacts"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
hidden_dim=1024

# Load saved components
label_encoder, gene_order, train_means, train_stds = load_artifacts()
knn = joblib.load(os.path.join(ARTIFACTS_DIR, "knn_lora.joblib"))


# --- 1️⃣ Load best LoRA weights ---
best_epoch = 10
save_dir = "weights/lora/20251020_050117"
lora_path = os.path.join(save_dir, f"lora_epoch{best_epoch}.pt")
lora_weights = torch.load(lora_path)

encoder_lora = load_lora_encoder(n_genes=len(gene_order), hidden_dim=hidden_dim)

# Inject into encoder_lora
for k, v in lora_weights.items():
    encoder_lora.lora_params[k].data = v.to(DEVICE)

encoder_lora.eval()

# Define request schema
class ExpressionInput(BaseModel):
    expression: dict

# Create FastAPI app
app = FastAPI(title="scimilarity LoRA Cell Type Predictor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify your frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    """
    Health check endpoint to verify backend is running.
    Returns basic info like status and device.
    """
    return {
        "status": "ok",
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }


@app.post("/predict")
async def predict_expression(file: UploadFile = File(...)):
    try:
        # Read JSON file
        content = await file.read()
        data = json.loads(content)
        expr_dict = data.get("expression", None)
        if expr_dict is None:
            raise HTTPException(status_code=400, detail="JSON must contain 'expression' key.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
    
    # Ensure all genes are present (fill missing with 0)
    expr_vec = np.array([expr_dict.get(gene, 0.0) for gene in gene_order], dtype=np.float32)

    # Standardize using training stats
    expr_vec = (expr_vec - train_means) / (train_stds + 1e-8)
    expr_vec = expr_vec / (expr_vec.sum() + 1e-8) * np.sum(train_means)
    # Ensure all values ≥ 0 before log1p
    expr_vec = np.clip(expr_vec, a_min=0.0, a_max=None)
    expr_vec = np.log1p(expr_vec)
    expr_vec = np.clip(expr_vec, -10, 10)

    # Convert to tensor
    xb = torch.tensor(expr_vec, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    # Get embedding
    with torch.no_grad():
        emb = encoder_lora(xb).cpu().numpy()

    # Predict with KNN
    probs = knn.predict_proba(emb)[0]
    pred_idx = np.argmax(probs)
    pred_label = label_encoder.classes_[pred_idx]
    confidence = float(probs[pred_idx])

    return {
        "cell_type": pred_label,
        "confidence": confidence
    }
