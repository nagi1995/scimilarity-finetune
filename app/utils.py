import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import scanpy as sc
import joblib
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from anndata import AnnData
import torch
from torch.utils.data import DataLoader, Dataset

from app.logger_config import get_logger 

# Create the logger instance
logger = get_logger(__name__)

# --- Configuration Constants (Can be moved to a separate config file if needed) ---
# NOTE: These paths assume you are running from the project_root/ directory
RAW_DATA_PATH = "data/raw/dataset.h5ad"
ARTIFACTS_DIR = "artifacts"
PROCESSED_DATA_DIR = "data/processed"
LABEL_COLUMN = "cell_type" # Column used for classification labels
N_HVGS = 3000 # Number of highly variable genes to select (if used)


# ------------------------------ 1. Data Loading ------------------------------

def load_raw_data(path: str = RAW_DATA_PATH) -> AnnData:
    """Loads the raw AnnData object."""
    try:
        adata = sc.read_h5ad(path)
        logger.info(f"✅ Loaded raw data: {adata.shape[0]} cells x {adata.shape[1]} genes.")
        return adata
    except FileNotFoundError:
        logger.error(f"❌ Error: Raw data not found at {path}")
        raise

class SparseDataset(Dataset):
    def __init__(self, adata, label_encoder):
        self.X = adata.X  # keep sparse
        self.y = label_encoder.transform(adata.obs["cell_type"])
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        # Convert only the row to dense
        x = torch.tensor(self.X[idx].toarray().flatten(), dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y

def load_preprocessed_data(path: str = PROCESSED_DATA_DIR):
    train_data = sc.read_h5ad(f"{PROCESSED_DATA_DIR}/train_data.h5ad")
    val_data   = sc.read_h5ad(f"{PROCESSED_DATA_DIR}/val_data.h5ad")
    test_data  = sc.read_h5ad(f"{PROCESSED_DATA_DIR}/test_data.h5ad")

    logger.info(f"Train: {train_data.shape}, Val: {val_data.shape}, Test: {test_data.shape}")

    label_encoder, _, _, _ = load_artifacts()

    train_dataset = SparseDataset(train_data, label_encoder)
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    val_dataset = SparseDataset(val_data, label_encoder)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)
    test_dataset = SparseDataset(test_data, label_encoder)
    test_loader  = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    return train_loader, val_loader, test_loader

# -------------------------- 2. Preprocessing Steps --------------------------

def perform_basic_preprocessing(adata: AnnData, target_sum: float = 1e4) -> None:
    """
    Performs basic scRNA-seq preprocessing:
    1. Clears .raw
    2. Total-count normalization
    3. Log-transformation
    """
    adata.raw = None
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    logger.info("✅ Performed total-count normalization and log1p transformation.")


def select_highly_variable_genes(adata: AnnData, n_top_genes: int = N_HVGS) -> None:
    """
    Identifies and keeps a specified number of highly variable genes (HVGs).
    Filters the AnnData object in place.
    """
    # HVG detection uses a different approach for raw counts, but here it's 
    # applied to the already normalized/log1p data as seen in many workflows.
    sc.pp.highly_variable_genes(
        adata, 
        n_top_genes=n_top_genes,
        subset=True # Filter data matrix to HVGs
    )
    logger.info(f"✅ Selected {n_top_genes} highly variable genes.")


def scale_data(adata: AnnData, means: np.ndarray = None, stds: np.ndarray = None) -> tuple[AnnData, np.ndarray, np.ndarray]:
    """
    Scales the data (features to mean=0, std=1).
    If means/stds are provided (e.g., from training set), they are applied 
    to the current data (e.g., val/test set).
    """
    if means is None or stds is None:
        # Fit scaling on the current data (likely the training set)
        logger.info("💡 Fitting and applying scaling statistics...")
        sc.pp.scale(adata, max_value=10, copy=False)
        # Scanpy's scale does not return the stats directly, so we must calculate
        # the means/stds from the scaled data or use the internally stored values
        # If the input was the training set, the scaling stats are derived here
        # For simplicity, we assume the input is the training set if stats are None:
        computed_means = adata.var['mean'].values if 'mean' in adata.var else np.zeros(adata.n_vars)
        computed_stds = adata.var['std'].values if 'std' in adata.var else np.ones(adata.n_vars)
        return adata, computed_means, computed_stds
    else:
        # Apply precomputed scaling to val/test set
        logger.info("💡 Applying precomputed scaling statistics...")
        X_scaled = (adata.X - means) / stds
        adata.X = X_scaled.copy()
        return adata, means, stds

# ------------------------- 3. Split -------------------------


def split_data(adata: AnnData, test_frac: float = 0.3, random_state: int = 42) -> tuple[AnnData, AnnData, AnnData]:
    """
    Splits the AnnData object into train, validation, and test sets based on cell index.
    The split is stratified by cell type labels.
    """
    
    train_idx, temp_idx = train_test_split(
        np.arange(adata.n_obs), test_size=test_frac, random_state=random_state, stratify=adata.obs[LABEL_COLUMN]
    )

    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=random_state, stratify=adata.obs[LABEL_COLUMN].iloc[temp_idx]
    )

    # Create AnnData subsets (views)
    train_data = adata[train_idx].copy()
    val_data = adata[val_idx].copy()
    test_data = adata[test_idx].copy()

    logger.info(f"✅ Data split: Train={train_data.n_obs}, Val={val_data.n_obs}, Test={test_data.n_obs}")

    return train_data, val_data, test_data

# ------------------------- 4. Artifacts and I/O -------------------------


def load_artifacts(artifacts_dir: str = ARTIFACTS_DIR) -> tuple[LabelEncoder, np.ndarray, np.ndarray, np.ndarray]:
    """Loads the LabelEncoder and scaling statistics from the artifacts directory."""
    le = joblib.load(os.path.join(artifacts_dir, "label_encoder.joblib"))
    gene_order = np.load(os.path.join(artifacts_dir, "gene_order.npy"), allow_pickle=True)
    train_means = np.load(os.path.join(artifacts_dir, "train_means.npy"), allow_pickle=True)
    train_stds = np.load(os.path.join(artifacts_dir, "train_stds.npy"), allow_pickle=True)

    logger.info(f"✅ Loaded LabelEncoder, gene order, and scaling stats.")
    return le, gene_order, train_means, train_stds

def compute_embeddings(model, dataloader, device):
    model.eval()
    embeddings, labels = [], []
    with torch.no_grad():
        for xb, yb in tqdm(dataloader):
            xb = xb.to(device)
            emb = model(xb)
            embeddings.append(emb.cpu())
            labels.append(yb.cpu())
    return torch.cat(embeddings), torch.cat(labels)

def evaluate_knn(train_emb, train_labels, val_emb, val_labels,
                test_emb, test_labels, label_encoder, model_name="Model", save_model=False,
                save_dir=ARTIFACTS_DIR):
    # Fit KNN on training embeddings
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    knn.fit(train_emb.numpy(), train_labels.numpy())
    # Optionally save the KNN model
    if save_model:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"knn_{model_name.lower()}.joblib")
        joblib.dump(knn, save_path)
        logger.info(f"✅ Saved KNN model to: {save_path}")
    
    # Dictionary to store metrics
    results = {}
    
    # Helper to compute metrics and confusion matrix
    def compute_metrics(x, y, dataset_name):
        preds = knn.predict(x.numpy())
        acc = accuracy_score(y, preds) * 100
        f1  = f1_score(y, preds, average='macro') * 100
        cm = confusion_matrix(y, preds)
        results[f"{model_name}_{dataset_name}_acc"] = acc
        results[f"{model_name}_{dataset_name}_f1"] = f1
        results[f"{model_name}_{dataset_name}_cm"] = cm
        
        # Plot heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=label_encoder.classes_,
                    yticklabels=label_encoder.classes_)
        plt.xlabel("Predicted class")
        plt.ylabel("True class")
        plt.title(f"{dataset_name} Confusion Matrix - {model_name}")
        plt.xticks()
        plt.yticks()
        plt.show()
    
    # Validation
    compute_metrics(val_emb, val_labels, "Val")
    # Test
    compute_metrics(test_emb, test_labels, "Test")
    
    # Print results neatly
    for key, value in results.items():
        if "cm" not in key:
            print(f"{key}: {value:.2f}%")
    
    return results


