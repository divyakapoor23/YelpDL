import os
import json
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# =========================================================
# Configuration
# =========================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Update these paths to match your machine
DATA_DIR = Path("./yelp_dataset")
BUSINESS_PATH = DATA_DIR / "business.json"
REVIEW_PATH = DATA_DIR / "review.json"
PHOTO_META_PATH = DATA_DIR / "photos.json"
PHOTO_DIR = DATA_DIR / "photos"

OUTPUT_DIR = Path("./outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_REVIEWS = 120000          # use smaller sample first if dataset is large
MAX_PHOTOS = None             # set to an integer for faster experimentation
MIN_REVIEWS_PER_BUSINESS = 3  # optional filter for data quality
MAX_TEXT_LEN = 150
VOCAB_SIZE = 20000
EMBED_DIM = 128
LSTM_HIDDEN = 128
REGION_EMBED_DIM = 16
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 1e-4
IMAGE_SIZE = 224
NUM_WORKERS = 2


# =========================================================
# Helpers
# =========================================================
def read_json_lines(path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if nrows is not None and i >= nrows:
                break
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def save_metrics(metrics: dict, filename: str) -> None:
    out_path = OUTPUT_DIR / filename
    pd.DataFrame([metrics]).to_csv(out_path, index=False)
    print(f"Saved metrics to {out_path}")


def compute_metrics(y_true: List[int], y_pred: List[int]) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


# =========================================================
# Step 1: Load Yelp data
# =========================================================
def load_yelp_data(
    business_path: Path,
    review_path: Path,
    photo_meta_path: Path,
    max_reviews: Optional[int] = None,
    max_photos: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("Loading business data...")
    business_df = read_json_lines(business_path)

    print("Loading review data...")
    review_df = read_json_lines(review_path, nrows=max_reviews)

    print("Loading photo metadata...")
    photo_df = read_json_lines(photo_meta_path, nrows=max_photos)

    print(f"Business shape: {business_df.shape}")
    print(f"Review shape: {review_df.shape}")
    print(f"Photo metadata shape: {photo_df.shape}")
    return business_df, review_df, photo_df


# =========================================================
# Step 2: Build multimodal dataset
# =========================================================
def prepare_multimodal_dataframe(
    business_df: pd.DataFrame,
    review_df: pd.DataFrame,
    photo_df: pd.DataFrame,
    photo_dir: Path,
) -> pd.DataFrame:
    # Keep useful business columns
    business_keep = business_df[[
        "business_id", "name", "city", "state", "categories", "stars", "review_count"
    ]].copy()

    # Optional quality filter
    business_keep = business_keep[business_keep["review_count"] >= MIN_REVIEWS_PER_BUSINESS]

    # Keep useful review columns
    review_keep = review_df[["review_id", "business_id", "stars", "text"]].copy()
    review_keep = review_keep.rename(columns={"stars": "review_stars", "text": "review_text"})

    # Create sentiment label from stars
    # binary setting: positive = 4 or 5, negative = 1 or 2, drop neutral 3
    review_keep = review_keep[review_keep["review_stars"].isin([1, 2, 4, 5])].copy()
    review_keep["sentiment"] = review_keep["review_stars"].apply(lambda x: 1 if x >= 4 else 0)

    # Keep useful photo metadata
    photo_keep = photo_df[["photo_id", "business_id", "caption", "label"]].copy()

    # Add image path (assumes image files are in PHOTO_DIR and named photo_id.jpg)
    photo_keep["image_path"] = photo_keep["photo_id"].apply(lambda x: str(photo_dir / f"{x}.jpg"))
    photo_keep = photo_keep[photo_keep["image_path"].apply(os.path.exists)].copy()

    # Merge photo + business
    merged = photo_keep.merge(business_keep, on="business_id", how="inner")

    # Merge one review per image by business_id
    # To keep this simple, sample one review for each business-photo pair
    reviews_grouped = review_keep.groupby("business_id").agg(list).reset_index()
    merged = merged.merge(reviews_grouped, on="business_id", how="inner")

    # explode review lists so each row is one image + one review
    merged = merged.explode(["review_id", "review_stars", "review_text", "sentiment"]).reset_index(drop=True)

    # Create region column
    merged["region"] = merged["city"].fillna("Unknown") + "_" + merged["state"].fillna("Unknown")

    # Basic cleanup
    merged = merged.dropna(subset=["image_path", "review_text", "region", "sentiment"])
    merged = merged[merged["review_text"].str.len() > 15].copy()

    print(f"Final multimodal dataframe shape: {merged.shape}")
    return merged


# =========================================================
# Step 3: Text preprocessing
# =========================================================
def fit_tokenizer(train_texts: List[str], vocab_size: int = VOCAB_SIZE) -> Tokenizer:
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_texts)
    return tokenizer


def encode_texts(tokenizer: Tokenizer, texts: List[str], max_len: int = MAX_TEXT_LEN) -> np.ndarray:
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")
    return padded


# =========================================================
# Step 4: Dataset and Dataloader
# =========================================================
class YelpMultimodalDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        text_sequences: np.ndarray,
        region_ids: np.ndarray,
        image_transform=None,
    ):
        self.df = df.reset_index(drop=True)
        self.text_sequences = text_sequences
        self.region_ids = region_ids
        self.image_transform = image_transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = Image.open(row["image_path"]).convert("RGB")
        if self.image_transform:
            image = self.image_transform(image)

        text_seq = torch.tensor(self.text_sequences[idx], dtype=torch.long)
        region_id = torch.tensor(self.region_ids[idx], dtype=torch.long)
        label = torch.tensor(int(row["sentiment"]), dtype=torch.float32)

        return image, text_seq, region_id, label


# =========================================================
# Step 5: Model definitions
# =========================================================
class ImageOnlyModel(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, images):
        features = self.backbone(images)
        logits = self.classifier(features).squeeze(1)
        return logits


class TextOnlyModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, text_seq):
        embedded = self.embedding(text_seq)
        _, (hidden, _) = self.lstm(embedded)
        features = hidden[-1]
        logits = self.classifier(features).squeeze(1)
        return logits


class MultimodalFusionModel(nn.Module):
    def __init__(self, vocab_size: int, num_regions: int):
        super().__init__()

        # Image branch
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        img_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.image_backbone = backbone
        self.image_proj = nn.Linear(img_features, 128)

        # Text branch
        self.text_embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=0)
        self.text_lstm = nn.LSTM(EMBED_DIM, LSTM_HIDDEN, batch_first=True)
        self.text_proj = nn.Linear(LSTM_HIDDEN, 128)

        # Region branch
        self.region_embedding = nn.Embedding(num_regions, REGION_EMBED_DIM)

        # Fusion head
        fusion_input_dim = 128 + 128 + REGION_EMBED_DIM
        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, images, text_seq, region_ids):
        # Image features
        img_feat = self.image_backbone(images)
        img_feat = self.image_proj(img_feat)

        # Text features
        txt_emb = self.text_embedding(text_seq)
        _, (hidden, _) = self.text_lstm(txt_emb)
        txt_feat = self.text_proj(hidden[-1])

        # Region features
        reg_feat = self.region_embedding(region_ids)

        # Fusion
        combined = torch.cat([img_feat, txt_feat, reg_feat], dim=1)
        logits = self.classifier(combined).squeeze(1)
        return logits


# =========================================================
# Step 6: Training and evaluation
# =========================================================
def train_epoch_multimodal(model, dataloader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for images, text_seq, region_ids, labels in dataloader:
        images = images.to(DEVICE)
        text_seq = text_seq.to(DEVICE)
        region_ids = region_ids.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        logits = model(images, text_seq, region_ids)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        preds = (torch.sigmoid(logits) >= 0.5).long().cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().astype(int).tolist())

    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = running_loss / len(dataloader.dataset)
    return metrics


def eval_epoch_multimodal(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, text_seq, region_ids, labels in dataloader:
            images = images.to(DEVICE)
            text_seq = text_seq.to(DEVICE)
            region_ids = region_ids.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(images, text_seq, region_ids)
            loss = criterion(logits, labels)

            running_loss += loss.item() * labels.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).long().cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().astype(int).tolist())

    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = running_loss / len(dataloader.dataset)
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
    return metrics


def train_multimodal_model(train_loader, val_loader, vocab_size: int, num_regions: int):
    model = MultimodalFusionModel(vocab_size=vocab_size, num_regions=num_regions).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_f1 = -1
    best_path = OUTPUT_DIR / "best_multimodal_model.pt"

    history = []

    for epoch in range(1, EPOCHS + 1):
        train_metrics = train_epoch_multimodal(model, train_loader, optimizer, criterion)
        val_metrics = eval_epoch_multimodal(model, val_loader, criterion)

        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Train F1: {train_metrics['f1']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(model.state_dict(), best_path)
            print(f"Saved best model to {best_path}")

    history_df = pd.DataFrame(history)
    history_df.to_csv(OUTPUT_DIR / "multimodal_training_history.csv", index=False)
    return model, history_df


# =========================================================
# Optional: simple ablation baseline hooks
# =========================================================
# In your report, compare:
# 1. ImageOnlyModel
# 2. TextOnlyModel
# 3. MultimodalFusionModel without region
# 4. MultimodalFusionModel with region
# You can reuse the same train/eval structure with minor changes.


# =========================================================
# Step 7: Main script
# =========================================================
def main():
    business_df, review_df, photo_df = load_yelp_data(
        BUSINESS_PATH,
        REVIEW_PATH,
        PHOTO_META_PATH,
        max_reviews=MAX_REVIEWS,
        max_photos=MAX_PHOTOS,
    )

    df = prepare_multimodal_dataframe(
        business_df=business_df,
        review_df=review_df,
        photo_df=photo_df,
        photo_dir=PHOTO_DIR,
    )

    # Optional: downsample for faster first run
    df = df.sample(min(len(df), 30000), random_state=SEED).reset_index(drop=True)

    # Split
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=SEED,
        stratify=df["sentiment"],
    )

    # Tokenizer only on train text
    tokenizer = fit_tokenizer(train_df["review_text"].tolist(), vocab_size=VOCAB_SIZE)
    X_train_text = encode_texts(tokenizer, train_df["review_text"].tolist(), max_len=MAX_TEXT_LEN)
    X_val_text = encode_texts(tokenizer, val_df["review_text"].tolist(), max_len=MAX_TEXT_LEN)

    # Region encoding
    region_encoder = LabelEncoder()
    train_region_ids = region_encoder.fit_transform(train_df["region"])
    val_region_ids = region_encoder.transform(val_df["region"])
    num_regions = len(region_encoder.classes_)
    print(f"Number of regions: {num_regions}")

    # Image transforms
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Datasets
    train_dataset = YelpMultimodalDataset(train_df, X_train_text, train_region_ids, image_transform=train_transform)
    val_dataset = YelpMultimodalDataset(val_df, X_val_text, val_region_ids, image_transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Train
    model, history_df = train_multimodal_model(
        train_loader=train_loader,
        val_loader=val_loader,
        vocab_size=VOCAB_SIZE,
        num_regions=num_regions,
    )

    print("Training complete.")
    print(history_df.tail())


if __name__ == "__main__":
    main()
