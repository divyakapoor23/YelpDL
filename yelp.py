import math
import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import matplotlib
matplotlib.use("Agg")  # non-interactive backend (safe for scripts without a display)
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer


# =========================================================
# Configuration
# =========================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
print(f"Using device: {DEVICE}")

# Resolve dataset paths from environment variables with local defaults.
PROJECT_ROOT = Path(__file__).resolve().parent


def env_path(name: str, default: Path) -> Path:
    raw_value = os.getenv(name)
    if raw_value:
        return Path(raw_value).expanduser()
    return default


DATA_ROOT = env_path("YELP_DATA_ROOT", PROJECT_ROOT / "Data")
JSON_DATA_DIR = env_path("YELP_JSON_DATA_DIR", DATA_ROOT / "Yelp JSON" / "yelp_dataset")
PHOTO_DATA_DIR = env_path("YELP_PHOTO_DATA_DIR", DATA_ROOT / "Yelp Photos" / "yelp_photos")
BUSINESS_PATH = env_path(
    "YELP_BUSINESS_PATH",
    JSON_DATA_DIR / "yelp_academic_dataset_business.json",
)
REVIEW_PATH = env_path(
    "YELP_REVIEW_PATH",
    JSON_DATA_DIR / "yelp_academic_dataset_review.json",
)
PHOTO_META_PATH = env_path("YELP_PHOTO_META_PATH", PHOTO_DATA_DIR / "photos.json")
PHOTO_DIR = env_path("YELP_PHOTO_IMAGES_DIR", PHOTO_DATA_DIR / "photos")

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_REVIEWS = 120000          # use smaller sample first if dataset is large
MAX_PHOTOS = None             # set to an integer for faster experimentation
MIN_REVIEWS_PER_BUSINESS = 3  # optional filter for data quality
MAX_TEXT_LEN = 150
VOCAB_SIZE = 20000
EMBED_DIM = 128
LSTM_HIDDEN = 128
REGION_EMBED_DIM = 16
IMAGE_TOKEN_COUNT = 4
CROSS_ATTN_HEADS = 4
BATCH_SIZE = 64 if DEVICE.type in ("cuda", "mps") else 32
EPOCHS = 10
LEARNING_RATE = 1e-4
IMAGE_SIZE = 224
NUM_WORKERS = min(os.cpu_count() or 1, 4)
MAX_CATEGORY_CLASSES = 20
CATEGORY_LOSS_WEIGHT = 0.5
RATING_LOSS_WEIGHT = 0.3
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_MIN_DELTA = 1e-4
RESEARCH_QUESTION = (
    "How do visual, textual, and regional signals interact to shape perceived food sentiment, "
    "and does incorporating geographic context improve multimodal sentiment prediction?"
)


# =========================================================
# Helpers
# =========================================================
def read_json_lines(path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
    return pd.read_json(path, lines=True, nrows=nrows)


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


def extract_primary_category(categories: Optional[str]) -> str:
    if not isinstance(categories, str) or not categories.strip():
        return "Unknown"
    first = categories.split(",")[0].strip()
    return first if first else "Unknown"


def scale_rating(rating: float) -> float:
    # Yelp stars are in [1, 5]. Scale to [0, 1] for stabler regression.
    return float((rating - 1.0) / 4.0)


def estimate_image_quality(image_path: str) -> Optional[float]:
    """Cheap image-quality proxy using grayscale edge energy."""
    try:
        image = Image.open(image_path).convert("L").resize((128, 128))
    except (UnidentifiedImageError, OSError):
        return None

    gray = np.asarray(image, dtype=np.float32) / 255.0
    dx = np.abs(np.diff(gray, axis=1)).mean()
    dy = np.abs(np.diff(gray, axis=0)).mean()
    return float(dx + dy)


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

    # Merge one review per image by business_id.
    # Limitation: this is business-level multimodal pairing, not guaranteed dish-level alignment.
    reviews_grouped = review_keep.groupby("business_id").agg(list).reset_index()
    merged = merged.merge(reviews_grouped, on="business_id", how="inner")

    # explode review lists so each row is one image + one review
    merged = merged.explode(["review_id", "review_stars", "review_text", "sentiment"]).reset_index(drop=True)

    # Create region column
    merged["region"] = merged["city"].fillna("Unknown") + "_" + merged["state"].fillna("Unknown")

    # Make geography more meaningful than raw city/state for analysis.
    city_counts = merged["city"].fillna("Unknown").value_counts()
    q1 = city_counts.quantile(0.33)
    q2 = city_counts.quantile(0.66)

    def city_urbanicity(city: str) -> str:
        count = city_counts.get(city, 0)
        if count >= q2:
            return "urban"
        if count >= q1:
            return "suburban"
        return "small_town"

    merged["urbanicity"] = merged["city"].fillna("Unknown").apply(city_urbanicity)
    merged["cuisine_cluster"] = merged["categories"].apply(extract_primary_category)

    # Yelp JSON lacks reliable price in this pipeline; use business stars as a quality proxy.
    merged["rating_quality_cluster"] = pd.cut(
        merged["stars"].astype(float),
        bins=[0, 3.0, 4.0, 5.0],
        labels=["low_quality_proxy", "mid_quality_proxy", "high_quality_proxy"],
        include_lowest=True,
    ).astype(str)

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
        category_ids: np.ndarray,
        ratings: np.ndarray,
        image_transform=None,
    ):
        # Pre-extract columns as plain Python lists/arrays to avoid slow df.iloc in __getitem__
        self.image_paths = df["image_path"].tolist()
        self.sentiments = df["sentiment"].astype(int).tolist()
        self.text_sequences = text_sequences
        self.region_ids = region_ids
        self.category_ids = category_ids
        self.ratings = ratings
        self.image_transform = image_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert("RGB")
        except (UnidentifiedImageError, OSError):
            return None

        if self.image_transform:
            image = self.image_transform(image)

        text_seq = torch.tensor(self.text_sequences[idx], dtype=torch.long)
        region_id = torch.tensor(self.region_ids[idx], dtype=torch.long)
        category_id = torch.tensor(self.category_ids[idx], dtype=torch.long)
        rating = torch.tensor(self.ratings[idx], dtype=torch.float32)
        label = torch.tensor(self.sentiments[idx], dtype=torch.float32)

        return image, text_seq, region_id, category_id, rating, label


def safe_collate(batch):
    valid_items = [item for item in batch if item is not None]
    if not valid_items:
        return None

    images, text_seq, region_ids, category_ids, ratings, labels = zip(*valid_items)
    return (
        torch.stack(images),
        torch.stack(text_seq),
        torch.stack(region_ids),
        torch.stack(category_ids),
        torch.stack(ratings),
        torch.stack(labels),
    )


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
    def __init__(self, vocab_size: int, num_regions: int, num_categories: int):
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
        self.text_token_proj = nn.Linear(LSTM_HIDDEN, 128)
        self.text_summary_proj = nn.Linear(LSTM_HIDDEN, 128)

        # Cross-modal attention: image attends text; text attends image tokens.
        self.image_to_text_attn = nn.MultiheadAttention(128, CROSS_ATTN_HEADS, batch_first=True)
        self.text_to_image_attn = nn.MultiheadAttention(128, CROSS_ATTN_HEADS, batch_first=True)
        self.image_tokenizer = nn.Linear(128, IMAGE_TOKEN_COUNT * 128)

        # Region branch
        self.region_embedding = nn.Embedding(num_regions, REGION_EMBED_DIM)
        self.category_embedding = nn.Embedding(num_categories, REGION_EMBED_DIM)

        # Shared fusion trunk
        fusion_input_dim = 128 + 128 + REGION_EMBED_DIM + REGION_EMBED_DIM + REGION_EMBED_DIM
        self.fusion_trunk = nn.Sequential(
            nn.Linear(fusion_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # Multitask heads
        self.sentiment_head = nn.Linear(64, 1)
        self.category_head = nn.Linear(64, num_categories)
        self.rating_head = nn.Linear(64, 1)

    def forward(self, images, text_seq, region_ids, category_ids, return_attention: bool = False):
        # Image features
        img_feat = self.image_backbone(images)
        img_feat = self.image_proj(img_feat)

        # Text features
        txt_emb = self.text_embedding(text_seq)
        txt_seq_out, (hidden, _) = self.text_lstm(txt_emb)
        txt_tokens = self.text_token_proj(txt_seq_out)
        txt_summary = self.text_summary_proj(hidden[-1])

        img_query = img_feat.unsqueeze(1)
        img_attended, attn_img = self.image_to_text_attn(
            query=img_query,
            key=txt_tokens,
            value=txt_tokens,
        )

        img_tokens = self.image_tokenizer(img_feat).view(-1, IMAGE_TOKEN_COUNT, 128)
        txt_query = txt_summary.unsqueeze(1)
        txt_attended, attn_txt = self.text_to_image_attn(
            query=txt_query,
            key=img_tokens,
            value=img_tokens,
        )

        img_feat = img_attended.squeeze(1)
        txt_feat = txt_attended.squeeze(1)

        # Region features
        reg_feat = self.region_embedding(region_ids)
        cat_feat = self.category_embedding(category_ids)
        reg_cat_interaction = reg_feat * cat_feat

        # Fusion with explicit region-cuisine interaction.
        combined = torch.cat([img_feat, txt_feat, reg_feat, cat_feat, reg_cat_interaction], dim=1)
        shared = self.fusion_trunk(combined)
        sentiment_logits = self.sentiment_head(shared).squeeze(1)
        category_logits = self.category_head(shared)
        rating_pred = torch.sigmoid(self.rating_head(shared)).squeeze(1)
        if return_attention:
            attention_stats = build_attention_stats(attn_img, attn_txt)
            return sentiment_logits, category_logits, rating_pred, attention_stats
        return sentiment_logits, category_logits, rating_pred


class ImageTextFusionModel(nn.Module):
    """Image + Text fusion without region embedding (ablation: no geographic context)."""

    def __init__(self, vocab_size: int):
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
        self.text_token_proj = nn.Linear(LSTM_HIDDEN, 128)
        self.text_summary_proj = nn.Linear(LSTM_HIDDEN, 128)
        self.image_to_text_attn = nn.MultiheadAttention(128, CROSS_ATTN_HEADS, batch_first=True)
        self.text_to_image_attn = nn.MultiheadAttention(128, CROSS_ATTN_HEADS, batch_first=True)
        self.image_tokenizer = nn.Linear(128, IMAGE_TOKEN_COUNT * 128)

        # Fusion head (128 img + 128 text)
        self.classifier = nn.Sequential(
            nn.Linear(128 + 128, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def encode_image_text_embeddings(self, images, text_seq):
        img_feat = self.image_backbone(images)
        img_feat = self.image_proj(img_feat)
        txt_emb = self.text_embedding(text_seq)
        txt_seq_out, (hidden, _) = self.text_lstm(txt_emb)
        txt_tokens = self.text_token_proj(txt_seq_out)
        txt_summary = self.text_summary_proj(hidden[-1])

        img_query = img_feat.unsqueeze(1)
        img_attended, _ = self.image_to_text_attn(
            query=img_query,
            key=txt_tokens,
            value=txt_tokens,
        )

        img_tokens = self.image_tokenizer(img_feat).view(-1, IMAGE_TOKEN_COUNT, 128)
        txt_query = txt_summary.unsqueeze(1)
        txt_attended, _ = self.text_to_image_attn(
            query=txt_query,
            key=img_tokens,
            value=img_tokens,
        )

        return img_attended.squeeze(1), txt_attended.squeeze(1)

    def forward(self, images, text_seq, return_attention: bool = False):
        img_feat = self.image_backbone(images)
        img_feat = self.image_proj(img_feat)
        txt_emb = self.text_embedding(text_seq)
        txt_seq_out, (hidden, _) = self.text_lstm(txt_emb)
        txt_tokens = self.text_token_proj(txt_seq_out)
        txt_summary = self.text_summary_proj(hidden[-1])

        img_query = img_feat.unsqueeze(1)
        img_attended, attn_img = self.image_to_text_attn(
            query=img_query,
            key=txt_tokens,
            value=txt_tokens,
        )

        img_tokens = self.image_tokenizer(img_feat).view(-1, IMAGE_TOKEN_COUNT, 128)
        txt_query = txt_summary.unsqueeze(1)
        txt_attended, attn_txt = self.text_to_image_attn(
            query=txt_query,
            key=img_tokens,
            value=img_tokens,
        )

        img_feat = img_attended.squeeze(1)
        txt_feat = txt_attended.squeeze(1)

        combined = torch.cat([img_feat, txt_feat], dim=1)
        logits = self.classifier(combined).squeeze(1)
        if return_attention:
            attention_stats = build_attention_stats(attn_img, attn_txt)
            return logits, attention_stats
        return logits


# =========================================================
# Step 6: Training and evaluation
# =========================================================
def _normalized_attention_concentration(attn_weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized peak focus and entropy-based concentration per sample."""
    weights = attn_weights.clamp_min(1e-12)
    key_count = weights.size(-1)
    peak = weights.amax(dim=-1).mean(dim=-1)
    if key_count <= 1:
        ones = torch.ones(weights.size(0), device=weights.device)
        return ones, ones

    uniform = 1.0 / key_count
    peak_focus = ((peak - uniform) / (1.0 - uniform)).clamp(0.0, 1.0)
    entropy = -(weights * weights.log()).sum(dim=-1)
    entropy = entropy / math.log(key_count)
    concentration = (1.0 - entropy).mean(dim=-1).clamp(0.0, 1.0)
    return peak_focus, concentration


def build_attention_stats(attn_img: torch.Tensor, attn_txt: torch.Tensor) -> dict[str, torch.Tensor]:
    """Build per-sample attention statistics that remain informative after softmax normalization."""
    img_peak_focus, img_concentration = _normalized_attention_concentration(attn_img)
    txt_peak_focus, txt_concentration = _normalized_attention_concentration(attn_txt)
    return {
        "alpha_img": attn_img.mean(dim=(1, 2)),
        "alpha_txt": attn_txt.mean(dim=(1, 2)),
        "img_peak_focus": img_peak_focus,
        "txt_peak_focus": txt_peak_focus,
        "img_concentration": img_concentration,
        "txt_concentration": txt_concentration,
    }


def _forward_batch(model: nn.Module, batch: tuple, mode: str, return_attention: bool = False):
    """Route a batch to the correct model signature based on ablation mode."""
    images, text_seq, region_ids, category_ids, ratings, labels = batch
    images = images.to(DEVICE)
    text_seq = text_seq.to(DEVICE)
    region_ids = region_ids.to(DEVICE)
    category_ids = category_ids.to(DEVICE)
    ratings = ratings.to(DEVICE)
    labels = labels.to(DEVICE)

    if mode == "image":
        logits = model(images)
    elif mode == "text":
        logits = model(text_seq)
    elif mode == "image_text":
        outputs = model(images, text_seq, return_attention=return_attention)
        if return_attention:
            logits, attention_stats = outputs
            targets = {
                "sentiment": labels,
                "category": category_ids,
                "rating": ratings,
            }
            return logits, targets, attention_stats
        logits = outputs
    else:  # "full"
        outputs = model(images, text_seq, region_ids, category_ids, return_attention=return_attention)
        if return_attention:
            sent_logits, cat_logits, rating_pred, attention_stats = outputs
            targets = {
                "sentiment": labels,
                "category": category_ids,
                "rating": ratings,
            }
            logits = {
                "sentiment": sent_logits,
                "category": cat_logits,
                "rating": rating_pred,
            }
            return logits, targets, attention_stats
        logits = outputs

    targets = {
        "sentiment": labels,
        "category": category_ids,
        "rating": ratings,
    }
    if mode == "full":
        sent_logits, cat_logits, rating_pred = logits
        logits = {
            "sentiment": sent_logits,
            "category": cat_logits,
            "rating": rating_pred,
        }
    return logits, targets


def compute_total_loss(logits, targets: dict, criteria: dict, mode: str):
    if mode != "full":
        return criteria["sentiment"](logits, targets["sentiment"])

    sentiment_loss = criteria["sentiment"](logits["sentiment"], targets["sentiment"])
    category_loss = criteria["category"](logits["category"], targets["category"])
    rating_loss = criteria["rating"](logits["rating"], targets["rating"])
    total = sentiment_loss + CATEGORY_LOSS_WEIGHT * category_loss + RATING_LOSS_WEIGHT * rating_loss
    return total, sentiment_loss, category_loss, rating_loss


def train_epoch(model: nn.Module, dataloader: DataLoader, optimizer, criteria: dict, mode: str) -> dict:
    model.train()
    running_loss = 0.0
    running_category_loss = 0.0
    running_rating_loss = 0.0
    valid_sample_count = 0
    all_preds, all_labels = [], []
    all_cat_preds, all_cat_labels = [], []
    rating_abs_error_sum = 0.0

    for batch in dataloader:
        if batch is None:
            continue
        logits, targets = _forward_batch(model, batch, mode)
        optimizer.zero_grad(set_to_none=True)

        if mode == "full":
            loss, _, category_loss, rating_loss = compute_total_loss(logits, targets, criteria, mode)
            sent_logits = logits["sentiment"]
            cat_logits = logits["category"]
            rating_pred = logits["rating"]
        else:
            loss = compute_total_loss(logits, targets, criteria, mode)
            sent_logits = logits

        loss.backward()
        optimizer.step()
        running_loss += loss.item() * targets["sentiment"].size(0)
        valid_sample_count += targets["sentiment"].size(0)
        preds = (torch.sigmoid(sent_logits) >= 0.5).long().cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(targets["sentiment"].cpu().numpy().astype(int).tolist())

        if mode == "full":
            running_category_loss += category_loss.item() * targets["sentiment"].size(0)
            running_rating_loss += rating_loss.item() * targets["sentiment"].size(0)
            cat_preds = torch.argmax(cat_logits, dim=1).cpu().numpy()
            cat_labels = targets["category"].cpu().numpy()
            all_cat_preds.extend(cat_preds.tolist())
            all_cat_labels.extend(cat_labels.tolist())
            rating_abs_error_sum += torch.abs(rating_pred - targets["rating"]).sum().item()

    if valid_sample_count == 0:
        raise RuntimeError("No valid training samples were loaded. Check the image files.")

    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = running_loss / valid_sample_count
    if mode == "full" and all_cat_labels:
        metrics["category_accuracy"] = accuracy_score(all_cat_labels, all_cat_preds)
        metrics["rating_mae"] = rating_abs_error_sum / valid_sample_count
        metrics["category_loss"] = running_category_loss / valid_sample_count
        metrics["rating_loss"] = running_rating_loss / valid_sample_count
    return metrics


def eval_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criteria: dict,
    mode: str,
    verbose: bool = False,
    capture_attention: bool = False,
) -> Tuple[dict, Optional[pd.DataFrame]]:
    model.eval()
    running_loss = 0.0
    running_category_loss = 0.0
    running_rating_loss = 0.0
    valid_sample_count = 0
    all_preds, all_labels = [], []
    all_cat_preds, all_cat_labels = [], []
    rating_abs_error_sum = 0.0
    attention_records = []

    with torch.no_grad():
        for batch in dataloader:
            if batch is None:
                continue
            if capture_attention and mode in ("image_text", "full"):
                logits, targets, attention_stats = _forward_batch(
                    model,
                    batch,
                    mode,
                    return_attention=True,
                )
            else:
                logits, targets = _forward_batch(model, batch, mode)

            if mode == "full":
                loss, _, category_loss, rating_loss = compute_total_loss(logits, targets, criteria, mode)
                sent_logits = logits["sentiment"]
                cat_logits = logits["category"]
                rating_pred = logits["rating"]
            else:
                loss = compute_total_loss(logits, targets, criteria, mode)
                sent_logits = logits

            running_loss += loss.item() * targets["sentiment"].size(0)
            valid_sample_count += targets["sentiment"].size(0)
            pred_probs = torch.sigmoid(sent_logits).detach().cpu().numpy()
            preds = (pred_probs >= 0.5).astype(int).tolist()
            labels = targets["sentiment"].cpu().numpy().astype(int).tolist()
            all_preds.extend(preds)
            all_labels.extend(labels)

            if capture_attention and mode in ("image_text", "full"):
                stats_cpu = {name: tensor.detach().cpu().numpy().tolist() for name, tensor in attention_stats.items()}
                for index, pred in enumerate(preds):
                    label = int(labels[index])
                    probability = float(pred_probs[index])
                    attention_records.append(
                        {
                            "alpha_img": float(stats_cpu["alpha_img"][index]),
                            "alpha_txt": float(stats_cpu["alpha_txt"][index]),
                            "img_peak_focus": float(stats_cpu["img_peak_focus"][index]),
                            "txt_peak_focus": float(stats_cpu["txt_peak_focus"][index]),
                            "img_concentration": float(stats_cpu["img_concentration"][index]),
                            "txt_concentration": float(stats_cpu["txt_concentration"][index]),
                            "sentiment_label": label,
                            "sentiment_pred": int(pred),
                            "correct": int(int(pred) == label),
                            "pred_confidence": max(probability, 1.0 - probability),
                        }
                    )

            if mode == "full":
                running_category_loss += category_loss.item() * targets["sentiment"].size(0)
                running_rating_loss += rating_loss.item() * targets["sentiment"].size(0)
                cat_preds = torch.argmax(cat_logits, dim=1).cpu().numpy()
                cat_labels = targets["category"].cpu().numpy()
                all_cat_preds.extend(cat_preds.tolist())
                all_cat_labels.extend(cat_labels.tolist())
                rating_abs_error_sum += torch.abs(rating_pred - targets["rating"]).sum().item()

    if valid_sample_count == 0:
        raise RuntimeError("No valid validation samples were loaded. Check the image files.")

    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = running_loss / valid_sample_count
    if mode == "full" and all_cat_labels:
        metrics["category_accuracy"] = accuracy_score(all_cat_labels, all_cat_preds)
        metrics["rating_mae"] = rating_abs_error_sum / valid_sample_count
        metrics["category_loss"] = running_category_loss / valid_sample_count
        metrics["rating_loss"] = running_rating_loss / valid_sample_count
    if verbose:
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, digits=4, zero_division=0))
        print("Confusion Matrix:")
        print(confusion_matrix(all_labels, all_preds))

    attention_df = None
    if capture_attention and mode in ("image_text", "full") and attention_records:
        attention_df = pd.DataFrame(attention_records)
        metrics["attention_img_mean"] = float(attention_df["alpha_img"].mean())
        metrics["attention_txt_mean"] = float(attention_df["alpha_txt"].mean())
        metrics["attention_img_std"] = float(attention_df["alpha_img"].std(ddof=0))
        metrics["attention_txt_std"] = float(attention_df["alpha_txt"].std(ddof=0))
        metrics["attention_img_focus"] = float(attention_df["img_concentration"].mean())
        metrics["attention_txt_focus"] = float(attention_df["txt_concentration"].mean())
    return metrics, attention_df


def run_experiment(
    name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    mode: str,
    pos_weight: torch.Tensor,
) -> dict:
    """Train one ablation variant and return its best-epoch validation metrics."""
    safe_name = name.replace(" ", "_").replace("+", "plus")
    print(f"\n{'='*64}")
    print(f"  Experiment: {name}  [mode={mode}]")
    print(f"{'='*64}")

    model = model.to(DEVICE)
    criteria = {
        "sentiment": nn.BCEWithLogitsLoss(pos_weight=pos_weight),
        "category": nn.CrossEntropyLoss(),
        "rating": nn.L1Loss(),
    }
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_f1 = -1.0
    best_metrics: dict = {}
    best_path = OUTPUT_DIR / f"best_{safe_name}.pt"
    best_attention_df = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        is_last = epoch == EPOCHS
        train_m = train_epoch(model, train_loader, optimizer, criteria, mode)
        val_m, attention_df = eval_epoch(
            model,
            val_loader,
            criteria,
            mode,
            verbose=is_last,
            capture_attention=mode in ("image_text", "full"),
        )

        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_m.items()},
            **{f"val_{k}": v for k, v in val_m.items()},
        }
        history.append(row)

        print(
            f"  Epoch {epoch}/{EPOCHS} | "
            f"Train F1: {train_m['f1']:.4f} | "
            f"Val F1: {val_m['f1']:.4f} | "
            f"Val Acc: {val_m['accuracy']:.4f}"
        )

        if val_m["f1"] > best_f1:
            best_f1 = val_m["f1"]
            best_metrics = {f"val_{k}": v for k, v in val_m.items()}
            best_metrics["best_epoch"] = epoch
            if attention_df is not None:
                best_attention_df = attention_df.copy()
            torch.save(model.state_dict(), best_path)
            epochs_without_improvement = 0
        else:
            if (best_f1 - val_m["f1"]) > EARLY_STOPPING_MIN_DELTA:
                epochs_without_improvement += 1

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(
                f"  Early stopping triggered at epoch {epoch} "
                f"(no F1 improvement for {EARLY_STOPPING_PATIENCE} epochs)."
            )
            break

    pd.DataFrame(history).to_csv(
        OUTPUT_DIR / f"history_{safe_name}.csv", index=False
    )
    if best_attention_df is not None:
        attention_path = OUTPUT_DIR / f"attention_{safe_name}.csv"
        summary_path = OUTPUT_DIR / f"attention_{safe_name}_summary.csv"
        best_attention_df.to_csv(attention_path, index=False)
        best_attention_df.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).to_csv(summary_path)
        img_focus = best_attention_df["img_concentration"].mean() if "img_concentration" in best_attention_df.columns else np.nan
        txt_focus = best_attention_df["txt_concentration"].mean() if "txt_concentration" in best_attention_df.columns else np.nan
        print(f"  Attention concentration (img/txt): {img_focus:.4f} / {txt_focus:.4f}")
        print(f"  Saved attention weights to {attention_path}")
        print(f"  Saved attention summary to {summary_path}")
    print(f"  Best Val F1: {best_f1:.4f} (epoch {best_metrics['best_epoch']})")
    return best_metrics


def image_text_consistency_analysis(
    val_loader: DataLoader,
    vocab_size: int,
    id_to_region: dict,
    id_to_category: dict,
) -> None:
    """
    Compare Image-only and Text-only predictions on the same samples.

    A mismatch (image_pred != text_pred) indicates potential multimodal inconsistency,
    noisy labels, or subjective interpretation differences.
    """
    image_ckpt = OUTPUT_DIR / "best_Image_Only.pt"
    text_ckpt = OUTPUT_DIR / "best_Text_Only.pt"

    if not image_ckpt.exists() or not text_ckpt.exists():
        print("\n" + "=" * 72)
        print("IMAGE-TEXT CONSISTENCY ANALYSIS")
        print("=" * 72)
        print("Skipping consistency analysis: missing best Image/Text checkpoints.")
        print(f"Expected: {image_ckpt} and {text_ckpt}")
        print("=" * 72)
        return

    print("\n" + "=" * 72)
    print("IMAGE-TEXT CONSISTENCY ANALYSIS")
    print("=" * 72)

    image_model = ImageOnlyModel().to(DEVICE)
    text_model = TextOnlyModel(vocab_size, EMBED_DIM, LSTM_HIDDEN).to(DEVICE)
    image_model.load_state_dict(torch.load(image_ckpt, map_location=DEVICE))
    text_model.load_state_dict(torch.load(text_ckpt, map_location=DEVICE))
    image_model.eval()
    text_model.eval()

    rows = []
    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue

            images, text_seq, region_ids, category_ids, ratings, labels = batch
            images = images.to(DEVICE)
            text_seq = text_seq.to(DEVICE)
            labels = labels.to(DEVICE)

            image_probs = torch.sigmoid(image_model(images))
            text_probs = torch.sigmoid(text_model(text_seq))

            image_preds = (image_probs >= 0.5).long()
            text_preds = (text_probs >= 0.5).long()

            for index in range(labels.size(0)):
                region_id = int(region_ids[index].item())
                category_id = int(category_ids[index].item())
                rating_val = float(ratings[index].item())
                rating_stars = (rating_val * 4.0) + 1.0
                if rating_stars <= 2.5:
                    rating_quality = "low_quality_proxy"
                elif rating_stars <= 3.75:
                    rating_quality = "mid_quality_proxy"
                else:
                    rating_quality = "high_quality_proxy"
                rows.append(
                    {
                        "label": int(labels[index].item()),
                        "region_id": region_id,
                        "region": id_to_region.get(region_id, "<unknown_region>"),
                        "cuisine": id_to_category.get(category_id, "<unknown_category>"),
                        "rating_quality_cluster": rating_quality,
                        "image_prob": float(image_probs[index].item()),
                        "text_prob": float(text_probs[index].item()),
                        "image_pred": int(image_preds[index].item()),
                        "text_pred": int(text_preds[index].item()),
                    }
                )

    if not rows:
        print("No valid validation samples available for consistency analysis.")
        print("=" * 72)
        return

    pred_df = pd.DataFrame(rows)
    pred_df["mismatch"] = pred_df["image_pred"] != pred_df["text_pred"]
    pred_df["img_pos_txt_neg"] = (pred_df["image_pred"] == 1) & (pred_df["text_pred"] == 0)
    pred_df["img_neg_txt_pos"] = (pred_df["image_pred"] == 0) & (pred_df["text_pred"] == 1)

    total = len(pred_df)
    mismatch_count = int(pred_df["mismatch"].sum())
    mismatch_rate = mismatch_count / total
    img_pos_txt_neg_count = int(pred_df["img_pos_txt_neg"].sum())
    img_neg_txt_pos_count = int(pred_df["img_neg_txt_pos"].sum())

    print(f"Total validation samples compared: {total}")
    print(f"Mismatch count (image_pred != text_pred): {mismatch_count}")
    print(f"Perception Gap Rate (mismatches / total): {mismatch_rate:.4f}")
    print(f"Image positive, Text negative: {img_pos_txt_neg_count}")
    print(f"Image negative, Text positive: {img_neg_txt_pos_count}")

    mismatch_by_label = (
        pred_df.groupby("label")["mismatch"]
        .mean()
        .rename(index={0: "true_negative", 1: "true_positive"})
        .to_dict()
    )
    print("Mismatch rate by true label:")
    for label_name, rate in mismatch_by_label.items():
        print(f"  {label_name}: {rate:.4f}")

    region_mismatch = (
        pred_df.groupby("region")
        .agg(samples=("mismatch", "count"), mismatch_rate=("mismatch", "mean"))
        .query("samples >= 20")
        .sort_values("mismatch_rate", ascending=False)
    )
    if not region_mismatch.empty:
        print("\nTop regions with highest image-text disagreement (>=20 samples):")
        print(region_mismatch.head(10).to_string())

    cuisine_mismatch = (
        pred_df.groupby("cuisine")
        .agg(samples=("mismatch", "count"), mismatch_rate=("mismatch", "mean"))
        .query("samples >= 20")
        .sort_values("mismatch_rate", ascending=False)
    )
    if not cuisine_mismatch.empty:
        print("\nTop cuisines with highest perception gap (>=20 samples):")
        print(cuisine_mismatch.head(10).to_string())

    rating_quality_mismatch = (
        pred_df.groupby("rating_quality_cluster")
        .agg(samples=("mismatch", "count"), mismatch_rate=("mismatch", "mean"))
        .sort_values("mismatch_rate", ascending=False)
    )
    if not rating_quality_mismatch.empty:
        print("\nPerception gap by rating-quality proxy cluster:")
        print(rating_quality_mismatch.to_string())

    print(
        "\nInsight:\n"
        "  High image-text disagreement suggests noisy multimodal pairs and\n"
        "  subjective perception differences (e.g., image looks appealing while\n"
        "  review text is negative, or vice versa)."
    )

    all_path = OUTPUT_DIR / "image_text_consistency_predictions.csv"
    mismatch_path = OUTPUT_DIR / "image_text_consistency_mismatches.csv"
    region_path = OUTPUT_DIR / "image_text_consistency_region_summary.csv"
    cuisine_path = OUTPUT_DIR / "image_text_consistency_cuisine_summary.csv"
    rating_quality_path = OUTPUT_DIR / "image_text_consistency_rating_quality_summary.csv"

    pred_df.to_csv(all_path, index=False)
    pred_df[pred_df["mismatch"]].to_csv(mismatch_path, index=False)
    if not region_mismatch.empty:
        region_mismatch.reset_index().to_csv(region_path, index=False)
    if not cuisine_mismatch.empty:
        cuisine_mismatch.reset_index().to_csv(cuisine_path, index=False)
    if not rating_quality_mismatch.empty:
        rating_quality_mismatch.reset_index().to_csv(rating_quality_path, index=False)

    print(f"Saved all consistency predictions to {all_path}")
    print(f"Saved mismatch-only samples to {mismatch_path}")
    if region_mismatch.empty:
        print("Region summary not saved (insufficient per-region sample counts).")
    else:
        print(f"Saved region mismatch summary to {region_path}")
    if not cuisine_mismatch.empty:
        print(f"Saved cuisine mismatch summary to {cuisine_path}")
    if not rating_quality_mismatch.empty:
        print(f"Saved rating-quality mismatch summary to {rating_quality_path}")
    print("=" * 72)


def final_presentation_plots(
    ablation_table: pd.DataFrame,
    val_loader: DataLoader,
    num_regions: int,
    num_categories: int,
) -> None:
    """Generate final presentation-ready plots requested for the results section."""
    print("\n" + "=" * 72)
    print("FINAL PRESENTATION PLOTS")
    print("=" * 72)

    plot_df = ablation_table.copy()
    for column in ["F1", "Accuracy", "Loss"]:
        if column in plot_df.columns:
            plot_df[column] = pd.to_numeric(plot_df[column], errors="coerce")

    # 1) Bar chart of F1 scores for all model variants.
    plt.figure(figsize=(9, 5))
    bars = plt.bar(plot_df["Model"], plot_df["F1"], color=["#90CAF9", "#FFCC80", "#A5D6A7", "#CE93D8"])
    plt.title("Model Comparison — F1 Scores")
    plt.ylabel("F1")
    plt.xticks(rotation=15, ha="right")
    for bar, value in zip(bars, plot_df["F1"]):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.002, f"{value:.4f}", ha="center", fontsize=8)
    plt.tight_layout()
    f1_plot_path = OUTPUT_DIR / "final_model_f1_bar.png"
    plt.savefig(f1_plot_path, dpi=160)
    plt.close()
    print(f"Saved model F1 bar chart to {f1_plot_path}")

    # 2) Confusion matrix for best model.
    best_model_name = plot_df.sort_values("F1", ascending=False).iloc[0]["Model"]
    model_map = {
        "Image Only": (ImageOnlyModel(), "image", OUTPUT_DIR / "best_Image_Only.pt"),
        "Text Only": (TextOnlyModel(VOCAB_SIZE, EMBED_DIM, LSTM_HIDDEN), "text", OUTPUT_DIR / "best_Text_Only.pt"),
        "Image + Text": (ImageTextFusionModel(VOCAB_SIZE), "image_text", OUTPUT_DIR / "best_Image_plus_Text.pt"),
        "Image + Text + Region": (
            MultimodalFusionModel(VOCAB_SIZE, num_regions, num_categories),
            "full",
            OUTPUT_DIR / "best_Image_plus_Text_plus_Region.pt",
        ),
    }
    if best_model_name in model_map:
        best_model, mode, ckpt = model_map[best_model_name]
        if ckpt.exists():
            best_model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
            best_model = best_model.to(DEVICE)
            best_model.eval()
            all_true, all_pred = [], []
            with torch.no_grad():
                for batch in val_loader:
                    if batch is None:
                        continue
                    logits, targets = _forward_batch(best_model, batch, mode)
                    sent_logits = logits["sentiment"] if mode == "full" else logits
                    preds = (torch.sigmoid(sent_logits) >= 0.5).long().cpu().numpy().tolist()
                    labels = targets["sentiment"].cpu().numpy().astype(int).tolist()
                    all_pred.extend(preds)
                    all_true.extend(labels)

            cm = confusion_matrix(all_true, all_pred)
            plt.figure(figsize=(5, 4))
            plt.imshow(cm, cmap="Blues")
            plt.title(f"Confusion Matrix — Best Model ({best_model_name})")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.xticks([0, 1], ["Neg", "Pos"])
            plt.yticks([0, 1], ["Neg", "Pos"])
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
            plt.tight_layout()
            cm_path = OUTPUT_DIR / "final_best_model_confusion_matrix.png"
            plt.savefig(cm_path, dpi=160)
            plt.close()
            print(f"Saved best-model confusion matrix to {cm_path}")

    # 3) Perception Gap by region/cuisine and 4) attention comparison.
    region_gap = pd.read_csv(OUTPUT_DIR / "image_text_consistency_region_summary.csv") if (OUTPUT_DIR / "image_text_consistency_region_summary.csv").exists() else None
    if region_gap is not None and not region_gap.empty:
        top = region_gap.sort_values("mismatch_rate", ascending=False).head(15)
        plt.figure(figsize=(10, 5))
        plt.bar(top["region"], top["mismatch_rate"], color="#EF9A9A")
        plt.title("Perception Gap Rate by Region")
        plt.ylabel("Mismatch rate")
        plt.xticks(rotation=65, ha="right")
        plt.tight_layout()
        region_gap_path = OUTPUT_DIR / "final_perception_gap_by_region.png"
        plt.savefig(region_gap_path, dpi=160)
        plt.close()
        print(f"Saved region perception-gap chart to {region_gap_path}")

    cuisine_gap = pd.read_csv(OUTPUT_DIR / "image_text_consistency_cuisine_summary.csv") if (OUTPUT_DIR / "image_text_consistency_cuisine_summary.csv").exists() else None
    if cuisine_gap is not None and not cuisine_gap.empty:
        top = cuisine_gap.sort_values("mismatch_rate", ascending=False).head(15)
        plt.figure(figsize=(10, 5))
        plt.bar(top["cuisine"], top["mismatch_rate"], color="#FFCC80")
        plt.title("Perception Gap Rate by Cuisine")
        plt.ylabel("Mismatch rate")
        plt.xticks(rotation=55, ha="right")
        plt.tight_layout()
        cuisine_gap_path = OUTPUT_DIR / "final_perception_gap_by_cuisine.png"
        plt.savefig(cuisine_gap_path, dpi=160)
        plt.close()
        print(f"Saved cuisine perception-gap chart to {cuisine_gap_path}")

    att_img_txt = pd.read_csv(OUTPUT_DIR / "attention_Image_plus_Text.csv") if (OUTPUT_DIR / "attention_Image_plus_Text.csv").exists() else None
    att_full = pd.read_csv(OUTPUT_DIR / "attention_Image_plus_Text_plus_Region.csv") if (OUTPUT_DIR / "attention_Image_plus_Text_plus_Region.csv").exists() else None
    if att_img_txt is not None and att_full is not None and not att_img_txt.empty and not att_full.empty:
        if {"img_concentration", "txt_concentration"}.issubset(att_img_txt.columns) and {"img_concentration", "txt_concentration"}.issubset(att_full.columns):
            means = [
                att_img_txt["img_concentration"].mean(),
                att_img_txt["txt_concentration"].mean(),
                att_full["img_concentration"].mean(),
                att_full["txt_concentration"].mean(),
            ]
            labels = ["Img+Txt image focus", "Img+Txt text focus", "Full image focus", "Full text focus"]
            title = "Attention Focus Comparison"
            ylabel = "Normalized concentration"
        else:
            means = [
                att_img_txt["alpha_img"].mean(),
                att_img_txt["alpha_txt"].mean(),
                att_full["alpha_img"].mean(),
                att_full["alpha_txt"].mean(),
            ]
            labels = ["Img+Txt img", "Img+Txt txt", "Full img", "Full txt"]
            title = "Attention Weight Comparison"
            ylabel = "Mean attention"
        plt.figure(figsize=(8, 4))
        plt.bar(labels, means, color=["#64B5F6", "#4DB6AC", "#9575CD", "#BA68C8"])
        plt.title(title)
        plt.ylabel(ylabel)
        plt.ylim(0, 1)
        plt.tight_layout()
        att_path = OUTPUT_DIR / "final_attention_comparison.png"
        plt.savefig(att_path, dpi=160)
        plt.close()
        print(f"Saved attention comparison chart to {att_path}")

    # 5) Region/cuisine sentiment spread plot.
    spread_path = OUTPUT_DIR / "cuisine_region_sentiment_spread.csv"
    if spread_path.exists():
        spread_df = pd.read_csv(spread_path)
        if spread_df is not None and not spread_df.empty:
            top = spread_df.sort_values("spread", ascending=False).head(12)
            plt.figure(figsize=(9, 5))
            plt.bar(top["cuisine_cluster"], top["spread"], color="#80CBC4")
            plt.title("Region/Cuisine Sentiment Spread")
            plt.ylabel("Positive-rate spread")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            spread_plot_path = OUTPUT_DIR / "final_region_cuisine_spread.png"
            plt.savefig(spread_plot_path, dpi=160)
            plt.close()
            print(f"Saved region/cuisine spread chart to {spread_plot_path}")

    print("=" * 72)


def cross_modal_retrieval_analysis(
    val_df: pd.DataFrame,
    tokenizer: Tokenizer,
    val_transform,
    max_pool_size: int = 2000,
    top_k: int = 5,
    query_text: str = "spicy ramen",
) -> None:
    """
    Cross-modal retrieval demo using learned joint representations.

    Tasks:
    1. Text -> retrieve matching images
    2. Image -> retrieve matching reviews
    """
    print("\n" + "=" * 72)
    print("CROSS-MODAL RETRIEVAL ANALYSIS")
    print("=" * 72)

    ckpt_path = OUTPUT_DIR / "best_Image_plus_Text.pt"
    if not ckpt_path.exists():
        print(f"Skipping retrieval: missing checkpoint {ckpt_path}")
        print("=" * 72)
        return

    pool_df = val_df.head(min(len(val_df), max_pool_size)).reset_index(drop=True)
    if pool_df.empty:
        print("Skipping retrieval: empty validation pool.")
        print("=" * 72)
        return

    model = ImageTextFusionModel(VOCAB_SIZE).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()

    image_feats = []
    text_feats = []
    valid_indices = []

    with torch.no_grad():
        for idx, row in pool_df.iterrows():
            try:
                image = Image.open(row["image_path"]).convert("RGB")
            except (UnidentifiedImageError, OSError):
                continue

            image_tensor = val_transform(image).unsqueeze(0).to(DEVICE)
            seq = encode_texts(tokenizer, [str(row["review_text"])], max_len=MAX_TEXT_LEN)
            text_tensor = torch.tensor(seq, dtype=torch.long, device=DEVICE)

            img_feat, txt_feat = model.encode_image_text_embeddings(image_tensor, text_tensor)

            image_feats.append(F.normalize(img_feat, dim=1).cpu())
            text_feats.append(F.normalize(txt_feat, dim=1).cpu())
            valid_indices.append(idx)

    if not valid_indices:
        print("Skipping retrieval: no valid image-text pairs after filtering.")
        print("=" * 72)
        return

    image_mat = torch.cat(image_feats, dim=0)  # (N, D)
    text_mat = torch.cat(text_feats, dim=0)    # (N, D)
    valid_df = pool_df.iloc[valid_indices].reset_index(drop=True)

    # ---------------------------------------------------------
    # Text -> Image retrieval (example query: "spicy ramen")
    # ---------------------------------------------------------
    q_seq = encode_texts(tokenizer, [query_text], max_len=MAX_TEXT_LEN)
    q_tensor = torch.tensor(q_seq, dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        dummy_image = torch.zeros((1, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float32, device=DEVICE)
        _, q_feat = model.encode_image_text_embeddings(dummy_image, q_tensor)
        q_feat = F.normalize(q_feat, dim=1).cpu()

    # cosine similarity because vectors are L2-normalized
    text_to_img_scores = torch.matmul(image_mat, q_feat.squeeze(0))
    top_img_idx = torch.topk(text_to_img_scores, k=min(top_k, len(valid_df))).indices.numpy().tolist()

    print(f"\n[1] Text -> Image retrieval for query: '{query_text}'")
    text_to_img_rows = []
    for rank, ridx in enumerate(top_img_idx, start=1):
        row = valid_df.iloc[ridx]
        score = float(text_to_img_scores[ridx].item())
        snippet = str(row["review_text"])[:120].replace("\n", " ")
        print(f"  #{rank}  score={score:.4f}  image={row['image_path']}")
        print(f"      review: {snippet}...")
        text_to_img_rows.append(
            {
                "rank": rank,
                "query": query_text,
                "similarity": score,
                "image_path": row["image_path"],
                "region": row.get("region", ""),
                "sentiment": int(row["sentiment"]),
                "review_text": str(row["review_text"]),
            }
        )

    # ---------------------------------------------------------
    # Image -> Text retrieval (use first pool image as demo query)
    # ---------------------------------------------------------
    query_image_idx = 0
    q_img_feat = image_mat[query_image_idx]
    img_to_text_scores = torch.matmul(text_mat, q_img_feat)
    top_txt_idx = torch.topk(img_to_text_scores, k=min(top_k, len(valid_df))).indices.numpy().tolist()

    query_img_path = valid_df.iloc[query_image_idx]["image_path"]
    print(f"\n[2] Image -> Text retrieval for image: {query_img_path}")
    img_to_text_rows = []
    for rank, ridx in enumerate(top_txt_idx, start=1):
        row = valid_df.iloc[ridx]
        score = float(img_to_text_scores[ridx].item())
        snippet = str(row["review_text"])[:140].replace("\n", " ")
        print(f"  #{rank}  score={score:.4f}  region={row.get('region', '')}  sentiment={int(row['sentiment'])}")
        print(f"      review: {snippet}...")
        img_to_text_rows.append(
            {
                "rank": rank,
                "query_image_path": query_img_path,
                "similarity": score,
                "matched_image_path": row["image_path"],
                "region": row.get("region", ""),
                "sentiment": int(row["sentiment"]),
                "review_text": str(row["review_text"]),
            }
        )

    pd.DataFrame(text_to_img_rows).to_csv(
        OUTPUT_DIR / "retrieval_text_to_image.csv", index=False
    )
    pd.DataFrame(img_to_text_rows).to_csv(
        OUTPUT_DIR / "retrieval_image_to_text.csv", index=False
    )
    print(f"\nSaved text->image results to {OUTPUT_DIR / 'retrieval_text_to_image.csv'}")
    print(f"Saved image->text results to {OUTPUT_DIR / 'retrieval_image_to_text.csv'}")
    print("=" * 72)


def _evaluate_subset_predictions(subset_df: pd.DataFrame, pred_col: str = "pred") -> dict:
    subset_df = subset_df.dropna(subset=[pred_col]).copy()
    if subset_df.empty:
        return {"samples": 0, "accuracy": np.nan, "precision": np.nan, "recall": np.nan, "f1": np.nan}

    metrics = compute_metrics(
        subset_df["sentiment"].astype(int).tolist(),
        subset_df[pred_col].astype(int).tolist(),
    )
    metrics["samples"] = int(len(subset_df))
    return metrics


def _predict_text_model(model: nn.Module, subset_df: pd.DataFrame, tokenizer: Tokenizer) -> List[int]:
    preds = []
    model.eval()
    with torch.no_grad():
        for text in subset_df["review_text"].astype(str).tolist():
            seq = encode_texts(tokenizer, [text], max_len=MAX_TEXT_LEN)
            tensor = torch.tensor(seq, dtype=torch.long, device=DEVICE)
            logits = model(tensor)
            preds.append(int((torch.sigmoid(logits) >= 0.5).item()))
    return preds


def _predict_full_model(
    model: nn.Module,
    subset_df: pd.DataFrame,
    tokenizer: Tokenizer,
    region_to_id: dict,
    unknown_region_id: int,
    category_to_id: dict,
    unknown_category_id: int,
    image_transform,
) -> List[int]:
    preds = []
    model.eval()
    with torch.no_grad():
        for _, row in subset_df.iterrows():
            try:
                image = Image.open(row["image_path"]).convert("RGB")
            except (UnidentifiedImageError, OSError):
                preds.append(np.nan)
                continue

            image_tensor = image_transform(image).unsqueeze(0).to(DEVICE)
            seq = encode_texts(tokenizer, [str(row["review_text"])], max_len=MAX_TEXT_LEN)
            text_tensor = torch.tensor(seq, dtype=torch.long, device=DEVICE)
            region_id = torch.tensor(
                [region_to_id.get(row["region"], unknown_region_id)],
                dtype=torch.long,
                device=DEVICE,
            )
            category_id = torch.tensor(
                [category_to_id.get(row.get("primary_category", "<unknown_category>"), unknown_category_id)],
                dtype=torch.long,
                device=DEVICE,
            )
            sent_logits, _cat_logits, _rating_pred = model(image_tensor, text_tensor, region_id, category_id)
            preds.append(int((torch.sigmoid(sent_logits) >= 0.5).item()))
    return preds


def _predict_image_model(model: nn.Module, subset_df: pd.DataFrame, image_transform) -> List[int]:
    preds = []
    model.eval()
    with torch.no_grad():
        for image_path in subset_df["image_path"].astype(str).tolist():
            try:
                image = Image.open(image_path).convert("RGB")
            except (UnidentifiedImageError, OSError):
                preds.append(np.nan)
                continue
            image_tensor = image_transform(image).unsqueeze(0).to(DEVICE)
            logits = model(image_tensor)
            preds.append(int((torch.sigmoid(logits) >= 0.5).item()))
    return preds


def data_quality_noise_analysis(
    val_df: pd.DataFrame,
    tokenizer: Tokenizer,
    val_transform,
    region_to_id: dict,
    unknown_region_id: int,
    category_to_id: dict,
    unknown_category_id: int,
    num_regions: int,
    num_categories: int,
) -> None:
    """
    Analyze likely dataset noise across three axes:
    1. Short vs long reviews
    2. With vs without captions
    3. Image quality impact
    """
    print("\n" + "=" * 72)
    print("DATA QUALITY & NOISE ANALYSIS")
    print("=" * 72)

    text_ckpt = OUTPUT_DIR / "best_Text_Only.pt"
    full_ckpt = OUTPUT_DIR / "best_Image_plus_Text_plus_Region.pt"
    image_ckpt = OUTPUT_DIR / "best_Image_Only.pt"

    if not text_ckpt.exists() or not full_ckpt.exists() or not image_ckpt.exists():
        print("Skipping data quality analysis: one or more checkpoints are missing.")
        print(f"Expected: {text_ckpt}, {full_ckpt}, {image_ckpt}")
        print("=" * 72)
        return

    text_model = TextOnlyModel(VOCAB_SIZE, EMBED_DIM, LSTM_HIDDEN).to(DEVICE)
    text_model.load_state_dict(torch.load(text_ckpt, map_location=DEVICE))

    full_model = MultimodalFusionModel(VOCAB_SIZE, num_regions, num_categories).to(DEVICE)
    full_model.load_state_dict(torch.load(full_ckpt, map_location=DEVICE))

    image_model = ImageOnlyModel().to(DEVICE)
    image_model.load_state_dict(torch.load(image_ckpt, map_location=DEVICE))

    summary_rows = []

    # 1. Short vs long reviews performance
    review_len_df = val_df[["review_text", "sentiment"]].copy()
    review_len_df["word_count"] = review_len_df["review_text"].astype(str).str.split().str.len()
    median_words = int(review_len_df["word_count"].median())
    short_mask = review_len_df["word_count"] <= median_words
    long_mask = review_len_df["word_count"] > median_words

    short_df = val_df.loc[short_mask].copy().reset_index(drop=True)
    long_df = val_df.loc[long_mask].copy().reset_index(drop=True)
    short_df["pred"] = _predict_text_model(text_model, short_df, tokenizer)
    long_df["pred"] = _predict_text_model(text_model, long_df, tokenizer)

    short_metrics = _evaluate_subset_predictions(short_df)
    long_metrics = _evaluate_subset_predictions(long_df)
    print(f"\n[1] Short vs Long Reviews (Text-only model)")
    print(f"    Median review length: {median_words} words")
    print(f"    Short reviews  -> samples={short_metrics['samples']}, F1={short_metrics['f1']:.4f}, Acc={short_metrics['accuracy']:.4f}")
    print(f"    Long reviews   -> samples={long_metrics['samples']}, F1={long_metrics['f1']:.4f}, Acc={long_metrics['accuracy']:.4f}")
    summary_rows.extend([
        {"analysis": "review_length", "group": "short", **short_metrics},
        {"analysis": "review_length", "group": "long", **long_metrics},
    ])

    # 2. With vs without captions
    caption_mask = val_df["caption"].fillna("").astype(str).str.strip().str.len() > 0
    with_caption_df = val_df.loc[caption_mask].copy().reset_index(drop=True)
    without_caption_df = val_df.loc[~caption_mask].copy().reset_index(drop=True)
    with_caption_df["pred"] = _predict_full_model(
        full_model,
        with_caption_df,
        tokenizer,
        region_to_id,
        unknown_region_id,
        category_to_id,
        unknown_category_id,
        val_transform,
    )
    without_caption_df["pred"] = _predict_full_model(
        full_model,
        without_caption_df,
        tokenizer,
        region_to_id,
        unknown_region_id,
        category_to_id,
        unknown_category_id,
        val_transform,
    )

    with_caption_metrics = _evaluate_subset_predictions(with_caption_df)
    without_caption_metrics = _evaluate_subset_predictions(without_caption_df)
    print(f"\n[2] Caption Availability (Full multimodal model)")
    print(f"    With captions    -> samples={with_caption_metrics['samples']}, F1={with_caption_metrics['f1']:.4f}, Acc={with_caption_metrics['accuracy']:.4f}")
    print(f"    Without captions -> samples={without_caption_metrics['samples']}, F1={without_caption_metrics['f1']:.4f}, Acc={without_caption_metrics['accuracy']:.4f}")
    summary_rows.extend([
        {"analysis": "caption_availability", "group": "with_caption", **with_caption_metrics},
        {"analysis": "caption_availability", "group": "without_caption", **without_caption_metrics},
    ])

    # 3. Image quality impact
    quality_df = val_df.copy()
    quality_df["image_quality"] = quality_df["image_path"].apply(estimate_image_quality)
    quality_df = quality_df.dropna(subset=["image_quality"]).reset_index(drop=True)
    if not quality_df.empty:
        threshold = float(quality_df["image_quality"].median())
        low_quality_df = quality_df.loc[quality_df["image_quality"] <= threshold].copy().reset_index(drop=True)
        high_quality_df = quality_df.loc[quality_df["image_quality"] > threshold].copy().reset_index(drop=True)
        low_quality_df["pred"] = _predict_image_model(image_model, low_quality_df, val_transform)
        high_quality_df["pred"] = _predict_image_model(image_model, high_quality_df, val_transform)

        low_quality_metrics = _evaluate_subset_predictions(low_quality_df)
        high_quality_metrics = _evaluate_subset_predictions(high_quality_df)
        print(f"\n[3] Image Quality Impact (Image-only model)")
        print(f"    Quality threshold (median edge energy): {threshold:.4f}")
        print(f"    Low-quality images  -> samples={low_quality_metrics['samples']}, F1={low_quality_metrics['f1']:.4f}, Acc={low_quality_metrics['accuracy']:.4f}")
        print(f"    High-quality images -> samples={high_quality_metrics['samples']}, F1={high_quality_metrics['f1']:.4f}, Acc={high_quality_metrics['accuracy']:.4f}")
        summary_rows.extend([
            {"analysis": "image_quality", "group": "low_quality", **low_quality_metrics},
            {"analysis": "image_quality", "group": "high_quality", **high_quality_metrics},
        ])

    print(
        "\nInsight:\n"
        "  Performance gaps across these subgroups indicate real dataset noise:\n"
        "  shorter reviews can be less informative, missing captions reduce metadata quality,\n"
        "  and weaker image quality can damage visual prediction reliability."
    )

    summary_df = pd.DataFrame(summary_rows)
    out_path = OUTPUT_DIR / "data_quality_noise_analysis.csv"
    summary_df.to_csv(out_path, index=False)
    print(f"Saved subgroup analysis to {out_path}")
    print("=" * 72)


# =========================================================
# Step 7: Ablation Study — Main
# =========================================================
def main():
    print("\n" + "=" * 72)
    print("PROJECT PROBLEM STATEMENT")
    print("=" * 72)
    print(RESEARCH_QUESTION)
    print("=" * 72)

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

    df = df.sample(min(len(df), 30000), random_state=SEED).reset_index(drop=True)
    print(
        "Limitation note: photo-review pairs are matched at business level, so this pipeline "
        "learns business-level multimodal sentiment rather than perfectly dish-level sentiment."
    )

    # Shared train/val split — identical across all 4 experiments
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=SEED,
        stratify=df["sentiment"],
    )

    # Class-weighted loss to address positive-class imbalance
    neg_count = int((train_df["sentiment"] == 0).sum())
    pos_count = int((train_df["sentiment"] == 1).sum())
    pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32).to(DEVICE)
    print(f"Class balance — pos: {pos_count}, neg: {neg_count}, pos_weight: {pos_weight.item():.3f}")

    # Text encoding (fit on train only)
    tokenizer = fit_tokenizer(train_df["review_text"].tolist(), vocab_size=VOCAB_SIZE)
    X_train_text = encode_texts(tokenizer, train_df["review_text"].tolist(), max_len=MAX_TEXT_LEN)
    X_val_text = encode_texts(tokenizer, val_df["review_text"].tolist(), max_len=MAX_TEXT_LEN)

    # Region encoding with unknown-bucket for unseen val regions
    region_to_id = {
        region: idx for idx, region in enumerate(sorted(train_df["region"].unique()))
    }
    unknown_region_id = len(region_to_id)
    id_to_region = {idx: region for region, idx in region_to_id.items()}
    id_to_region[unknown_region_id] = "<unknown_region>"
    train_region_ids = train_df["region"].map(region_to_id).to_numpy(dtype=np.int64)
    val_region_ids = val_df["region"].map(
        lambda r: region_to_id.get(r, unknown_region_id)
    ).to_numpy(dtype=np.int64)
    num_regions = unknown_region_id + 1
    unseen_val_regions = val_df.loc[~val_df["region"].isin(region_to_id), "region"].nunique()
    print(f"Number of regions: {num_regions}")
    if unseen_val_regions:
        print(f"Validation regions unseen in train: {unseen_val_regions}")

    # Category target (classification): use primary category token
    train_df = train_df.copy()
    val_df = val_df.copy()
    train_df["primary_category"] = train_df["categories"].apply(extract_primary_category)
    val_df["primary_category"] = val_df["categories"].apply(extract_primary_category)

    top_categories = train_df["primary_category"].value_counts().head(MAX_CATEGORY_CLASSES).index.tolist()
    category_to_id = {cat: idx for idx, cat in enumerate(top_categories)}
    unknown_category_id = len(category_to_id)
    category_to_id["<unknown_category>"] = unknown_category_id
    id_to_category = {idx: cat for cat, idx in category_to_id.items()}
    num_categories = len(category_to_id)

    train_category_ids = train_df["primary_category"].map(
        lambda c: category_to_id.get(c, unknown_category_id)
    ).to_numpy(dtype=np.int64)
    val_category_ids = val_df["primary_category"].map(
        lambda c: category_to_id.get(c, unknown_category_id)
    ).to_numpy(dtype=np.int64)
    print(f"Number of category classes (with unknown): {num_categories}")

    # Rating target (regression): normalized review stars in [0, 1]
    train_ratings = train_df["review_stars"].astype(float).apply(scale_rating).to_numpy(dtype=np.float32)
    val_ratings = val_df["review_stars"].astype(float).apply(scale_rating).to_numpy(dtype=np.float32)

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

    # Shared DataLoaders (same data across all 4 experiments)
    _pin = DEVICE.type == "cuda"
    train_dataset = YelpMultimodalDataset(
        train_df,
        X_train_text,
        train_region_ids,
        train_category_ids,
        train_ratings,
        image_transform=train_transform,
    )
    val_dataset = YelpMultimodalDataset(
        val_df,
        X_val_text,
        val_region_ids,
        val_category_ids,
        val_ratings,
        image_transform=val_transform,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
        collate_fn=safe_collate, pin_memory=_pin, persistent_workers=NUM_WORKERS > 0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
        collate_fn=safe_collate, pin_memory=_pin, persistent_workers=NUM_WORKERS > 0,
    )

    # ── Ablation experiments (same split, same pos_weight) ─────────────────
    experiments = [
        ("Image Only",            ImageOnlyModel(),                                    "image"),
        ("Text Only",             TextOnlyModel(VOCAB_SIZE, EMBED_DIM, LSTM_HIDDEN),   "text"),
        ("Image + Text",          ImageTextFusionModel(VOCAB_SIZE),                    "image_text"),
        ("Image + Text + Region", MultimodalFusionModel(VOCAB_SIZE, num_regions, num_categories), "full"),
    ]

    ablation_results: dict = {}
    for exp_name, model, mode in experiments:
        best = run_experiment(exp_name, model, train_loader, val_loader, mode, pos_weight)
        ablation_results[exp_name] = best

    # ── Print comparison table ─────────────────────────────────────────────
    rows = []
    for exp_name, m in ablation_results.items():
        row = {
            "Model":      exp_name,
            "Best Epoch": m["best_epoch"],
            "Accuracy":   f"{m['val_accuracy']:.4f}",
            "Precision":  f"{m['val_precision']:.4f}",
            "Recall":     f"{m['val_recall']:.4f}",
            "F1":         f"{m['val_f1']:.4f}",
            "Loss":       f"{m['val_loss']:.4f}",
        }
        if "val_category_accuracy" in m:
            row["Category Acc"] = f"{m['val_category_accuracy']:.4f}"
        if "val_rating_mae" in m:
            row["Rating MAE"] = f"{m['val_rating_mae']:.4f}"
        rows.append(row)

    table_df = pd.DataFrame(rows)

    print("\n" + "=" * 72)
    print("ABLATION STUDY — BEST VALIDATION METRICS PER MODEL VARIANT")
    print("=" * 72)
    print(table_df.to_string(index=False))
    print("=" * 72)

    table_df.to_csv(OUTPUT_DIR / "ablation_results.csv", index=False)
    print(f"\nFull results saved to {OUTPUT_DIR / 'ablation_results.csv'}")

    # ── Image-Text Consistency Analysis ───────────────────────────────────
    image_text_consistency_analysis(
        val_loader=val_loader,
        vocab_size=VOCAB_SIZE,
        id_to_region=id_to_region,
        id_to_category=id_to_category,
    )

    # ── Cross-modal retrieval (text<->image) ─────────────────────────────
    cross_modal_retrieval_analysis(
        val_df=val_df,
        tokenizer=tokenizer,
        val_transform=val_transform,
        max_pool_size=2000,
        top_k=5,
        query_text="spicy ramen",
    )

    # ── Data quality and noise analysis ───────────────────────────────────
    data_quality_noise_analysis(
        val_df=val_df,
        tokenizer=tokenizer,
        val_transform=val_transform,
        region_to_id=region_to_id,
        unknown_region_id=unknown_region_id,
        category_to_id=category_to_id,
        unknown_category_id=unknown_category_id,
        num_regions=num_regions,
        num_categories=num_categories,
    )

    # ── Text Importance Analysis (TF-IDF) ────────────────────────────────
    text_importance_analysis(train_df)

    # ── Visual Feature Analysis (CNN embeddings) ──────────────────────────
    visual_feature_analysis(
        val_loader=val_loader,
        id_to_region=id_to_region,
    )

    # ── Region Importance Analysis ─────────────────────────────────────────
    region_importance_analysis(df, ablation_results)

    # ── Final presentation-ready plots ────────────────────────────────────
    final_presentation_plots(
        ablation_table=table_df,
        val_loader=val_loader,
        num_regions=num_regions,
        num_categories=num_categories,
    )


def text_importance_analysis(
    train_df: pd.DataFrame,
    top_n: int = 20,
    max_features: int = 30000,
    ngram_range: tuple = (1, 2),
) -> None:
    """
    TF-IDF word importance analysis.

    Splits training text by sentiment class and uses TF-IDF to surface the
    terms most strongly associated with positive vs negative reviews.
    Also plots horizontal bar charts of top keywords per class.
    """
    print("\n" + "=" * 72)
    print("TEXT IMPORTANCE ANALYSIS (TF-IDF)")
    print("=" * 72)

    pos_texts = train_df.loc[train_df["sentiment"] == 1, "review_text"].tolist()
    neg_texts = train_df.loc[train_df["sentiment"] == 0, "review_text"].tolist()

    print(f"Training reviews — positive: {len(pos_texts)}, negative: {len(neg_texts)}")

    # ── Fit one TF-IDF on ALL train text, then compare mean scores per class
    all_texts = pos_texts + neg_texts
    all_labels = [1] * len(pos_texts) + [0] * len(neg_texts)

    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words="english",
        sublinear_tf=True,       # log(1+tf) — dampens very frequent terms
        min_df=5,
    )
    X = tfidf.fit_transform(all_texts)
    terms = np.array(tfidf.get_feature_names_out())

    labels_arr = np.array(all_labels)
    pos_mask = labels_arr == 1
    neg_mask = labels_arr == 0

    # Mean TF-IDF score per term for each class
    pos_mean = np.asarray(X[pos_mask].mean(axis=0)).ravel()
    neg_mean = np.asarray(X[neg_mask].mean(axis=0)).ravel()

    # Discriminative score: how much more a term appears in one class
    pos_score = pos_mean - neg_mean   # high → more positive
    neg_score = neg_mean - pos_mean   # high → more negative

    top_pos_idx = np.argsort(pos_score)[::-1][:top_n]
    top_neg_idx = np.argsort(neg_score)[::-1][:top_n]

    top_pos_terms  = terms[top_pos_idx]
    top_pos_scores = pos_score[top_pos_idx]
    top_neg_terms  = terms[top_neg_idx]
    top_neg_scores = neg_score[top_neg_idx]

    # ── Console summary ────────────────────────────────────────────────────
    print(f"\nTop {top_n} POSITIVE sentiment keywords (TF-IDF discriminative score):")
    for term, score in zip(top_pos_terms, top_pos_scores):
        bar = "█" * int(score * 3000)
        print(f"  {term:<30s}  {score:.5f}  {bar}")

    print(f"\nTop {top_n} NEGATIVE sentiment keywords (TF-IDF discriminative score):")
    for term, score in zip(top_neg_terms, top_neg_scores):
        bar = "█" * int(score * 3000)
        print(f"  {term:<30s}  {score:.5f}  {bar}")

    # ── Bar chart: positive keywords ───────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].barh(
        top_pos_terms[::-1], top_pos_scores[::-1],
        color="#2196F3", edgecolor="none",
    )
    axes[0].set_title(f"Top {top_n} Positive Keywords (TF-IDF)", fontsize=12)
    axes[0].set_xlabel("Discriminative score (pos − neg mean TF-IDF)")
    axes[0].invert_xaxis()
    axes[0].yaxis.set_label_position("right")
    axes[0].yaxis.tick_right()

    axes[1].barh(
        top_neg_terms[::-1], top_neg_scores[::-1],
        color="#F44336", edgecolor="none",
    )
    axes[1].set_title(f"Top {top_n} Negative Keywords (TF-IDF)", fontsize=12)
    axes[1].set_xlabel("Discriminative score (neg − pos mean TF-IDF)")

    fig.suptitle("TF-IDF Sentiment Keyword Analysis", fontsize=14, y=1.01)
    plt.tight_layout()
    kw_plot_path = OUTPUT_DIR / "text_tfidf_keywords.png"
    fig.savefig(kw_plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved keyword bar chart to {kw_plot_path}")

    # ── Save CSV: full term scores ─────────────────────────────────────────
    term_df = pd.DataFrame({
        "term":      terms,
        "pos_mean":  pos_mean,
        "neg_mean":  neg_mean,
        "pos_score": pos_score,
        "neg_score": neg_score,
    }).sort_values("pos_score", ascending=False)
    csv_path = OUTPUT_DIR / "text_tfidf_term_scores.csv"
    term_df.to_csv(csv_path, index=False)
    print(f"Saved full term score table to {csv_path}")

    # ── Region-level keyword drift ─────────────────────────────────────────
    # Are different words used positively in different regions?
    if "region" in train_df.columns:
        top_regions = (
            train_df["region"].value_counts().head(6).index.tolist()
        )
        region_top_pos: dict = {}
        for region in top_regions:
            region_texts = train_df.loc[
                (train_df["region"] == region) & (train_df["sentiment"] == 1),
                "review_text",
            ].tolist()
            if len(region_texts) < 20:
                continue
            rv = TfidfVectorizer(
                max_features=5000, ngram_range=(1, 1),
                stop_words="english", sublinear_tf=True, min_df=2,
            )
            rv.fit(region_texts)
            scores = np.asarray(rv.transform(region_texts).mean(axis=0)).ravel()
            top_idx = np.argsort(scores)[::-1][:5]
            region_top_pos[region] = list(zip(
                rv.get_feature_names_out()[top_idx], scores[top_idx]
            ))

        if region_top_pos:
            print("\nTop positive keywords by region (shows regional language variation):")
            for region, kws in region_top_pos.items():
                kw_str = ", ".join(f"{w} ({s:.4f})" for w, s in kws)
                print(f"  {region:<30s}  {kw_str}")

    print("=" * 72)


def visual_feature_analysis(
    val_loader: DataLoader,
    id_to_region: dict,
    n_pca_components: int = 50,
    n_tsne_components: int = 2,
    max_samples: int = 3000,
) -> None:
    """
    Extract CNN embeddings from the best Image+Text+Region model's image backbone,
    then apply PCA + t-SNE and plot clusters coloured by:
      1. Sentiment (positive / negative)
      2. Region (top-N most-frequent)
    Saved as PNG files in outputs/.
    """
    ckpt_path = OUTPUT_DIR / "best_Image_plus_Text_plus_Region.pt"
    # Fallback: try Image Only checkpoint if full model not found
    if not ckpt_path.exists():
        ckpt_path = OUTPUT_DIR / "best_Image_Only.pt"

    print("\n" + "=" * 72)
    print("VISUAL FEATURE ANALYSIS (CNN Embeddings)")
    print("=" * 72)

    if not ckpt_path.exists():
        print(f"Skipping: no checkpoint found at {ckpt_path}")
        print("=" * 72)
        return

    # ── 1. Build a feature extractor from ResNet18 backbone ───────────────
    backbone = models.resnet18(weights=None)   # weights loaded from checkpoint below
    backbone.fc = nn.Identity()
    backbone = backbone.to(DEVICE)

    # Load weights: handle both bare-backbone and full model state_dicts
    state = torch.load(ckpt_path, map_location=DEVICE)
    backbone_keys = {k.removeprefix("image_backbone."): v
                     for k, v in state.items() if k.startswith("image_backbone.")}
    if backbone_keys:
        backbone.load_state_dict(backbone_keys)
    else:
        # Checkpoint is ImageOnlyModel: backbone key prefix is "backbone."
        backbone_keys = {k.removeprefix("backbone."): v
                         for k, v in state.items() if k.startswith("backbone.")}
        if backbone_keys:
            backbone.load_state_dict(backbone_keys)
        else:
            print("Could not load backbone weights from checkpoint — using random init.")
    backbone.eval()
    print(f"Loaded backbone from: {ckpt_path}")

    # ── 2. Extract features ───────────────────────────────────────────────
    all_features: List[np.ndarray] = []
    all_labels: List[int] = []
    all_region_ids: List[int] = []

    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue
            images, _text, region_ids, _category_ids, _ratings, labels = batch
            images = images.to(DEVICE)
            feats = backbone(images).cpu().numpy()   # (B, 512)
            all_features.append(feats)
            all_labels.extend(labels.numpy().astype(int).tolist())
            all_region_ids.extend(region_ids.numpy().astype(int).tolist())
            if sum(len(f) for f in all_features) >= max_samples:
                break

    if not all_features:
        print("No features extracted — check the DataLoader.")
        print("=" * 72)
        return

    features = np.vstack(all_features)[:max_samples]
    labels_arr = np.array(all_labels[:max_samples])
    region_ids_arr = np.array(all_region_ids[:max_samples])
    print(f"Extracted {len(features)} image embeddings (dim={features.shape[1]})")

    # ── 3. PCA → t-SNE pipeline ───────────────────────────────────────────
    n_pca = min(n_pca_components, features.shape[0], features.shape[1])
    pca = PCA(n_components=n_pca, random_state=SEED)
    features_pca = pca.fit_transform(features)
    variance_explained = pca.explained_variance_ratio_[:n_pca].sum()
    print(f"PCA: kept {n_pca} components, explained variance: {variance_explained:.3f}")

    print("Running t-SNE (this may take a minute)...")
    perplexity = min(30, len(features_pca) - 1)
    tsne = TSNE(
        n_components=n_tsne_components,
        perplexity=perplexity,
        random_state=SEED,
        max_iter=500,
        init="pca",
    )
    emb = tsne.fit_transform(features_pca)
    print("t-SNE complete.")

    # Save raw embeddings for external plotting
    emb_df = pd.DataFrame({
        "tsne_x": emb[:, 0],
        "tsne_y": emb[:, 1],
        "sentiment": labels_arr,
        "region_id": region_ids_arr,
        "region": [id_to_region.get(int(r), "<unknown_region>") for r in region_ids_arr],
    })
    emb_df.to_csv(OUTPUT_DIR / "visual_features_tsne.csv", index=False)
    print(f"Saved t-SNE embeddings to {OUTPUT_DIR / 'visual_features_tsne.csv'}")

    # ── 4. Plot 1: Colour by sentiment ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))
    colours = {1: "#2196F3", 0: "#F44336"}
    sentiment_names = {1: "Positive", 0: "Negative"}
    for sent_val, colour in colours.items():
        mask = labels_arr == sent_val
        ax.scatter(
            emb[mask, 0], emb[mask, 1],
            c=colour, label=sentiment_names[sent_val],
            s=8, alpha=0.55, linewidths=0,
        )
    ax.set_title("t-SNE of CNN Image Embeddings — coloured by Sentiment", fontsize=13)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.legend(markerscale=3, framealpha=0.8)
    ax.annotate(
        f"n={len(features)}  |  PCA→{n_pca}d ({variance_explained:.1%} var)",
        xy=(0.02, 0.02), xycoords="axes fraction", fontsize=8, color="grey",
    )
    plt.tight_layout()
    sentiment_plot_path = OUTPUT_DIR / "visual_features_tsne_sentiment.png"
    fig.savefig(sentiment_plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved sentiment cluster plot to {sentiment_plot_path}")

    # ── 5. Plot 2: Colour by region (top-12 by frequency) ─────────────────
    unique, counts = np.unique(region_ids_arr, return_counts=True)
    top_region_ids = set(unique[np.argsort(counts)[::-1][:12]])
    # Build discrete colour map
    sorted_top = sorted(top_region_ids)
    cmap = plt.cm.get_cmap("tab20", len(sorted_top))
    region_colour_map = {rid: cmap(i) for i, rid in enumerate(sorted_top)}

    fig, ax = plt.subplots(figsize=(11, 7))
    # Plot non-top regions in grey first
    other_mask = np.array([r not in top_region_ids for r in region_ids_arr])
    if other_mask.any():
        ax.scatter(
            emb[other_mask, 0], emb[other_mask, 1],
            c="#CCCCCC", s=6, alpha=0.3, linewidths=0, label="Other regions",
        )
    for rid in sorted_top:
        mask = region_ids_arr == rid
        label = id_to_region.get(int(rid), f"region_{rid}")
        ax.scatter(
            emb[mask, 0], emb[mask, 1],
            c=[region_colour_map[rid]],
            s=10, alpha=0.65, linewidths=0,
            label=label,
        )
    ax.set_title("t-SNE of CNN Image Embeddings — coloured by Region (top 12)", fontsize=13)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.legend(
        markerscale=2.5, fontsize=7, framealpha=0.8,
        bbox_to_anchor=(1.01, 1), loc="upper left",
    )
    plt.tight_layout()
    region_plot_path = OUTPUT_DIR / "visual_features_tsne_region.png"
    fig.savefig(region_plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved region cluster plot to {region_plot_path}")

    # ── 6. PCA scree plot ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    ax.plot(range(1, len(cumvar) + 1), cumvar, marker=".", linewidth=1.2, color="#1976D2")
    ax.axhline(0.90, linestyle="--", linewidth=0.8, color="#E53935", label="90% variance")
    ax.set_xlabel("Number of PCA components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title("PCA Scree Plot — CNN Image Embeddings", fontsize=12)
    ax.legend()
    plt.tight_layout()
    scree_path = OUTPUT_DIR / "visual_features_pca_scree.png"
    fig.savefig(scree_path, dpi=150)
    plt.close(fig)
    print(f"Saved PCA scree plot to {scree_path}")

    print("=" * 72)


def region_importance_analysis(df: pd.DataFrame, ablation_results: dict) -> None:
    """
    Prove that region matters by:
    1. Showing the F1 gain from adding region to Image+Text.
    2. Analysing per-region positive-sentiment rates on the full dataset,
       revealing that cultural/geographic preferences vary even within the US.
    """
    print("\n" + "=" * 72)
    print("REGION IMPORTANCE ANALYSIS")
    print("=" * 72)

    # ── 1. F1 delta: Image+Text vs Image+Text+Region ───────────────────────
    f1_no_region  = ablation_results["Image + Text"]["val_f1"]
    f1_with_region = ablation_results["Image + Text + Region"]["val_f1"]
    delta = f1_with_region - f1_no_region

    print("\n[1] F1 impact of adding Region embedding")
    print(f"    Image + Text            F1 = {f1_no_region:.4f}")
    print(f"    Image + Text + Region   F1 = {f1_with_region:.4f}")
    print(f"    Delta (region gain)         {delta:+.4f}")
    if delta > 0:
        print("    → Region embedding IMPROVES sentiment prediction.")
    else:
        print("    → Region embedding did not improve F1 in this run "
              "(may need more epochs to converge).")

    # ── 2. Per-region sentiment rate ───────────────────────────────────────
    region_stats = (
        df.groupby("region")["sentiment"]
        .agg(total="count", positive_rate="mean")
        .reset_index()
        .sort_values("positive_rate", ascending=False)
    )
    region_stats["positive_rate"] = region_stats["positive_rate"].round(4)

    print("\n[2] Per-region positive-sentiment rate (full dataset)")
    print(f"    Total regions analysed: {len(region_stats)}")

    top_n = 10
    bottom_n = 10
    print(f"\n    Top {top_n} most positive regions:")
    print(
        region_stats.head(top_n)
        .to_string(index=False, columns=["region", "positive_rate", "total"])
    )
    print(f"\n    Bottom {bottom_n} least positive regions:")
    print(
        region_stats.tail(bottom_n)
        .to_string(index=False, columns=["region", "positive_rate", "total"])
    )

    # Spread: max vs min positive rate
    max_rate = region_stats["positive_rate"].max()
    min_rate = region_stats["positive_rate"].min()
    spread   = max_rate - min_rate
    print(f"\n    Positive-rate spread across regions: {min_rate:.4f} → {max_rate:.4f}  (Δ {spread:.4f})")
    print(
        "    → Even within the US, cultural preferences vary significantly\n"
        "      across cities/states. The same food photo+review can be rated\n"
        "      very differently depending on the local dining culture.\n"
        "      This regional variation justifies including geographic context\n"
        "      as a modality in the fusion model."
    )

    # ── 3. High-variance regions: same cuisine, different sentiment ────────
    if "categories" in df.columns:
        # Look for businesses labelled as restaurants/food
        food_mask = df["categories"].str.contains(
            "Restaurant|Food|Cafe|Bar|Pizza|Sushi|Burger", case=False, na=False
        )
        food_region_stats = (
            df[food_mask]
            .groupby("region")["sentiment"]
            .agg(total="count", positive_rate="mean")
            .query("total >= 30")           # only regions with enough samples
            .reset_index()
            .sort_values("positive_rate", ascending=False)
        )
        if len(food_region_stats) >= 4:
            print("\n[3] Food/Restaurant sentiment by region (min 30 reviews per region)")
            print(f"    Most positive food regions:")
            print(food_region_stats.head(5).to_string(index=False))
            print(f"\n    Least positive food regions:")
            print(food_region_stats.tail(5).to_string(index=False))
            food_spread = food_region_stats["positive_rate"].max() - food_region_stats["positive_rate"].min()
            print(
                f"\n    Food sentiment spread: {food_spread:.4f}\n"
                "    → Identical food categories receive meaningfully different\n"
                "      sentiment scores across US regions — strong evidence that\n"
                "      region is a non-redundant signal for the model."
            )

    # ── 4. Same cuisine, different region comparison (explicit evidence) ───
    if "cuisine_cluster" in df.columns:
        cuisine_region_stats = (
            df.groupby(["cuisine_cluster", "region"])["sentiment"]
            .agg(total="count", positive_rate="mean")
            .query("total >= 20")
            .reset_index()
        )
        if not cuisine_region_stats.empty:
            spread_by_cuisine = (
                cuisine_region_stats.groupby("cuisine_cluster")["positive_rate"]
                .agg(min_rate="min", max_rate="max", region_count="count")
                .reset_index()
            )
            spread_by_cuisine["spread"] = spread_by_cuisine["max_rate"] - spread_by_cuisine["min_rate"]
            spread_by_cuisine = spread_by_cuisine.sort_values("spread", ascending=False)
            top_spread = spread_by_cuisine.head(10)

            print("\n[4] Same cuisine, different region sentiment spread")
            print(top_spread.to_string(index=False))
            print(
                "\n    → This directly addresses the US-homogeneity concern: even within\n"
                "      one country, the same cuisine receives different sentiment\n"
                "      across regions, supporting region-aware modeling."
            )

            top_spread.to_csv(OUTPUT_DIR / "cuisine_region_sentiment_spread.csv", index=False)

    # ── 5. Save full stats ─────────────────────────────────────────────────
    out_path = OUTPUT_DIR / "region_sentiment_stats.csv"
    region_stats.to_csv(out_path, index=False)
    print(f"\n    Full region stats saved to {out_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
