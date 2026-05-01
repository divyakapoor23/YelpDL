from pathlib import Path
import io
import os
import pickle
import re
import sys
import shutil
import tarfile
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image, ImageStat


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CITY_QUICK_LOAD_EXAMPLES_PATH = OUTPUT_DIR / "city_quick_load_examples.csv"

DEMO_SCENARIOS = {
	"Custom input": {
		"review": "",
		"region": "Nashville_TN",
		"cuisine": "American (New)",
		"insight": "Bring your own food image and review text to simulate a real customer-feedback workflow.",
	},
	"Perception gap demo - Nashville": {
		"review": "The food looked amazing, but it tasted bland and service was slow.",
		"region": "Nashville_TN",
		"cuisine": "American (New)",
		"insight": "Classic mismatch case: strong visual appeal, but text describes a disappointing experience.",
	},
	"Positive hero dish - Philadelphia": {
		"review": "The burger was juicy, perfectly cooked, and absolutely worth the hype. I would order it again tonight.",
		"region": "Philadelphia_PA",
		"cuisine": "American (Traditional)",
		"insight": "High-confidence positive case where both wording and likely visual presentation align.",
	},
	"Region-sensitive seafood - Tampa": {
		"review": "Fresh seafood, great plating, but the portion felt small for the price in this area.",
		"region": "Tampa_FL",
		"cuisine": "Seafood",
		"insight": "Useful for showing how cuisine and location can shift customer expectations even when the dish looks strong.",
	},
}

POSITIVE_TERMS = {
	"amazing", "beautiful", "best", "crispy", "delicious", "excellent", "fantastic", "fresh",
	"friendly", "good", "great", "juicy", "love", "perfect", "tasty", "wonderful", "worth",
}

NEGATIVE_TERMS = {
	"awful", "bad", "bland", "burnt", "cold", "confusing", "disappointing", "dry", "greasy",
	"late", "mediocre", "negative", "overpriced", "salty", "slow", "soggy", "terrible", "underseasoned", "worst",
}

APP_CSS = """
<style>
div.block-container {
	padding-top: 1.6rem;
	padding-bottom: 3rem;
	max-width: 1320px;
}

.stApp {
	background:
		radial-gradient(circle at top right, rgba(205, 120, 63, 0.16), transparent 26%),
		radial-gradient(circle at top left, rgba(47, 91, 70, 0.12), transparent 22%),
		linear-gradient(180deg, #fffaf5 0%, #f7efe5 48%, #f3eee9 100%);
}

div[data-testid="stMetric"] {
	background: rgba(255, 252, 247, 0.82);
	border: 1px solid rgba(130, 98, 70, 0.16);
	border-radius: 20px;
	padding: 1rem 1.1rem;
	box-shadow: 0 16px 40px rgba(76, 55, 35, 0.08);
}

div[data-testid="stMetricLabel"] {
	font-size: 0.92rem;
	letter-spacing: 0.02em;
}

.stButton > button, div[data-testid="stFormSubmitButton"] button {
	background: linear-gradient(135deg, #a34a1a 0%, #d06f3b 100%);
	color: white;
	border: none;
	border-radius: 999px;
	padding: 0.7rem 1.2rem;
	font-weight: 600;
	box-shadow: 0 12px 30px rgba(163, 74, 26, 0.22);
}

.stButton > button:hover, div[data-testid="stFormSubmitButton"] button:hover {
	background: linear-gradient(135deg, #8b3810 0%, #be5d27 100%);
}

div[data-baseweb="select"] > div,
div[data-baseweb="base-input"] > div,
textarea {
	border-radius: 16px !important;
	background: rgba(255, 253, 250, 0.88) !important;
	border-color: rgba(130, 98, 70, 0.18) !important;
}

.yfi-hero {
	background: linear-gradient(135deg, rgba(46, 88, 69, 0.96) 0%, rgba(148, 67, 27, 0.96) 100%);
	border-radius: 28px;
	padding: 2rem 2.2rem;
	color: #fff8f2;
	box-shadow: 0 24px 60px rgba(53, 34, 20, 0.18);
	margin-bottom: 1rem;
}

.yfi-hero h1 {
	margin: 0;
	font-size: 2.35rem;
	line-height: 1.05;
}

.yfi-hero p {
	margin: 0.85rem 0 0 0;
	font-size: 1rem;
	max-width: 760px;
	line-height: 1.55;
}

.yfi-badge-row {
	display: flex;
	gap: 0.6rem;
	flex-wrap: wrap;
	margin-top: 1rem;
}

.yfi-badge {
	padding: 0.4rem 0.8rem;
	border-radius: 999px;
	background: rgba(255, 248, 242, 0.14);
	border: 1px solid rgba(255, 248, 242, 0.24);
	font-size: 0.9rem;
}

.yfi-panel {
	background: rgba(255, 252, 247, 0.82);
	border: 1px solid rgba(130, 98, 70, 0.14);
	border-radius: 24px;
	padding: 1.2rem 1.25rem;
	box-shadow: 0 16px 40px rgba(76, 55, 35, 0.06);
	height: 100%;
}

.yfi-panel h3 {
	margin-top: 0;
	margin-bottom: 0.45rem;
	font-size: 1.05rem;
}

.yfi-panel p {
	margin-bottom: 0;
	line-height: 1.5;
	font-size: 0.95rem;
}

.yfi-section-label {
	text-transform: uppercase;
	letter-spacing: 0.08em;
	font-size: 0.8rem;
	color: #91552b;
	font-weight: 700;
	margin-bottom: 0.35rem;
}
</style>
"""


def _resolve_env_path(name: str, default: Path) -> Path:
	raw_value = os.getenv(name)
	if raw_value:
		return Path(raw_value).expanduser()
	try:
		secret_value = st.secrets.get(name)
	except Exception:
		secret_value = None
	if secret_value:
		return Path(str(secret_value)).expanduser()
	return default


DATA_PATH_ENV_VARS = [
	"YELP_DATA_ROOT",
	"YELP_JSON_DATA_DIR",
	"YELP_PHOTO_DATA_DIR",
	"YELP_BUSINESS_PATH",
	"YELP_REVIEW_PATH",
	"YELP_PHOTO_META_PATH",
	"YELP_PHOTO_IMAGES_DIR",
]

DATA_URL_ENV_VARS = [
	"YELP_BUSINESS_URL",
	"YELP_REVIEW_URL",
	"YELP_PHOTO_META_URL",
	"YELP_PHOTO_IMAGES_ARCHIVE_URL",
	"YELP_RUNTIME_DATA_ROOT",
]


def _resolve_setting(name: str) -> str | None:
	raw_value = os.getenv(name)
	if raw_value:
		return raw_value
	try:
		secret_value = st.secrets.get(name)
	except Exception:
		secret_value = None
	if secret_value:
		return str(secret_value)
	return None


def _set_pipeline_paths(data_root_override: Path | None = None) -> None:
	global DATA_ROOT, JSON_DATA_DIR, PHOTO_DATA_DIR
	global BUSINESS_PATH, REVIEW_PATH, PHOTO_META_PATH, PHOTO_IMAGES_DIR
	global REQUIRED_PIPELINE_INPUTS

	if data_root_override is not None:
		DATA_ROOT = data_root_override
	else:
		DATA_ROOT = _resolve_env_path("YELP_DATA_ROOT", PROJECT_ROOT / "Data")

	JSON_DATA_DIR = _resolve_env_path("YELP_JSON_DATA_DIR", DATA_ROOT / "Yelp JSON" / "yelp_dataset")
	PHOTO_DATA_DIR = _resolve_env_path("YELP_PHOTO_DATA_DIR", DATA_ROOT / "Yelp Photos" / "yelp_photos")
	BUSINESS_PATH = _resolve_env_path(
		"YELP_BUSINESS_PATH",
		JSON_DATA_DIR / "yelp_academic_dataset_business.json",
	)
	REVIEW_PATH = _resolve_env_path(
		"YELP_REVIEW_PATH",
		JSON_DATA_DIR / "yelp_academic_dataset_review.json",
	)
	PHOTO_META_PATH = _resolve_env_path("YELP_PHOTO_META_PATH", PHOTO_DATA_DIR / "photos.json")
	PHOTO_IMAGES_DIR = _resolve_env_path("YELP_PHOTO_IMAGES_DIR", PHOTO_DATA_DIR / "photos")

	REQUIRED_PIPELINE_INPUTS = [
		BUSINESS_PATH,
		REVIEW_PATH,
		PHOTO_META_PATH,
		PHOTO_IMAGES_DIR,
	]


def _download_file(url: str, destination: Path) -> None:
	destination.parent.mkdir(parents=True, exist_ok=True)
	with urllib.request.urlopen(url) as response, destination.open("wb") as out_file:
		shutil.copyfileobj(response, out_file)


def prepare_pipeline_inputs_from_urls() -> tuple[list[str], list[str]]:
	logs: list[str] = []
	errors: list[str] = []

	runtime_root = Path(
		_resolve_setting("YELP_RUNTIME_DATA_ROOT")
		or os.getenv("YELP_RUNTIME_DATA_ROOT")
		or "/tmp/yelp_data"
	).expanduser()
	json_dir = runtime_root / "Yelp JSON" / "yelp_dataset"
	photo_dir = runtime_root / "Yelp Photos" / "yelp_photos"
	photos_dir = photo_dir / "photos"

	url_targets = [
		("YELP_BUSINESS_URL", json_dir / "yelp_academic_dataset_business.json"),
		("YELP_REVIEW_URL", json_dir / "yelp_academic_dataset_review.json"),
		("YELP_PHOTO_META_URL", photo_dir / "photos.json"),
	]

	for key, target in url_targets:
		if target.exists():
			logs.append(f"Found existing file: {target}")
			continue
		url = _resolve_setting(key)
		if not url:
			errors.append(f"Missing configuration: {key}")
			continue
		try:
			_download_file(url, target)
			logs.append(f"Downloaded {key} -> {target}")
		except Exception as exc:
			errors.append(f"Failed downloading {key}: {exc}")

	archive_url = _resolve_setting("YELP_PHOTO_IMAGES_ARCHIVE_URL")
	if photos_dir.exists() and any(photos_dir.glob("*.jpg")):
		logs.append(f"Found existing photos directory: {photos_dir}")
	elif archive_url:
		archive_name = Path(archive_url).name or "photos_archive"
		archive_path = runtime_root / archive_name
		try:
			_download_file(archive_url, archive_path)
			photos_dir.mkdir(parents=True, exist_ok=True)
			if archive_path.suffix.lower() == ".zip":
				with zipfile.ZipFile(archive_path, "r") as zf:
					zf.extractall(photo_dir)
			elif archive_path.suffix.lower() in {".gz", ".tgz", ".tar"}:
				with tarfile.open(archive_path, "r:*") as tf:
					tf.extractall(photo_dir)
			else:
				errors.append(
					"Unsupported archive format for YELP_PHOTO_IMAGES_ARCHIVE_URL. "
					"Use .zip, .tar, .tar.gz, or .tgz."
				)
			logs.append(f"Downloaded and extracted image archive -> {photo_dir}")
		except Exception as exc:
			errors.append(f"Failed preparing photo archive: {exc}")
	else:
		errors.append("Missing configuration: YELP_PHOTO_IMAGES_ARCHIVE_URL")

	# Only redirect paths to the runtime root if the required files are
	# actually present there (i.e. downloads succeeded).  Overriding the
	# env var when downloads failed would mask the local Data/ directory.
	runtime_inputs_ready = all([
		(json_dir / "yelp_academic_dataset_business.json").exists(),
		(json_dir / "yelp_academic_dataset_review.json").exists(),
		(photo_dir / "photos.json").exists(),
		photos_dir.exists(),
	])
	if runtime_inputs_ready:
		os.environ["YELP_DATA_ROOT"] = str(runtime_root)
		_set_pipeline_paths(data_root_override=runtime_root)
	else:
		# Leave the path config pointing at whatever was resolved at startup
		# (usually the local Data/ directory).
		logs.append(
			"Runtime data root not fully populated — keeping existing path config."
		)

	return logs, errors

_set_pipeline_paths()

RESEARCH_QUESTION = (
	"How do visual, textual, and regional signals interact to shape perceived food sentiment, "
	"and does incorporating geographic context improve multimodal sentiment prediction?"
)


st.set_page_config(
	page_title="Yelp Food Intelligence App",
	page_icon="🍽️",
	layout="wide",
)


@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame | None:
	if not path.exists():
		return None
	return pd.read_csv(path)


def metric_card(label: str, value: str) -> None:
	st.metric(label, value)


def apply_app_theme() -> None:
	st.markdown(APP_CSS, unsafe_allow_html=True)

def _coerce_numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
	clean = df.copy()
	for column in columns:
		if column in clean.columns:
			clean[column] = pd.to_numeric(clean[column], errors="coerce")
	return clean


def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
	return max(lower, min(upper, value))


@st.cache_data(show_spinner=False)
def _get_demo_reference_data() -> dict[str, pd.DataFrame | None]:
	return {
		"region_stats": load_csv(OUTPUT_DIR / "region_sentiment_stats.csv"),
		"region_gap": load_csv(OUTPUT_DIR / "image_text_consistency_region_summary.csv"),
		"cuisine_spread": load_csv(OUTPUT_DIR / "cuisine_region_sentiment_spread.csv"),
		"mismatch": load_csv(OUTPUT_DIR / "image_text_consistency_predictions.csv"),
		"attention_no_region": load_csv(OUTPUT_DIR / "attention_Image_plus_Text.csv"),
		"attention_region": load_csv(OUTPUT_DIR / "attention_Image_plus_Text_plus_Region.csv"),
		"retrieval": load_csv(OUTPUT_DIR / "retrieval_text_to_image.csv"),
	}


def _get_demo_regions(reference_data: dict[str, pd.DataFrame | None]) -> list[str]:
	regions: set[str] = set()
	region_stats = reference_data.get("region_stats")
	if region_stats is not None and not region_stats.empty and "region" in region_stats.columns:
		regions.update(region_stats["region"].dropna().astype(str).tolist())
	mismatch = reference_data.get("mismatch")
	if mismatch is not None and not mismatch.empty and "region" in mismatch.columns:
		regions.update(mismatch["region"].dropna().astype(str).tolist())
	region_ids_path = OUTPUT_DIR / "region_to_id.pkl"
	if region_ids_path.exists():
		try:
			with open(region_ids_path, "rb") as file:
				region_to_id = pickle.load(file)
			if isinstance(region_to_id, dict):
				regions.update(str(region) for region in region_to_id.keys())
		except Exception:
			pass
	for scenario in DEMO_SCENARIOS.values():
		regions.add(str(scenario["region"]))
	return sorted(regions)


def _get_model_supported_cuisines(reference_data: dict[str, pd.DataFrame | None]) -> list[str]:
	model_categories_path = OUTPUT_DIR / "category_to_id.pkl"
	if model_categories_path.exists():
		try:
			with open(model_categories_path, "rb") as file:
				category_to_id = pickle.load(file)
			if isinstance(category_to_id, dict):
				model_categories = sorted(
					str(category)
					for category in category_to_id.keys()
					if str(category) != "<unknown_category>"
				)
				if model_categories:
					return model_categories
		except Exception:
			pass

	cuisines: set[str] = set()
	cuisine_spread = reference_data.get("cuisine_spread")
	if cuisine_spread is not None and not cuisine_spread.empty and "cuisine_cluster" in cuisine_spread.columns:
		cuisines.update(cuisine_spread["cuisine_cluster"].dropna().astype(str).tolist())
	mismatch = reference_data.get("mismatch")
	if mismatch is not None and not mismatch.empty and "cuisine" in mismatch.columns:
		cuisines.update(mismatch["cuisine"].dropna().astype(str).tolist())
	for scenario in DEMO_SCENARIOS.values():
		cuisines.add(scenario["cuisine"])
	return sorted(cuisines)


def _get_region_tab_cuisines(reference_data: dict[str, pd.DataFrame | None]) -> list[str]:
	cuisines: set[str] = set()
	cuisine_spread = reference_data.get("cuisine_spread")
	if cuisine_spread is not None and not cuisine_spread.empty and "cuisine_cluster" in cuisine_spread.columns:
		cuisines.update(cuisine_spread["cuisine_cluster"].dropna().astype(str).tolist())
	mismatch = reference_data.get("mismatch")
	if mismatch is not None and not mismatch.empty and "cuisine" in mismatch.columns:
		cuisines.update(mismatch["cuisine"].dropna().astype(str).tolist())
	for scenario in DEMO_SCENARIOS.values():
		cuisines.add(scenario["cuisine"])
	return sorted(cuisines)


def _load_sample_demo_image(reference_data: dict[str, pd.DataFrame | None]) -> str | None:
	retrieval_df = reference_data.get("retrieval")
	if retrieval_df is not None and not retrieval_df.empty and "image_path" in retrieval_df.columns:
		for raw_path in retrieval_df["image_path"].dropna().astype(str).tolist():
			image_path = Path(raw_path)
			if image_path.is_file():
				return str(image_path)
	if PHOTO_IMAGES_DIR.exists() and PHOTO_IMAGES_DIR.is_dir():
		for pattern in ("*.jpg", "*.jpeg", "*.png"):
			match = next(PHOTO_IMAGES_DIR.glob(pattern), None)
			if match is not None and match.is_file():
				return str(match)
	return None


@st.cache_data(show_spinner=False)
def _load_city_quick_load_examples() -> pd.DataFrame:
	if not CITY_QUICK_LOAD_EXAMPLES_PATH.exists():
		return pd.DataFrame(columns=["region", "review_text", "cuisine", "image_path"])
	examples = pd.read_csv(CITY_QUICK_LOAD_EXAMPLES_PATH)
	required_columns = ["region", "review_text", "cuisine", "image_path"]
	if not set(required_columns).issubset(examples.columns):
		return pd.DataFrame(columns=required_columns)
	examples = examples.dropna(subset=required_columns).copy()
	for column in required_columns:
		examples[column] = examples[column].astype(str).str.strip()
	examples = examples[
		(examples["region"] != "") &
		(examples["review_text"] != "") &
		(examples["cuisine"] != "") &
		(examples["image_path"] != "")
	].copy()
	examples["image_exists"] = examples["image_path"].map(lambda value: Path(value).is_file())
	return examples[examples["image_exists"]].drop(columns=["image_exists"])


def _get_city_quick_load_regions() -> list[str]:
	examples = _load_city_quick_load_examples()
	if examples.empty:
		return []
	return sorted(examples["region"].dropna().astype(str).unique().tolist())


def _sample_city_quick_load_example(region: str) -> dict[str, str] | None:
	examples = _load_city_quick_load_examples()
	if examples.empty:
		return None
	region_examples = examples[examples["region"] == region]
	if region_examples.empty:
		return None
	row = region_examples.sample(n=1).iloc[0]
	return {
		"review": str(row["review_text"]),
		"region": str(row["region"]),
		"cuisine": str(row["cuisine"]),
		"image_path": str(row["image_path"]),
		"use_sample_image": False,
	}


def _construct_model(model_cls, *args, **kwargs):
	try:
		return model_cls(*args, **kwargs)
	except TypeError as exc:
		if "use_pretrained" in kwargs and "unexpected keyword argument 'use_pretrained'" in str(exc):
			fallback_kwargs = dict(kwargs)
			fallback_kwargs.pop("use_pretrained", None)
			return model_cls(*args, **fallback_kwargs)
		raise


def _get_upload_quick_load_options(regions: list[str] | None = None) -> list[dict[str, str]]:
	options = [{
		"label": "Custom input",
		"review": "",
		"region": DEMO_SCENARIOS["Custom input"]["region"],
		"cuisine": DEMO_SCENARIOS["Custom input"]["cuisine"],
		"image_path": "",
		"use_sample_image": False,
	}]
	for label, scenario in DEMO_SCENARIOS.items():
		if label == "Custom input":
			continue
		options.append({
			"label": label,
			"review": scenario["review"],
			"region": scenario["region"],
			"cuisine": scenario["cuisine"],
			"image_path": "",
			"use_sample_image": True,
		})
	city_example_regions = set(_get_city_quick_load_regions())
	for region in regions or []:
		region = str(region).strip()
		if not region or region not in city_example_regions:
			continue
		options.append({
			"label": f"City: {region}",
			"region": region,
			"city_preset": True,
		})
	deduped_options: list[dict[str, str]] = []
	seen_labels: set[str] = set()
	for option in options:
		label = str(option.get("label", "")).strip()
		if not label or label in seen_labels:
			continue
		seen_labels.add(label)
		deduped_options.append(option)
	return deduped_options


@st.cache_data(show_spinner=False)
def _load_model_supported_cuisine_set() -> set[str]:
	model_categories_path = OUTPUT_DIR / "category_to_id.pkl"
	if not model_categories_path.exists():
		return set()
	try:
		with open(model_categories_path, "rb") as file:
			category_to_id = pickle.load(file)
	except Exception:
		return set()
	if not isinstance(category_to_id, dict):
		return set()
	return {
		str(category).strip()
		for category in category_to_id.keys()
		if str(category).strip() and str(category) != "<unknown_category>"
	}


@st.cache_resource(show_spinner=False)
def _load_checkpoint_inference_bundle() -> dict[str, object]:
	try:
		import torch
		import yelp as yelp_module
		from torchvision import transforms
	except Exception as exc:
		return {
			"error": (
				f"Unable to import the model runtime: {exc}\n\n"
				f"Python interpreter: {sys.executable}\n"
				"Install missing deps in this same environment, e.g.:\n"
				"python -m pip install torch torchvision\n"
				"Then restart Streamlit."
			)
		}

	checkpoint_path = OUTPUT_DIR / "best_Image_plus_Text_plus_Region.pt"
	if not checkpoint_path.exists():
		return {"error": f"Missing checkpoint: {checkpoint_path}"}

	try:
		with open(OUTPUT_DIR / "tokenizer.pkl", "rb") as file:
			tokenizer = pickle.load(file)
		with open(OUTPUT_DIR / "region_to_id.pkl", "rb") as file:
			region_to_id = pickle.load(file)
		with open(OUTPUT_DIR / "category_to_id.pkl", "rb") as file:
			category_to_id = pickle.load(file)
		with open(OUTPUT_DIR / "id_to_category.pkl", "rb") as file:
			id_to_category = pickle.load(file)
		with open(OUTPUT_DIR / "mappings_meta.pkl", "rb") as file:
			mappings_meta = pickle.load(file)

		if not isinstance(region_to_id, dict) or not isinstance(category_to_id, dict) or not isinstance(id_to_category, dict):
			return {"error": "Saved mapping artifacts are invalid or unreadable."}

		unknown_region_id = int(mappings_meta.get("unknown_region_id", len(region_to_id)))
		unknown_category_id = int(mappings_meta.get("unknown_category_id", category_to_id.get("<unknown_category>", len(category_to_id))))
		category_count = int(mappings_meta.get("num_categories", len(category_to_id)))

		image_transform = transforms.Compose([
			transforms.Resize((yelp_module.IMAGE_SIZE, yelp_module.IMAGE_SIZE)),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		])

		device = torch.device("cpu")
		model = _construct_model(
			yelp_module.MultimodalFusionModel,
			yelp_module.VOCAB_SIZE,
			unknown_region_id + 1,
			category_count,
			use_pretrained=False,
		)
		state_dict = torch.load(checkpoint_path, map_location=device)
		model.load_state_dict(state_dict)
		model = model.to(device)
		model.eval()
	except Exception as exc:
		return {"error": f"Unable to prepare checkpoint inference: {exc}"}

	return {
		"torch": torch,
		"yelp_module": yelp_module,
		"device": device,
		"model": model,
		"tokenizer": tokenizer,
		"image_transform": image_transform,
		"region_to_id": region_to_id,
		"unknown_region_id": unknown_region_id,
		"category_to_id": category_to_id,
		"unknown_category_id": unknown_category_id,
		"id_to_category": id_to_category,
	}


def _resolve_demo_image(uploaded_image, sample_image_path: str | None, use_sample_image: bool) -> tuple[Image.Image | None, str | None]:
	if uploaded_image is not None:
		uploaded_image.seek(0)
		return Image.open(uploaded_image).convert("RGB"), "Uploaded image"
	if use_sample_image and sample_image_path:
		return Image.open(sample_image_path).convert("RGB"), "Sample dataset image"
	return None, None



def _get_image_from_session_or_sample(image_bytes, sample_image_path: str | None, use_sample: bool):
	"""Return a file-like image object from session bytes or the sample image path."""
	if image_bytes:
		return io.BytesIO(image_bytes)

	if use_sample and sample_image_path:
		with open(sample_image_path, "rb") as file:
			return io.BytesIO(file.read())

	return None

def _analyze_text_signal(review_text: str) -> dict[str, object]:
	tokens = re.findall(r"[a-z']+", review_text.lower())
	positive_hits = [token for token in tokens if token in POSITIVE_TERMS]
	negative_hits = [token for token in tokens if token in NEGATIVE_TERMS]
	delta = len(positive_hits) - len(negative_hits)
	base_score = 0.55 + 0.08 * delta
	negation_patterns = ["but", "however", "although", "except"]
	if any(pattern in review_text.lower() for pattern in negation_patterns) and negative_hits:
		base_score -= 0.08
	if len(tokens) < 8:
		base_score = (base_score + 0.5) / 2
	return {
		"score": _clip(base_score),
		"positive_hits": positive_hits,
		"negative_hits": negative_hits,
		"summary": f"{len(positive_hits)} positive cues vs {len(negative_hits)} negative cues in the review text.",
	}


def _analyze_image_signal(uploaded_image) -> dict[str, object]:
	if uploaded_image is None:
		return {
			"score": 0.5,
			"summary": "No uploaded image, so the demo keeps the visual signal neutral.",
			"brightness": None,
			"contrast": None,
			"warmth": None,
		}

	uploaded_image.seek(0)
	image = Image.open(uploaded_image).convert("RGB")
	uploaded_image.seek(0)
	gray = image.convert("L")
	rgb_stats = ImageStat.Stat(image)
	gray_stats = ImageStat.Stat(gray)
	r_mean, g_mean, b_mean = [channel / 255.0 for channel in rgb_stats.mean[:3]]
	brightness = gray_stats.mean[0] / 255.0
	contrast = _clip(gray_stats.stddev[0] / 90.0)
	warmth = _clip(0.5 + ((r_mean - b_mean) / 2.0))
	colorfulness = _clip((abs(r_mean - g_mean) + abs(r_mean - b_mean) + abs(g_mean - b_mean)) / 1.5)
	balanced_brightness = _clip(1.0 - abs(brightness - 0.62) / 0.62)
	visual_score = _clip(0.35 * balanced_brightness + 0.25 * contrast + 0.20 * warmth + 0.20 * colorfulness)
	return {
		"score": visual_score,
		"summary": f"Visual score combines brightness ({brightness:.0%}), contrast ({contrast:.0%}), and warmth ({warmth:.0%}).",
		"brightness": brightness,
		"contrast": contrast,
		"warmth": warmth,
	}


def _analyze_pil_image_signal(image: Image.Image | None) -> dict[str, object]:
	if image is None:
		return _analyze_image_signal(None)
	gray = image.convert("L")
	rgb_stats = ImageStat.Stat(image)
	gray_stats = ImageStat.Stat(gray)
	r_mean, g_mean, b_mean = [channel / 255.0 for channel in rgb_stats.mean[:3]]
	brightness = gray_stats.mean[0] / 255.0
	contrast = _clip(gray_stats.stddev[0] / 90.0)
	warmth = _clip(0.5 + ((r_mean - b_mean) / 2.0))
	colorfulness = _clip((abs(r_mean - g_mean) + abs(r_mean - b_mean) + abs(g_mean - b_mean)) / 1.5)
	balanced_brightness = _clip(1.0 - abs(brightness - 0.62) / 0.62)
	visual_score = _clip(0.35 * balanced_brightness + 0.25 * contrast + 0.20 * warmth + 0.20 * colorfulness)
	return {
		"score": visual_score,
		"summary": f"Visual score combines brightness ({brightness:.0%}), contrast ({contrast:.0%}), and warmth ({warmth:.0%}).",
		"brightness": brightness,
		"contrast": contrast,
		"warmth": warmth,
	}


def _lookup_region_metrics(reference_data: dict[str, pd.DataFrame | None], region: str) -> dict[str, float | None]:
	region_stats = reference_data.get("region_stats")
	region_gap = reference_data.get("region_gap")
	positive_rate = None
	mismatch_rate = None
	if region_stats is not None and not region_stats.empty and {"region", "positive_rate"}.issubset(region_stats.columns):
		stats = region_stats[region_stats["region"] == region]
		if not stats.empty:
			positive_rate = float(pd.to_numeric(stats.iloc[0]["positive_rate"], errors="coerce"))
		else:
			positive_rate = float(pd.to_numeric(region_stats["positive_rate"], errors="coerce").mean())
	if region_gap is not None and not region_gap.empty and {"region", "mismatch_rate"}.issubset(region_gap.columns):
		gap = region_gap[region_gap["region"] == region]
		if not gap.empty:
			mismatch_rate = float(pd.to_numeric(gap.iloc[0]["mismatch_rate"], errors="coerce"))
		else:
			mismatch_rate = float(pd.to_numeric(region_gap["mismatch_rate"], errors="coerce").mean())
	return {
		"positive_rate": positive_rate,
		"mismatch_rate": mismatch_rate,
	}


def _lookup_cuisine_metrics(reference_data: dict[str, pd.DataFrame | None], cuisine: str) -> dict[str, float | None]:
	cuisine_spread = reference_data.get("cuisine_spread")
	if cuisine_spread is None or cuisine_spread.empty or "cuisine_cluster" not in cuisine_spread.columns:
		return {"min_rate": None, "max_rate": None, "spread": None, "region_count": None}
	match = cuisine_spread[cuisine_spread["cuisine_cluster"] == cuisine]
	if match.empty:
		return {
			"min_rate": float(pd.to_numeric(cuisine_spread["min_rate"], errors="coerce").median()),
			"max_rate": float(pd.to_numeric(cuisine_spread["max_rate"], errors="coerce").median()),
			"spread": float(pd.to_numeric(cuisine_spread["spread"], errors="coerce").median()),
			"region_count": float(pd.to_numeric(cuisine_spread["region_count"], errors="coerce").median()),
		}
	row = match.iloc[0]
	return {
		"min_rate": float(pd.to_numeric(row["min_rate"], errors="coerce")),
		"max_rate": float(pd.to_numeric(row["max_rate"], errors="coerce")),
		"spread": float(pd.to_numeric(row["spread"], errors="coerce")),
		"region_count": float(pd.to_numeric(row["region_count"], errors="coerce")),
	}


def _estimate_demo_prediction(
	review_text: str,
	uploaded_image,
	region: str,
	cuisine: str,
	reference_data: dict[str, pd.DataFrame | None],
	sample_image_path: str | None,
	use_sample_image: bool,
) -> dict[str, object]:
	bundle = _load_checkpoint_inference_bundle()
	if "error" in bundle:
		raise RuntimeError(str(bundle["error"]))

	torch = bundle["torch"]
	yelp_module = bundle["yelp_module"]
	device = bundle["device"]
	model = bundle["model"]
	tokenizer = bundle["tokenizer"]
	image_transform = bundle["image_transform"]
	region_to_id = bundle["region_to_id"]
	unknown_region_id = bundle["unknown_region_id"]
	category_to_id = bundle["category_to_id"]
	unknown_category_id = bundle["unknown_category_id"]
	id_to_category = bundle["id_to_category"]

	pil_image, image_source = _resolve_demo_image(uploaded_image, sample_image_path, use_sample_image)
	if pil_image is None:
		raise RuntimeError("Please upload a food image or enable the sample dataset image for checkpoint inference.")

	text_result = _analyze_text_signal(review_text)
	image_result = {
		**_analyze_pil_image_signal(pil_image),
		"summary": f"Inference image source: {image_source}. {_analyze_pil_image_signal(pil_image)['summary']}",
	}
	region_metrics = _lookup_region_metrics(reference_data, region)
	cuisine_metrics = _lookup_cuisine_metrics(reference_data, cuisine)
	region_score = region_metrics["positive_rate"] if region_metrics["positive_rate"] is not None else 0.72

	image_tensor = image_transform(pil_image).unsqueeze(0).to(device)
	text_seq = yelp_module.encode_texts(tokenizer, [review_text], max_len=yelp_module.MAX_TEXT_LEN)
	text_tensor = torch.tensor(text_seq, dtype=torch.long, device=device)
	region_tensor = torch.tensor([region_to_id.get(region, unknown_region_id)], dtype=torch.long, device=device)
	category_tensor = torch.tensor([category_to_id.get(cuisine, unknown_category_id)], dtype=torch.long, device=device)

	with torch.no_grad():
		sentiment_logits, category_logits, rating_pred, attention_stats = model(
			image_tensor,
			text_tensor,
			region_tensor,
			category_tensor,
			return_attention=True,
		)
		positive_probability = float(torch.sigmoid(sentiment_logits).item())
		predicted_category_id = int(torch.argmax(category_logits, dim=1).item())
		predicted_rating = float(rating_pred.item())
		image_attention_raw = float(attention_stats["img_concentration"].squeeze().item())
		text_attention_raw = float(attention_stats["txt_concentration"].squeeze().item())

	final_score = _clip(positive_probability)
	confidence = _clip(max(final_score, 1.0 - final_score))
	label = "Positive" if final_score >= 0.5 else "Negative"
	attention_total = image_attention_raw + text_attention_raw
	if attention_total <= 0:
		text_focus = 0.5
		image_focus = 0.5
	else:
		text_focus = text_attention_raw / attention_total
		image_focus = image_attention_raw / attention_total

	if text_focus >= image_focus:
		attention_explanation = "The review text is carrying more of the decision than the image in this case."
	else:
		attention_explanation = "The food image is carrying more of the decision than the text in this case."
	attention_explanation += f" Raw model attention concentration: text {text_attention_raw:.1%}, image {image_attention_raw:.1%}."

	if label == "Negative" and text_focus >= image_focus:
		business_insight = "This looks like a service or taste problem rather than a presentation problem. Route it to review-response or operations teams first."
	elif label == "Negative":
		business_insight = "This looks visually weak. Restaurant teams should inspect plating, photography quality, or delivery condition."
	elif (region_metrics["mismatch_rate"] or 0) > 0.4:
		business_insight = "The item trends positive, but this is a high-variance market. Personalization or local messaging may matter more than average sentiment suggests."
	else:
		business_insight = "This is a strong candidate for recommendation surfaces, featured menu placement, or paid promotion in similar markets."

	region_insight = f"{region} has an estimated positive-rate baseline of {region_score:.0%}"
	if region_metrics["mismatch_rate"] is not None:
		region_insight += f" and an image/text mismatch rate of {region_metrics['mismatch_rate']:.0%}."
	else:
		region_insight += "."
	if cuisine_metrics["spread"] is not None:
		region_insight += f" {cuisine} shows regional sentiment spread of {cuisine_metrics['spread']:.2f} across {int(cuisine_metrics['region_count'] or 0)} regions."
	region_insight += f" The multitask head predicts a normalized star score of {predicted_rating:.2f} and category alignment closest to {id_to_category.get(predicted_category_id, '<unknown_category>')}."

	return {
		"label": label,
		"positive_probability": final_score,
		"negative_probability": 1.0 - final_score,
		"confidence": confidence,
		"input_cuisine": cuisine,
		"text_focus": text_focus,
		"image_focus": image_focus,
		"text_result": text_result,
		"image_result": image_result,
		"region_insight": region_insight,
		"attention_explanation": attention_explanation,
		"business_insight": business_insight,
		"region_metrics": region_metrics,
		"cuisine_metrics": cuisine_metrics,
		"predicted_rating": predicted_rating,
		"predicted_category": id_to_category.get(predicted_category_id, "<unknown_category>"),
		"image_source": image_source,
	}


@st.cache_resource(show_spinner=False)
def _load_multimodel_inference_bundle() -> dict[str, object]:
	base_bundle = _load_checkpoint_inference_bundle()
	if "error" in base_bundle:
		return base_bundle

	torch = base_bundle["torch"]
	yelp_module = base_bundle["yelp_module"]
	device = base_bundle["device"]

	checkpoint_map = {
		"Image Only": OUTPUT_DIR / "best_Image_Only.pt",
		"Text Only": OUTPUT_DIR / "best_Text_Only.pt",
		"Image + Text": OUTPUT_DIR / "best_Image_plus_Text.pt",
	}

	try:
		for label, path in checkpoint_map.items():
			if not path.exists():
				return {"error": f"Missing checkpoint for {label}: {path}"}

		image_model = _construct_model(yelp_module.ImageOnlyModel, use_pretrained=False)
		image_model.load_state_dict(torch.load(checkpoint_map["Image Only"], map_location=device))
		image_model = image_model.to(device)
		image_model.eval()

		text_model = yelp_module.TextOnlyModel(yelp_module.VOCAB_SIZE, yelp_module.EMBED_DIM, yelp_module.LSTM_HIDDEN)
		text_model.load_state_dict(torch.load(checkpoint_map["Text Only"], map_location=device))
		text_model = text_model.to(device)
		text_model.eval()

		image_text_model = _construct_model(
			yelp_module.ImageTextFusionModel,
			yelp_module.VOCAB_SIZE,
			use_pretrained=False,
		)
		image_text_model.load_state_dict(torch.load(checkpoint_map["Image + Text"], map_location=device))
		image_text_model = image_text_model.to(device)
		image_text_model.eval()
	except Exception as exc:
		return {"error": f"Unable to load ablation checkpoints: {exc}"}

	return {
		**base_bundle,
		"image_model": image_model,
		"text_model": text_model,
		"image_text_model": image_text_model,
		"full_model": base_bundle["model"],
	}


def _run_multimodal_comparison(
	review_text: str,
	uploaded_image,
	region: str,
	cuisine: str,
	sample_image_path: str | None,
	use_sample_image: bool,
) -> dict[str, object]:
	bundle = _load_multimodel_inference_bundle()
	if "error" in bundle:
		raise RuntimeError(str(bundle["error"]))

	torch = bundle["torch"]
	yelp_module = bundle["yelp_module"]
	device = bundle["device"]
	tokenizer = bundle["tokenizer"]
	image_transform = bundle["image_transform"]
	region_to_id = bundle["region_to_id"]
	unknown_region_id = bundle["unknown_region_id"]
	category_to_id = bundle["category_to_id"]
	unknown_category_id = bundle["unknown_category_id"]
	id_to_category = bundle["id_to_category"]

	image_model = bundle["image_model"]
	text_model = bundle["text_model"]
	image_text_model = bundle["image_text_model"]
	full_model = bundle["full_model"]

	pil_image, image_source = _resolve_demo_image(uploaded_image, sample_image_path, use_sample_image)
	if pil_image is None:
		raise RuntimeError("Please upload a food image or enable the sample dataset image.")

	image_tensor = image_transform(pil_image).unsqueeze(0).to(device)
	text_seq = yelp_module.encode_texts(tokenizer, [review_text], max_len=yelp_module.MAX_TEXT_LEN)
	text_tensor = torch.tensor(text_seq, dtype=torch.long, device=device)
	region_tensor = torch.tensor([region_to_id.get(region, unknown_region_id)], dtype=torch.long, device=device)
	category_tensor = torch.tensor([category_to_id.get(cuisine, unknown_category_id)], dtype=torch.long, device=device)

	with torch.no_grad():
		image_prob = float(torch.sigmoid(image_model(image_tensor)).item())
		text_prob = float(torch.sigmoid(text_model(text_tensor)).item())
		image_text_prob = float(torch.sigmoid(image_text_model(image_tensor, text_tensor)).item())
		full_sentiment_logits, full_category_logits, full_rating_pred = full_model(
			image_tensor,
			text_tensor,
			region_tensor,
			category_tensor,
		)
		full_prob = float(torch.sigmoid(full_sentiment_logits).item())
		full_rating = float(full_rating_pred.item())
		predicted_category_id = int(torch.argmax(full_category_logits, dim=1).item())

	def to_row(model_name: str, positive_prob: float) -> dict[str, object]:
		label = "Positive" if positive_prob >= 0.5 else "Negative"
		confidence = max(positive_prob, 1.0 - positive_prob)
		return {
			"Model": model_name,
			"Prediction": label,
			"Confidence": confidence,
		}

	comparison_rows = [
		to_row("Image Only", image_prob),
		to_row("Text Only", text_prob),
		to_row("Image + Text", image_text_prob),
		to_row("Image + Text + Region", full_prob),
	]

	return {
		"rows": comparison_rows,
		"predicted_rating": full_rating,
		"predicted_category": id_to_category.get(predicted_category_id, "<unknown_category>"),
		"image_source": image_source,
	}


def render_region_impact_demo() -> None:
	st.header("Region & Cuisine / Category")
	st.caption(
		"Explore how sentiment varies by location and how the same cuisine changes across markets. "
		"These metrics come from precomputed dataset summaries, not your live upload."
	)

	reference_data = _get_demo_reference_data()
	regions_df = reference_data.get("region_stats")
	cuisine_spread_df = reference_data.get("cuisine_spread")
	mismatch_df = reference_data.get("mismatch")
	ablation_df = load_csv(OUTPUT_DIR / "ablation_results.csv")

	cuisines = _get_region_tab_cuisines(reference_data)
	regions = _get_demo_regions(reference_data)
	selected_cuisine = st.selectbox("Cuisine / category", cuisines, key="region_impact_cuisine")
	selected_region = st.selectbox("Region", regions, key="region_impact_region")
	cuisine_metrics = _lookup_cuisine_metrics(reference_data, selected_cuisine)

	metric_row1_col1, metric_row1_col2 = st.columns(2)
	with metric_row1_col1:
		if regions_df is not None and not regions_df.empty and {"region", "positive_rate"}.issubset(regions_df.columns):
			region_row = regions_df[regions_df["region"] == selected_region]
			if not region_row.empty:
				st.metric("Selected region positive rate", f"{float(region_row.iloc[0]['positive_rate']):.1%}")
			else:
				st.metric("Selected region positive rate", "N/A")
		else:
			st.metric("Selected region positive rate", "N/A")

	with metric_row1_col2:
		if cuisine_metrics["spread"] is not None:
			st.metric("Cuisine / category spread across regions", f"{float(cuisine_metrics['spread']):.2f}")
		else:
			st.metric("Cuisine / category spread across regions", "N/A")

	metric_row2_col1, metric_row2_col2 = st.columns(2)
	with metric_row2_col1:
		if cuisine_metrics["min_rate"] is not None:
			st.metric("Cuisine / category min regional rate", f"{float(cuisine_metrics['min_rate']):.1%}")
		else:
			st.metric("Cuisine / category min regional rate", "N/A")

	with metric_row2_col2:
		if cuisine_metrics["max_rate"] is not None:
			st.metric("Cuisine / category max regional rate", f"{float(cuisine_metrics['max_rate']):.1%}")
		else:
			st.metric("Cuisine / category max regional rate", "N/A")

	metric_row3_col1, metric_row3_col2 = st.columns(2)
	with metric_row3_col1:
		if cuisine_metrics["region_count"] is not None:
			st.metric("Regions in cuisine/category summary", f"{int(cuisine_metrics['region_count'])}")
		else:
			st.metric("Regions in cuisine/category summary", "N/A")

	with metric_row3_col2:
		if ablation_df is not None and not ablation_df.empty and {"Model", "F1"}.issubset(ablation_df.columns):
			ablation = _coerce_numeric_columns(ablation_df, ["F1"])
			row_it = ablation[ablation["Model"] == "Image + Text"]
			row_full = ablation[ablation["Model"] == "Image + Text + Region"]
			if not row_it.empty and not row_full.empty:
				delta = float(row_full.iloc[0]["F1"] - row_it.iloc[0]["F1"])
				st.metric("Global F1 gain from region", f"{delta:+.4f}")
			else:
				st.metric("Global F1 gain from region", "N/A")
		else:
			st.metric("Global F1 gain from region", "N/A")

	left, right = st.columns(2)
	with left:
		if regions_df is not None and not regions_df.empty and {"region", "positive_rate"}.issubset(regions_df.columns):
			region_chart_df = _coerce_numeric_columns(regions_df, ["positive_rate", "total"])
			fig = px.bar(
					region_chart_df.sort_values("positive_rate", ascending=False).head(15),
					x="region",
					y="positive_rate",
					color="positive_rate",
					text="positive_rate",
					title="Top regions by positive sentiment rate",
				)
			fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
			fig.update_yaxes(tickformat=".0%")
			st.plotly_chart(fig, width="stretch", key="region_impact_positive_rate")
		else:
			st.info("Region sentiment stats are unavailable.")

	with right:
		if mismatch_df is not None and not mismatch_df.empty and {"cuisine", "region", "text_prob"}.issubset(mismatch_df.columns):
			cuisine_slice = mismatch_df[mismatch_df["cuisine"] == selected_cuisine].copy()
			if not cuisine_slice.empty:
				cuisine_slice = _coerce_numeric_columns(cuisine_slice, ["text_prob"])
				top_regions = (
					cuisine_slice.groupby("region", as_index=False)["text_prob"].mean()
					.sort_values("text_prob", ascending=False)
					.head(10)
				)
				fig = px.bar(
					top_regions,
					x="region",
					y="text_prob",
					color="text_prob",
					text="text_prob",
					title=f"Top regions for {selected_cuisine} by text-sentiment proxy",
				)
				fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
				fig.update_yaxes(tickformat=".0%")
				st.plotly_chart(fig, width="stretch", key="region_impact_top_cuisine_regions")
			else:
				st.info(f"No region-level rows available for cuisine '{selected_cuisine}'.")
		else:
			st.info("Cuisine-region sentiment rows are unavailable.")


def render_perception_gap_explorer() -> None:
	st.header("Perception Gap Explorer")
	st.caption(
		"Inspect validation-set cases where the image-only and text-only checkpoints disagree. "
		"This is based on saved model predictions, not live user input."
	)

	mismatch_df = load_csv(OUTPUT_DIR / "image_text_consistency_predictions.csv")
	if mismatch_df is None or mismatch_df.empty:
		st.info("No perception-gap outputs found. Run yelp.py to generate image-text consistency files.")
		return

	required_columns = {
		"region", "cuisine", "label", "image_prob", "text_prob",
		"image_pred", "text_pred", "mismatch", "img_pos_txt_neg", "img_neg_txt_pos",
	}
	missing_columns = sorted(required_columns - set(mismatch_df.columns))
	if missing_columns:
		st.warning(
			"Perception-gap output is missing required columns: "
			+ ", ".join(missing_columns)
		)
		return

	mismatch_df = _coerce_numeric_columns(
		mismatch_df,
		[
			"label", "image_prob", "text_prob", "image_pred", "text_pred",
			"mismatch", "img_pos_txt_neg", "img_neg_txt_pos",
		],
	)
	for column in ("mismatch", "img_pos_txt_neg", "img_neg_txt_pos"):
		if column in mismatch_df.columns:
			numeric_values = pd.to_numeric(mismatch_df[column], errors="coerce")
			mismatch_df[column] = np.where(
				numeric_values.notna(),
				numeric_values.fillna(0).astype(int) != 0,
				mismatch_df[column].astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
			)
	for column in ("label", "image_pred", "text_pred"):
		mismatch_df[column] = pd.to_numeric(mismatch_df[column], errors="coerce").fillna(0).astype(int)

	mismatch_df["true_sentiment"] = np.where(mismatch_df["label"] == 1, "Positive", "Negative")
	mismatch_df["image_prediction"] = np.where(mismatch_df["image_pred"] == 1, "Positive", "Negative")
	mismatch_df["text_prediction"] = np.where(mismatch_df["text_pred"] == 1, "Positive", "Negative")
	mismatch_df["gap_direction"] = np.select(
		[
			mismatch_df["img_pos_txt_neg"],
			mismatch_df["img_neg_txt_pos"],
		],
		[
			"Image positive / Text negative",
			"Image negative / Text positive",
		],
		default="No gap",
	)

	mode = st.radio(
		"Gap mode",
		["Image positive / Text negative", "Image negative / Text positive", "All mismatches"],
		horizontal=True,
	)
	if mode == "Image positive / Text negative":
		st.info("This means the food image looked positive, but the review text was negative.")
	elif mode == "Image negative / Text positive":
		st.info("This means the food image looked negative, but the review text was positive.")
	else:
		st.info("This shows all cases where image-only and text-only predictions disagree.")
	regions = ["All"] + sorted(mismatch_df["region"].dropna().astype(str).unique().tolist())
	cuisines = ["All"] + sorted(mismatch_df["cuisine"].dropna().astype(str).unique().tolist())
	selected_region = st.selectbox("Region filter", regions, key="gap_region_filter")
	selected_cuisine = st.selectbox("Cuisine / category filter", cuisines, key="gap_cuisine_filter")

	filtered = mismatch_df.copy()
	if mode == "Image positive / Text negative":
		filtered = filtered[filtered["img_pos_txt_neg"]]
	elif mode == "Image negative / Text positive":
		filtered = filtered[filtered["img_neg_txt_pos"]]
	else:
		filtered = filtered[filtered["mismatch"]]

	if selected_region != "All":
		filtered = filtered[filtered["region"] == selected_region]
	if selected_cuisine != "All":
		filtered = filtered[filtered["cuisine"] == selected_cuisine]

	left, right = st.columns(2)
	with left:
		st.metric("Filtered mismatch cases", f"{len(filtered):,}")
	with right:
		if len(mismatch_df) > 0:
			st.metric("Overall mismatch rate", f"{float(mismatch_df['mismatch'].mean()):.1%}")

	metric_col1, metric_col2, metric_col3 = st.columns(3)
	with metric_col1:
		st.metric("Total validation rows", f"{len(mismatch_df):,}")
	with metric_col2:
		st.metric("Image positive / Text negative", f"{int(mismatch_df['img_pos_txt_neg'].sum()):,}")
	with metric_col3:
		st.metric("Image negative / Text positive", f"{int(mismatch_df['img_neg_txt_pos'].sum()):,}")

	if filtered.empty:
		st.warning("No rows match the selected filters.")
		return

	scatter_df = filtered.head(800)
	fig = px.scatter(
		scatter_df,
		x="image_prob",
		y="text_prob",
		color="region",
		hover_data=["cuisine", "true_sentiment", "image_prediction", "text_prediction", "gap_direction"],
		title="Perception gap cases (image probability vs text probability)",
		labels={
			"image_prob": "Image-only positive probability",
			"text_prob": "Text-only positive probability",
			"region": "Region",
		},
	)
	fig.add_hline(y=0.5, line_dash="dot")
	fig.add_vline(x=0.5, line_dash="dot")
	st.plotly_chart(fig, width="stretch", key="gap_scatter")

	st.markdown("**Filtered examples**")
	display_columns = [
		"region",
		"cuisine",
		"true_sentiment",
		"gap_direction",
		"image_prob",
		"text_prob",
		"image_prediction",
		"text_prediction",
	]
	st.dataframe(
		filtered[display_columns].head(120),
		width="stretch",
		hide_index=True,
	)

	st.info(
		"Perception gap means the image-only checkpoint and text-only checkpoint disagree. "
		"The saved file does not include the original review text or image path, so this tab shows model-level disagreement by region and cuisine."
	)


def missing_pipeline_inputs() -> list[Path]:
	return [path for path in REQUIRED_PIPELINE_INPUTS if not path.exists()]


def _clear_upload_review_results() -> None:
	for key in ("last_prediction_result", "last_comparison_result"):
		st.session_state.pop(key, None)


def _load_upload_review_scenario(option: dict[str, str]) -> None:
	if option.get("label") == "Custom input":
		st.session_state["input_review_text"] = ""
		st.session_state["input_region"] = DEMO_SCENARIOS["Custom input"]["region"]
		st.session_state["input_cuisine"] = DEMO_SCENARIOS["Custom input"]["cuisine"]
		st.session_state["input_use_sample"] = False
	elif option.get("city_preset"):
		city_example = _sample_city_quick_load_example(str(option.get("region", "")))
		if city_example is None:
			st.session_state["input_review_text"] = ""
			st.session_state["input_region"] = option.get("region", DEMO_SCENARIOS["Custom input"]["region"])
			st.session_state["input_use_sample"] = False
			st.session_state.pop("input_image_bytes", None)
			st.session_state.pop("input_image_name", None)
		else:
			st.session_state["input_review_text"] = city_example["review"]
			st.session_state["input_region"] = city_example["region"]
			image_path = Path(city_example["image_path"])
			st.session_state["input_image_bytes"] = image_path.read_bytes()
			st.session_state["input_image_name"] = image_path.name
			st.session_state["input_use_sample"] = False
	else:
		st.session_state["input_review_text"] = option.get("review", "")
		st.session_state["input_region"] = option.get("region", DEMO_SCENARIOS["Custom input"]["region"])
		st.session_state["input_cuisine"] = option.get("cuisine", DEMO_SCENARIOS["Custom input"]["cuisine"])
		st.session_state["input_use_sample"] = bool(option.get("use_sample_image", False))
		image_path = Path(option.get("image_path", ""))
		if image_path.is_file():
			st.session_state["input_image_bytes"] = image_path.read_bytes()
			st.session_state["input_image_name"] = image_path.name
			st.session_state["input_use_sample"] = False
		else:
			st.session_state.pop("input_image_bytes", None)
			st.session_state.pop("input_image_name", None)
	if option.get("label") == "Custom input":
		st.session_state.pop("input_image_bytes", None)
		st.session_state.pop("input_image_name", None)
	st.session_state["input_uploader_version"] = st.session_state.get("input_uploader_version", 0) + 1
	_clear_upload_review_results()


def _sync_selected_upload_scenario() -> None:
	options = st.session_state.get("upload_quick_load_options", [])
	selected_label = st.session_state.get("upload_scenario_select")
	selected_option = next(
		(option for option in options if option.get("label") == selected_label),
		None,
	)
	if selected_option is not None:
		_load_upload_review_scenario(selected_option)


def sidebar_controls() -> None:
	st.sidebar.title("Checkpoint / Output Status")
	st.sidebar.caption("Status only.")

	missing_inputs = missing_pipeline_inputs()
	if missing_inputs:
		st.sidebar.warning(
			"Pipeline inputs are missing in this environment. "
			"You can still explore the precomputed outputs below."
		)

		with st.sidebar.expander("Pipeline input status"):
			for path in REQUIRED_PIPELINE_INPUTS:
				status = "OK" if path.exists() else "Missing"
				st.write(f"{status}: {path}")
			st.caption("Configure one or more path variables via environment or Streamlit secrets.")
			st.caption(", ".join(DATA_PATH_ENV_VARS))










# ──────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Home
# ──────────────────────────────────────────────────────────────────────────────

def render_home() -> None:
	ablation_df = load_csv(OUTPUT_DIR / "ablation_results.csv")

	st.markdown(
		"""
		<div class="yfi-hero">
			<div class="yfi-section-label">Indiana University — Multimodal Sentiment Demo</div>
			<h1>Yelp Food Intelligence</h1>
			<p>
				Can a model look at a food photo, read a Yelp review, and factor in the local market
				to accurately predict whether the dining experience was positive or negative?
				This demo answers that question with four checkpointed models, real Yelp data, and transparent explanations.
			</p>
			<div class="yfi-badge-row">
				<div class="yfi-badge">Checkpoint-only inference — no training in demo</div>
				<div class="yfi-badge">4 ablation variants</div>
				<div class="yfi-badge">Region-aware fusion</div>
				<div class="yfi-badge">Perception gap analysis</div>
			</div>
		</div>
		""",
		unsafe_allow_html=True,
	)

	st.markdown("### What problem does this solve?")
	p1, p2, p3 = st.columns(3)
	with p1:
		st.markdown(
			"""
			<div class="yfi-panel">
				<h3>The gap problem</h3>
				<p>A dish can look beautiful in a photo but generate negative reviews.
				A text-only or image-only model misses this disconnect. Multimodal fusion catches it.</p>
			</div>
			""",
			unsafe_allow_html=True,
		)
	with p2:
		st.markdown(
			"""
			<div class="yfi-panel">
				<h3>The region problem</h3>
				<p>Customer expectations differ by city and cuisine.
				A seafood dish in Tampa has different local benchmarks than the same dish in Philadelphia.
				Geographic context improves predictions.</p>
			</div>
			""",
			unsafe_allow_html=True,
		)
	with p3:
		st.markdown(
			"""
			<div class="yfi-panel">
				<h3>The explainability problem</h3>
				<p>A black-box score is not actionable. This system shows whether the model
				leaned on image evidence or text evidence, turning a prediction into a specific
				operational recommendation.</p>
			</div>
			""",
			unsafe_allow_html=True,
		)

	st.markdown("### Research question")
	st.info(RESEARCH_QUESTION)

	st.markdown("### Model variants (inference only — checkpoints pre-trained)")
	model_rows = [
		{"Variant": "Image Only", "Inputs": "Food photo (ResNet18 CNN)", "Checkpoint": "best_Image_Only.pt"},
		{"Variant": "Text Only", "Inputs": "Review text (LSTM)", "Checkpoint": "best_Text_Only.pt"},
		{"Variant": "Image + Text", "Inputs": "Photo + review (CNN + LSTM fusion)", "Checkpoint": "best_Image_plus_Text.pt"},
		{"Variant": "Image + Text + Region", "Inputs": "Photo + review + city/cuisine context", "Checkpoint": "best_Image_plus_Text_plus_Region.pt"},
	]
	st.dataframe(pd.DataFrame(model_rows), hide_index=True, use_container_width=True)

	if ablation_df is not None and not ablation_df.empty and {"Model", "F1", "Accuracy"}.issubset(ablation_df.columns):
		ablation = _coerce_numeric_columns(ablation_df, ["F1", "Accuracy", "Precision", "Recall"])
		best_row = ablation.sort_values("F1", ascending=False).iloc[0]
		m1, m2, m3 = st.columns(3)
		with m1:
			st.metric("Best model", str(best_row["Model"]))
		with m2:
			st.metric("Best F1", f"{float(best_row['F1']):.4f}")
		with m3:
			st.metric("Best accuracy", f"{float(best_row['Accuracy']):.4f}")

	st.markdown("### Demo flow — use the tabs above in order")
	steps = [
		("Upload & Review", "Upload a food photo and type (or paste) a Yelp review. Choose region and cuisine."),
		("Predict Sentiment", "Run the full Image + Text + Region model. See label, confidence, and explanation."),
		("Compare Models", "Run all four variants on your input to see the effect of each modality."),
		("Region & Cuisine", "Explore how sentiment varies by city and cuisine across the dataset."),
		("Perception Gaps", "Browse cases where image and text sentiment disagree."),
		("Results & Limitations", "Full ablation table, key findings, and honest limitations."),
	]
	for i, (tab_name, description) in enumerate(steps, 1):
		st.markdown(f"**{i}. {tab_name}** — {description}")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Upload & Review
# ──────────────────────────────────────────────────────────────────────────────

def render_upload_review() -> None:
	reference_data = _get_demo_reference_data()
	regions = _get_demo_regions(reference_data)
	cuisines = _get_model_supported_cuisines(reference_data)
	sample_image_path = _load_sample_demo_image(reference_data)
	quick_load_options = _get_upload_quick_load_options(regions=regions)
	quick_load_labels = [option["label"] for option in quick_load_options]
	st.session_state["upload_quick_load_options"] = quick_load_options

	st.header("Upload Food Image & Enter Review")
	st.caption(
		"Provide a food photo and a Yelp-style review. "
		"Your inputs are saved to session state and shared with the Predict and Compare tabs."
	)

	# ── defaults ──────────────────────────────────────────────────────────────
	if "input_review_text" not in st.session_state:
		st.session_state["input_review_text"] = DEMO_SCENARIOS["Perception gap demo - Nashville"]["review"]
	if "input_region" not in st.session_state:
		st.session_state["input_region"] = DEMO_SCENARIOS["Perception gap demo - Nashville"]["region"]
	if "input_cuisine" not in st.session_state:
		st.session_state["input_cuisine"] = DEMO_SCENARIOS["Perception gap demo - Nashville"]["cuisine"]
	if "input_uploader_version" not in st.session_state:
		st.session_state["input_uploader_version"] = 0
	if "upload_scenario_select" not in st.session_state:
		st.session_state["upload_scenario_select"] = "Perception gap demo - Nashville"
	if st.session_state["upload_scenario_select"] not in quick_load_labels:
		st.session_state["upload_scenario_select"] = "Custom input"
	if st.session_state["input_region"] not in regions:
		regions = sorted([*regions, st.session_state["input_region"]])
	if st.session_state["input_cuisine"] not in cuisines:
		cuisines = sorted([*cuisines, st.session_state["input_cuisine"]])

	# ── scenario loader ───────────────────────────────────────────────────────
	selected_label = st.selectbox(
		"Quick-load a scenario",
		quick_load_labels,
		key="upload_scenario_select",
		on_change=_sync_selected_upload_scenario,
	)
	st.caption("Showing curated scenarios plus all dataset cities.")

	col1, col2 = st.columns((1.15, 0.85))
	with col1:
		uploaded_image = st.file_uploader(
			"Food image (optional — enable sample image below if none uploaded)",
			type=["png", "jpg", "jpeg"],
			key=f"input_uploader_{st.session_state['input_uploader_version']}",
		)
		use_sample = st.checkbox(
			"Use sample dataset image when no upload is provided",
			value=st.session_state.get("input_use_sample", bool(sample_image_path)),
			key="input_use_sample",
		)

		# Immediately persist bytes so they survive tab switches
		if uploaded_image is not None:
			uploaded_image.seek(0)
			st.session_state["input_image_bytes"] = uploaded_image.read()
			st.session_state["input_image_name"] = uploaded_image.name

		review_text = st.text_area(
			"Yelp review text",
			key="input_review_text",
			height=150,
			placeholder="Paste or type a food review here…",
		)

		default_region_idx = regions.index(st.session_state["input_region"]) if st.session_state["input_region"] in regions else 0
		st.selectbox("Region / city", regions, index=default_region_idx, key="input_region")

		default_cuisine_idx = cuisines.index(st.session_state["input_cuisine"]) if st.session_state["input_cuisine"] in cuisines else 0
		st.selectbox("Cuisine / category", cuisines, index=default_cuisine_idx, key="input_cuisine")
		st.caption("Choose the cuisine / category for prediction. City presets do not overwrite this field.")

		selected_option = next(
			(option for option in quick_load_options if option["label"] == selected_label),
			quick_load_options[0],
		)
		if selected_option["label"] == "Custom input":
			st.caption("Manual mode: provide your own review text, region, cuisine, and optional image.")
		elif selected_option["label"].startswith("City: "):
			st.caption(
				f"City preset for {selected_option['region']} loads a random real Yelp review and image for that city. Choose cuisine / category manually."
			)
		else:
			st.caption(
				f"Dataset example from {selected_option['region']} in {selected_option['cuisine']}."
			)

	with col2:
		st.markdown("### Input preview")
		image_bytes = st.session_state.get("input_image_bytes")
		if image_bytes:
			st.image(image_bytes, caption="Saved food image", use_container_width=True)
		elif use_sample and sample_image_path:
			st.image(sample_image_path, caption="Sample dataset image", use_container_width=True)
		else:
			st.info("Upload a food photo or enable the sample image.")

		saved_review = st.session_state.get("input_review_text", "")
		st.markdown(f"**Region:** {st.session_state.get('input_region', '—')}")
		st.markdown(f"**Cuisine / category:** {st.session_state.get('input_cuisine', '—')}")
		if saved_review:
			st.markdown("**Review snippet:**")
			snippet = saved_review[:160] + ("…" if len(saved_review) > 160 else "")
			st.caption(snippet)
		else:
			st.caption("No review text saved.")

	st.success(
		"Inputs are saved automatically. "
		"Switch to **Predict Sentiment** to run the model, or **Compare Models** to see all four variants."
	)


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Predict Sentiment
# ──────────────────────────────────────────────────────────────────────────────

def _display_prediction_result(result: dict) -> None:
	"""Render the full prediction result card (metrics, charts, explanation, insights)."""
	st.markdown("### Prediction result")
	top_row_col1, top_row_col2, top_row_col3 = st.columns(3)
	with top_row_col1:
		st.metric("Sentiment", result["label"])
	with top_row_col2:
		st.metric("Confidence", f"{float(result['confidence']):.0%}")
	with top_row_col3:
		st.metric("Cuisine / category", str(result["input_cuisine"]))

	bottom_row_col1, bottom_row_col2 = st.columns(2)
	with bottom_row_col1:
		st.metric("Predicted rating", f"{float(result['predicted_rating']):.2f}")
	with bottom_row_col2:
		st.metric("Predicted cuisine / category", str(result["predicted_category"]))
	st.caption(f"Inference source: {result['image_source']}")

	probability_df = pd.DataFrame({
		"Sentiment": ["Positive", "Negative"],
		"Probability": [float(result["positive_probability"]), float(result["negative_probability"])],
	})
	focus_df = pd.DataFrame({
		"Signal": ["Text", "Image"],
		"Focus": [float(result["text_focus"]), float(result["image_focus"])],
	})

	left, right = st.columns(2)
	with left:
		fig = px.bar(
			probability_df,
			x="Sentiment",
			y="Probability",
			color="Sentiment",
			text="Probability",
			title="Sentiment probabilities",
		)
		fig.update_traces(texttemplate="%{text:.0%}", textposition="outside")
		fig.update_yaxes(tickformat=".0%", range=[0, 1])
		fig.update_layout(showlegend=False)
		st.plotly_chart(fig, use_container_width=True, key="predict_prob_bar")
	with right:
		fig = px.bar(
			focus_df,
			x="Signal",
			y="Focus",
			color="Signal",
			text="Focus",
			title="Attention — which signal drove the decision?",
		)
		fig.update_traces(texttemplate="%{text:.0%}", textposition="outside")
		fig.update_yaxes(tickformat=".0%", range=[0, 1])
		fig.update_layout(showlegend=False)
		st.plotly_chart(fig, use_container_width=True, key="predict_focus_bar")

	st.markdown("### Explanation")
	st.info(result["attention_explanation"])

	explain_left, explain_right = st.columns(2)
	with explain_left:
		st.markdown("**Text signal**")
		st.write(result["text_result"]["summary"])
		positive_hits = result["text_result"]["positive_hits"]
		negative_hits = result["text_result"]["negative_hits"]
		st.caption(
			"Positive cues: " + (", ".join(positive_hits) if positive_hits else "none") +
			" | Negative cues: " + (", ".join(negative_hits) if negative_hits else "none")
		)
	with explain_right:
		st.markdown("**Visual signal**")
		st.write(result["image_result"]["summary"])
		if result["image_result"].get("brightness") is not None:
			st.caption(
				f"Brightness {float(result['image_result']['brightness']):.0%} | "
				f"Contrast {float(result['image_result']['contrast']):.0%} | "
				f"Warmth {float(result['image_result']['warmth']):.0%}"
			)

	st.markdown("### Business insight")
	bi_left, bi_right = st.columns(2)
	with bi_left:
		st.success(result["region_insight"])
	with bi_right:
		st.warning(result["business_insight"])


def render_predict_sentiment() -> None:
	reference_data = _get_demo_reference_data()
	sample_image_path = _load_sample_demo_image(reference_data)

	st.header("Predict Sentiment")
	st.caption(
		"Runs the full Image + Text + Region checkpoint on your saved inputs and "
		"explains what drove the decision — image attention or text attention."
	)

	review_text = st.session_state.get("input_review_text", "")
	region = st.session_state.get("input_region", "Nashville_TN")
	cuisine = st.session_state.get("input_cuisine", "American (New)")
	use_sample = st.session_state.get("input_use_sample", True)
	image_bytes = st.session_state.get("input_image_bytes")
	current_signature = {
		"review_text": review_text,
		"region": region,
		"cuisine": cuisine,
		"has_image": bool(image_bytes),
		"use_sample": use_sample,
	}
	current_signature = {
		"review_text": review_text,
		"region": region,
		"cuisine": cuisine,
		"has_image": bool(image_bytes),
		"use_sample": use_sample,
	}

	# ── current inputs summary ─────────────────────────────────────────────
	st.markdown("**Current inputs (from Upload & Review tab):**")
	inp1, inp2, inp3 = st.columns(3)
	with inp1:
		st.markdown(f"**Region:** {region}  \n**Cuisine / category:** {cuisine}")
	with inp2:
		if review_text:
			snippet = review_text[:130] + ("…" if len(review_text) > 130 else "")
			st.caption(snippet)
		else:
			st.caption("No review text saved.")
	with inp3:
		if image_bytes:
			st.image(image_bytes, width=140, caption="Input image")
		elif use_sample and sample_image_path:
			st.image(sample_image_path, width=140, caption="Sample image")
		else:
			st.caption("No image.")

	st.caption("To change inputs, switch to the Upload & Review tab.")

	if not review_text:
		st.warning("No review text found. Go to the **Upload & Review** tab and enter a review first.")
		return

	if not image_bytes and not (use_sample and sample_image_path):
		st.warning("No image available. Go to the **Upload & Review** tab and upload an image or enable the sample image.")
		return

	run_pred = st.button("Run sentiment prediction", type="primary", key="run_predict_btn")

	# Persist and show last result
	if (
		not run_pred
		and "last_prediction_result" in st.session_state
		and st.session_state.get("last_prediction_signature") == current_signature
	):
		st.divider()
		st.caption("Showing last prediction result. Click the button above to re-run.")
		_display_prediction_result(st.session_state["last_prediction_result"])
		return

	if run_pred:
		image_file = io.BytesIO(image_bytes) if image_bytes else None
		with st.spinner("Running checkpoint inference…"):
			try:
				result = _estimate_demo_prediction(
					review_text, image_file, region, cuisine,
					reference_data, sample_image_path, use_sample,
				)
			except RuntimeError as exc:
				st.error(str(exc))
				return
			except Exception as exc:
				st.error(
					"Prediction failed due to an unexpected runtime error. "
					f"Details: {exc}"
				)
				return
		st.session_state["last_prediction_result"] = result
		st.session_state["last_prediction_signature"] = current_signature
		st.divider()
		_display_prediction_result(result)


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Compare Models
# ──────────────────────────────────────────────────────────────────────────────

def _display_comparison_result(comparison: dict) -> None:
	"""Render the 4-model comparison table and bar chart."""
	comparison_df = pd.DataFrame(comparison["rows"])
	comparison_df["Confidence %"] = comparison_df["Confidence"].map(lambda v: f"{v:.0%}")
	st.markdown("### Comparison table")
	st.dataframe(
		comparison_df[["Model", "Prediction", "Confidence %"]],
		hide_index=True,
		use_container_width=True,
	)

	fig = px.bar(
		pd.DataFrame(comparison["rows"]),
		x="Model",
		y="Confidence",
		color="Prediction",
		text="Confidence",
		title="Confidence by model variant",
		color_discrete_map={"Positive": "#2a9d5c", "Negative": "#e05c2c"},
	)
	fig.update_traces(texttemplate="%{text:.0%}", textposition="outside")
	fig.update_yaxes(tickformat=".0%", range=[0, 1.1])
	st.plotly_chart(fig, use_container_width=True, key="compare_confidence_bar")

	st.caption(
		f"Image source: {comparison['image_source']} | "
		f"Predicted rating: {comparison['predicted_rating']:.2f} | "
		f"Predicted cuisine / category: {comparison['predicted_category']}"
	)

	predictions = {row["Model"]: row["Prediction"] for row in comparison["rows"]}
	image_pred = predictions.get("Image Only", "—")
	text_pred = predictions.get("Text Only", "—")
	full_pred = predictions.get("Image + Text + Region", "—")

	st.markdown("### What this shows")
	if image_pred != text_pred:
		st.info(
			f"Image-only predicts **{image_pred}** while text-only predicts **{text_pred}**. "
			"This is a perception gap case — a core research finding."
		)
	if full_pred != image_pred or full_pred != text_pred:
		st.success(
			f"The full model (Image + Text + Region) predicts **{full_pred}** — "
			"incorporating all signals resolves the modality conflict."
		)
	else:
		st.success(
			"All four models agree. Strong alignment between visual and textual signals for this input."
		)


def render_compare_models() -> None:
	reference_data = _get_demo_reference_data()
	sample_image_path = _load_sample_demo_image(reference_data)

	st.header("Compare Model Variants")
	st.caption(
		"Run all four ablation checkpoints on your saved inputs side by side. "
		"This directly demonstrates the research question: does adding region context help?"
	)

	review_text = st.session_state.get("input_review_text", "")
	region = st.session_state.get("input_region", "Nashville_TN")
	cuisine = st.session_state.get("input_cuisine", "American (New)")
	use_sample = st.session_state.get("input_use_sample", True)
	image_bytes = st.session_state.get("input_image_bytes")
	current_signature = {
		"review_text": review_text,
		"region": region,
		"cuisine": cuisine,
		"has_image": bool(image_bytes),
		"use_sample": use_sample,
	}

	st.markdown("**Using inputs from Upload & Review tab:**")
	inp1, inp2, inp3 = st.columns(3)
	with inp1:
		st.markdown(f"**Region:** {region}  \n**Cuisine / category:** {cuisine}")
	with inp2:
		if review_text:
			st.caption(review_text[:120] + ("…" if len(review_text) > 120 else ""))
		else:
			st.caption("No review text saved.")
	with inp3:
		if image_bytes:
			st.image(image_bytes, width=130, caption="Input image")
		elif use_sample and sample_image_path:
			st.image(sample_image_path, width=130, caption="Sample image")
		else:
			st.caption("No image.")

	if not review_text:
		st.warning("No review text found. Go to the **Upload & Review** tab first.")
		return
	if not image_bytes and not (use_sample and sample_image_path):
		st.warning("No image available. Go to the **Upload & Review** tab first.")
		return

	run_compare = st.button("Run all 4 checkpoints", type="primary", key="run_compare_btn")

	if (
		not run_compare
		and "last_comparison_result" in st.session_state
		and st.session_state.get("last_comparison_signature") == current_signature
	):
		st.divider()
		st.caption("Showing last comparison result. Click the button above to re-run.")
		_display_comparison_result(st.session_state["last_comparison_result"])
		return

	if run_compare:
		image_file = io.BytesIO(image_bytes) if image_bytes else None
		with st.spinner("Running all four checkpoints…"):
			try:
				comparison = _run_multimodal_comparison(
					review_text=review_text,
					uploaded_image=image_file,
					region=region,
					cuisine=cuisine,
					sample_image_path=sample_image_path,
					use_sample_image=use_sample,
				)
			except RuntimeError as exc:
				st.error(str(exc))
				return
		st.session_state["last_comparison_result"] = comparison
		st.session_state["last_comparison_signature"] = current_signature
		st.divider()
		_display_comparison_result(comparison)


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 7 — Results & Limitations
# ──────────────────────────────────────────────────────────────────────────────

def render_results_and_limitations() -> None:
	st.header("Results & Limitations")
	st.caption(
		"Full ablation table, key research findings, and an honest discussion of "
		"where the model falls short."
	)

	ablation_df = load_csv(OUTPUT_DIR / "ablation_results.csv")

	st.markdown("### Ablation results")
	if ablation_df is not None and not ablation_df.empty:
		ablation = _coerce_numeric_columns(ablation_df, ["F1", "Accuracy", "Precision", "Recall"])
		st.dataframe(ablation, hide_index=True, use_container_width=True)

		if {"Model", "F1", "Accuracy"}.issubset(ablation.columns):
			fig = px.bar(
				ablation.sort_values("F1"),
				x="Model",
				y="F1",
				text="F1",
				color="F1",
				title="F1 Score by model variant",
				color_continuous_scale="RdYlGn",
			)
			fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
			fig.update_yaxes(range=[0, 1.08])
			st.plotly_chart(fig, use_container_width=True, key="results_f1_bar")

			best_row = ablation.sort_values("F1", ascending=False).iloc[0]
			it_rows = ablation[
				ablation["Model"].str.contains("Text", na=False, case=False) &
				~ablation["Model"].str.contains("Region", na=False, case=False) &
				ablation["Model"].str.contains("Image", na=False, case=False)
			]
			full_rows = ablation[ablation["Model"].str.contains("Region", na=False, case=False)]
			img_only = ablation[ablation["Model"].str.strip() == "Image Only"]
			txt_only = ablation[ablation["Model"].str.strip() == "Text Only"]

			st.markdown("### Key findings")
			findings: list[str] = []
			findings.append(
				f"**Best model:** {best_row['Model']} with F1 = {float(best_row['F1']):.4f} "
				f"and Accuracy = {float(best_row['Accuracy']):.4f}."
			)
			if not it_rows.empty and not full_rows.empty:
				f1_gain = float(full_rows.iloc[0]["F1"]) - float(it_rows.iloc[0]["F1"])
				acc_gain = float(full_rows.iloc[0]["Accuracy"]) - float(it_rows.iloc[0]["Accuracy"])
				findings.append(
					f"**Region context adds value:** Adding region to Image + Text improves "
					f"F1 by {f1_gain:+.4f} and Accuracy by {acc_gain:+.4f}."
				)
			if not img_only.empty and not txt_only.empty:
				if float(txt_only.iloc[0]["F1"]) > float(img_only.iloc[0]["F1"]):
					findings.append(
						"**Text dominates image:** Text-only outperforms image-only, indicating "
						"review text carries the stronger individual sentiment signal in this dataset."
					)
				else:
					findings.append(
						"**Visual signal is strong:** Image-only matches or outperforms text-only in this dataset."
					)
			for finding in findings:
				st.markdown(f"- {finding}")
	else:
		st.info("Ablation results not found. Run yelp.py to generate outputs/ablation_results.csv.")

	st.markdown("### Limitations")
	for lim in [
		"**Dataset scope:** Trained on the Yelp academic dataset — results may not generalize to all review platforms or international markets.",
		"**Image availability:** Many Yelp businesses lack photo submissions. The model defaults to text-only fallback when no image is available.",
		"**Region granularity:** Region is encoded at the city level. Neighborhood or zip-code context may improve accuracy further.",
		"**Static checkpoints:** The deployed checkpoints are snapshots. New food trends, cities, and evolving language require periodic retraining.",
		"**Binary sentiment:** The model outputs positive/negative sentiment. Nuanced aspects (service vs. food vs. ambiance) are collapsed.",
		"**Perception gap coverage:** The mismatch detector was built on a subset of the dataset and may undercount gaps in specific niches.",
	]:
		st.markdown(f"- {lim}")


# ──────────────────────────────────────────────────────────────────────────────
# Main — 7-tab demo flow
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
	apply_app_theme()
	sidebar_controls()

	st.title("Yelp Food Intelligence")
	st.caption(
		"A multimodal sentiment system trained on Yelp reviews and food photos. "
		"No training happens here — the app loads pre-trained checkpoints only."
	)

	(
		home_tab,
		upload_tab,
		predict_tab,
		compare_tab,
		region_tab,
		gap_tab,
		results_tab,
	) = st.tabs([
		"Home",
		"Upload & Review",
		"Predict Sentiment",
		"Compare Models",
		"Region & Cuisine",
		"Perception Gaps",
		"Results & Limitations",
	])

	with home_tab:
		render_home()
	with upload_tab:
		render_upload_review()
	with predict_tab:
		render_predict_sentiment()
	with compare_tab:
		render_compare_models()
	with region_tab:
		render_region_impact_demo()
	with gap_tab:
		render_perception_gap_explorer()
	with results_tab:
		render_results_and_limitations()


if __name__ == "__main__":
	main()
