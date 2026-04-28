from pathlib import Path
import io
import os
import re
import sys
import shutil
import tarfile
import urllib.request
import zipfile

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image, ImageStat


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"

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


def render_table_or_message(df: pd.DataFrame | None, message: str) -> bool:
	if df is None or df.empty:
		st.info(message)
		return False
	st.dataframe(df, width="stretch")
	return True


def apply_app_theme() -> None:
	st.markdown(APP_CSS, unsafe_allow_html=True)


def render_landing_page() -> None:
	reference_data = _get_demo_reference_data()
	ablation_df = load_csv(OUTPUT_DIR / "ablation_results.csv")
	mismatch_df = reference_data.get("mismatch")
	region_gap_df = reference_data.get("region_gap")

	best_model = "Checkpoint not available"
	best_f1 = None
	if ablation_df is not None and not ablation_df.empty and {"Model", "F1"}.issubset(ablation_df.columns):
		ablation = _coerce_numeric_columns(ablation_df, ["F1", "Accuracy"])
		best_row = ablation.sort_values("F1", ascending=False).iloc[0]
		best_model = str(best_row["Model"])
		best_f1 = float(best_row["F1"])

	top_region_text = "Regional hotspot unavailable"
	if region_gap_df is not None and not region_gap_df.empty and {"region", "mismatch_rate"}.issubset(region_gap_df.columns):
		gap_df = _coerce_numeric_columns(region_gap_df, ["mismatch_rate", "samples"])
		top_region = gap_df.sort_values("mismatch_rate", ascending=False).iloc[0]
		top_region_text = f"{top_region['region']} gap {float(top_region['mismatch_rate']):.1%}"

	perception_gap = None
	if mismatch_df is not None and not mismatch_df.empty and "mismatch" in mismatch_df.columns:
		perception_gap = float(pd.to_numeric(mismatch_df["mismatch"], errors="coerce").mean())

	st.markdown(
		f"""
		<div class=\"yfi-hero\">
			<div class=\"yfi-section-label\">Presentation-ready demo</div>
			<h1>Yelp Food Intelligence</h1>
			<p>
				A real-world multimodal sentiment workflow for restaurants, delivery platforms, and growth teams.
				Upload a food image, pair it with a Yelp review and market, then explain the prediction with region-aware model evidence.
			</p>
			<div class=\"yfi-badge-row\">
				<div class=\"yfi-badge\">Live checkpoint inference</div>
				<div class=\"yfi-badge\">Attention-based explanation</div>
				<div class=\"yfi-badge\">Market and cuisine context</div>
			</div>
		</div>
		""",
		unsafe_allow_html=True,
	)

	col1, col2, col3 = st.columns(3)
	with col1:
		st.markdown(
			"""
			<div class=\"yfi-panel\">
				<h3>Live prediction flow</h3>
				<p>Step through the presentation story with image upload, review text, region, cuisine, and a real model prediction.</p>
			</div>
			""",
			unsafe_allow_html=True,
		)
	with col2:
		st.markdown(
			"""
			<div class=\"yfi-panel\">
				<h3>Transparent explanation</h3>
				<p>Show whether the decision leaned more on visual evidence or textual evidence instead of treating the system like a black box.</p>
			</div>
			""",
			unsafe_allow_html=True,
		)
	with col3:
		st.markdown(
			"""
			<div class=\"yfi-panel\">
				<h3>Business impact</h3>
				<p>Convert the prediction into actions for operators, recommendation teams, and regional marketing strategy.</p>
			</div>
			""",
			unsafe_allow_html=True,
		)

	metric1, metric2, metric3 = st.columns(3)
	with metric1:
		st.metric("Best checkpoint", best_model, delta=f"F1 {best_f1:.4f}" if best_f1 is not None else None)
	with metric2:
		st.metric("Perception gap", f"{perception_gap:.1%}" if perception_gap is not None else "N/A")
	with metric3:
		st.metric("Top mismatch market", top_region_text)

	st.markdown("### Demo flow")
	st.markdown("1. Show the input: food image, Yelp review text, region, and cuisine.")
	st.markdown("2. Run the checkpointed multimodal model and display sentiment plus confidence.")
	st.markdown("3. Explain whether the model relied more on image or text attention.")
	st.markdown("4. Translate the result into a restaurant, delivery, or marketing action.")


def _coerce_numeric_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
	clean = df.copy()
	for column in columns:
		if column in clean.columns:
			clean[column] = pd.to_numeric(clean[column], errors="coerce")
	return clean


def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
	return max(lower, min(upper, value))


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
	region_stats = reference_data.get("region_stats")
	regions: list[str] = []
	if region_stats is not None and not region_stats.empty and "region" in region_stats.columns:
		regions = sorted(region_stats["region"].dropna().astype(str).unique().tolist())
	for scenario in DEMO_SCENARIOS.values():
		if scenario["region"] not in regions:
			regions.append(scenario["region"])
	return sorted(regions)


def _get_demo_cuisines(reference_data: dict[str, pd.DataFrame | None]) -> list[str]:
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
	if retrieval_df is None or retrieval_df.empty or "image_path" not in retrieval_df.columns:
		return None
	image_path = Path(str(retrieval_df.iloc[0]["image_path"]))
	return str(image_path) if image_path.exists() else None


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
		business_df, review_df, photo_df = yelp_module.load_yelp_data(
			yelp_module.BUSINESS_PATH,
			yelp_module.REVIEW_PATH,
			yelp_module.PHOTO_META_PATH,
			max_reviews=yelp_module.MAX_REVIEWS,
			max_photos=yelp_module.MAX_PHOTOS,
		)
		df = yelp_module.prepare_multimodal_dataframe(
			business_df=business_df,
			review_df=review_df,
			photo_df=photo_df,
			photo_dir=yelp_module.PHOTO_DIR,
		)
		df = df.sample(min(len(df), 30000), random_state=yelp_module.SEED).reset_index(drop=True)
		train_df, _ = yelp_module.train_test_split(
			df,
			test_size=0.2,
			random_state=yelp_module.SEED,
			stratify=df["sentiment"],
		)
		train_df = train_df.copy()
		train_df["primary_category"] = train_df["categories"].apply(yelp_module.extract_primary_category)

		tokenizer = yelp_module.fit_tokenizer(train_df["review_text"].tolist(), vocab_size=yelp_module.VOCAB_SIZE)
		region_to_id = {region: idx for idx, region in enumerate(sorted(train_df["region"].unique()))}
		unknown_region_id = len(region_to_id)
		top_categories = train_df["primary_category"].value_counts().head(yelp_module.MAX_CATEGORY_CLASSES).index.tolist()
		category_to_id = {cat: idx for idx, cat in enumerate(top_categories)}
		unknown_category_id = len(category_to_id)
		category_to_id["<unknown_category>"] = unknown_category_id
		id_to_category = {idx: cat for cat, idx in category_to_id.items()}

		image_transform = transforms.Compose([
			transforms.Resize((yelp_module.IMAGE_SIZE, yelp_module.IMAGE_SIZE)),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		])

		device = torch.device("cpu")
		model = yelp_module.MultimodalFusionModel(
			yelp_module.VOCAB_SIZE,
			unknown_region_id + 1,
			len(category_to_id),
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
		return {"spread": None, "region_count": None}
	match = cuisine_spread[cuisine_spread["cuisine_cluster"] == cuisine]
	if match.empty:
		return {
			"spread": float(pd.to_numeric(cuisine_spread["spread"], errors="coerce").median()),
			"region_count": float(pd.to_numeric(cuisine_spread["region_count"], errors="coerce").median()),
		}
	row = match.iloc[0]
	return {
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

		image_model = yelp_module.ImageOnlyModel(use_pretrained=False)
		image_model.load_state_dict(torch.load(checkpoint_map["Image Only"], map_location=device))
		image_model = image_model.to(device)
		image_model.eval()

		text_model = yelp_module.TextOnlyModel(yelp_module.VOCAB_SIZE, yelp_module.EMBED_DIM, yelp_module.LSTM_HIDDEN)
		text_model.load_state_dict(torch.load(checkpoint_map["Text Only"], map_location=device))
		text_model = text_model.to(device)
		text_model.eval()

		image_text_model = yelp_module.ImageTextFusionModel(yelp_module.VOCAB_SIZE, use_pretrained=False)
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


def render_live_prediction_demo() -> None:
	st.header("Food Sentiment Predictor Demo")
	st.caption("Upload a food image, add review text, choose region and cuisine, then walk through the same real-world demo flow you would use in a presentation.")

	reference_data = _get_demo_reference_data()
	regions = _get_demo_regions(reference_data)
	cuisines = _get_demo_cuisines(reference_data)
	sample_image_path = _load_sample_demo_image(reference_data)

	if "demo_review_text" not in st.session_state:
		st.session_state["demo_review_text"] = DEMO_SCENARIOS["Perception gap demo - Nashville"]["review"]
	if "demo_region" not in st.session_state:
		st.session_state["demo_region"] = DEMO_SCENARIOS["Perception gap demo - Nashville"]["region"]
	if "demo_cuisine" not in st.session_state:
		st.session_state["demo_cuisine"] = DEMO_SCENARIOS["Perception gap demo - Nashville"]["cuisine"]

	step1, step2 = st.columns((1.1, 0.9))
	with step1:
		st.markdown("### Step 1: Show input")
		st.write("Here, a restaurant or food delivery platform can upload a food image, review text, and location.")
		scenario_name = st.selectbox("Demo scenario", list(DEMO_SCENARIOS.keys()), index=1)
		if st.button("Load selected example", width="stretch"):
			scenario = DEMO_SCENARIOS[scenario_name]
			st.session_state["demo_review_text"] = scenario["review"]
			st.session_state["demo_region"] = scenario["region"]
			st.session_state["demo_cuisine"] = scenario["cuisine"]

		with st.form("live_prediction_form"):
			uploaded_image = st.file_uploader("Food image", type=["png", "jpg", "jpeg"])
			use_sample_image = st.checkbox("Use sample dataset image when no upload is provided", value=bool(sample_image_path))
			review_text = st.text_area("Yelp review text", key="demo_review_text", height=150, placeholder="Paste a review here...")
			region = st.selectbox(
				"Region / city",
				regions,
				index=regions.index(st.session_state["demo_region"]) if st.session_state["demo_region"] in regions else 0,
				key="demo_region",
			)
			cuisine = st.selectbox(
				"Cuisine",
				cuisines,
				index=cuisines.index(st.session_state["demo_cuisine"]) if st.session_state["demo_cuisine"] in cuisines else 0,
				key="demo_cuisine",
			)
			run_demo = st.form_submit_button("Run sentiment analysis", width="stretch")

		st.caption(DEMO_SCENARIOS[scenario_name]["insight"])

	with step2:
		st.markdown("### Input preview")
		if uploaded_image is not None:
			st.image(uploaded_image, caption="Uploaded food image", width="stretch")
		elif use_sample_image and sample_image_path:
			st.image(sample_image_path, caption="Sample food image available in the dataset", width="stretch")
		else:
			st.info("Upload a food photo, or use the sample dataset image for the live demo visual.")
		st.markdown("**Presentation flow**")
		st.markdown("1. Show the input a business user would provide.")
		st.markdown("2. Run a multimodal sentiment prediction.")
		st.markdown("3. Explain whether text or image drove the result.")
		st.markdown("4. Turn the result into an operational recommendation.")

	if run_demo:
		with st.spinner("Running checkpoint inference..."):
			try:
				result = _estimate_demo_prediction(review_text, uploaded_image, region, cuisine, reference_data, sample_image_path, use_sample_image)
			except RuntimeError as exc:
				st.error(str(exc))
				return
		st.divider()
		st.markdown("### Step 2: Run prediction")
		st.write("The model combines visual, textual, and regional signals to predict sentiment.")
		m1, m2, m3, m4 = st.columns(4)
		with m1:
			st.metric("Predicted sentiment", result["label"])
		with m2:
			st.metric("Confidence", f"{float(result['confidence']):.0%}")
		with m3:
			st.metric("Predicted rating estimate", f"{float(result['predicted_rating']):.2f}")
		with m4:
			st.metric("Predicted cuisine/category", str(result["predicted_category"]))
		st.caption(f"Inference source: {result['image_source']}")

		probability_df = pd.DataFrame(
			{
				"Sentiment": ["Positive", "Negative"],
				"Probability": [float(result["positive_probability"]), float(result["negative_probability"])],
			}
		)
		focus_df = pd.DataFrame(
			{
				"Signal": ["Text", "Image"],
				"Focus": [float(result["text_focus"]), float(result["image_focus"])],
			}
		)
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
			st.plotly_chart(fig, width="stretch", key="demo_probability_bar")
		with right:
			fig = px.bar(
				focus_df,
				x="Signal",
				y="Focus",
				color="Signal",
				text="Focus",
				title="Attention explanation",
			)
			fig.update_traces(texttemplate="%{text:.0%}", textposition="outside")
			fig.update_yaxes(tickformat=".0%", range=[0, 1])
			fig.update_layout(showlegend=False)
			st.plotly_chart(fig, width="stretch", key="demo_focus_bar")

		st.markdown("### Step 3: Show explanation")
		st.info("The model used image + review text + region context.")
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
					f"Brightness {float(result['image_result']['brightness']):.0%} | Contrast {float(result['image_result']['contrast']):.0%} | Warmth {float(result['image_result']['warmth']):.0%}"
				)

		st.markdown("### Step 4: Show business insight")
		insight_left, insight_right = st.columns(2)
		with insight_left:
			st.success(result["region_insight"])
		with insight_right:
			st.warning(result["business_insight"])


def render_multimodal_comparison_demo() -> None:
	st.header("Multimodal Comparison Demo")
	st.caption("Compare all four ablation variants side by side on the same user input.")

	reference_data = _get_demo_reference_data()
	regions = _get_demo_regions(reference_data)
	cuisines = _get_demo_cuisines(reference_data)
	sample_image_path = _load_sample_demo_image(reference_data)

	if "comparison_review_text" not in st.session_state:
		st.session_state["comparison_review_text"] = DEMO_SCENARIOS["Perception gap demo - Nashville"]["review"]
	if "comparison_region" not in st.session_state:
		st.session_state["comparison_region"] = DEMO_SCENARIOS["Perception gap demo - Nashville"]["region"]
	if "comparison_cuisine" not in st.session_state:
		st.session_state["comparison_cuisine"] = DEMO_SCENARIOS["Perception gap demo - Nashville"]["cuisine"]

	left, right = st.columns((1.1, 0.9))
	with left:
		with st.form("multimodal_comparison_form"):
			uploaded_image = st.file_uploader("Food image", type=["png", "jpg", "jpeg"], key="comparison_uploader")
			use_sample_image = st.checkbox("Use sample dataset image when no upload is provided", value=bool(sample_image_path), key="comparison_use_sample")
			review_text = st.text_area("Review text", key="comparison_review_text", height=140)
			region = st.selectbox("Region / city", regions, key="comparison_region")
			cuisine = st.selectbox("Cuisine", cuisines, key="comparison_cuisine")
			run_compare = st.form_submit_button("Run multimodal comparison", width="stretch")

	with right:
		st.markdown("**Expected comparison view**")
		st.markdown("- Image-only model")
		st.markdown("- Text-only model")
		st.markdown("- Image + Text model")
		st.markdown("- Image + Text + Region model")
		st.caption("This directly matches your ablation research question.")

	if run_compare:
		with st.spinner("Running all four checkpoints..."):
			try:
				comparison = _run_multimodal_comparison(
					review_text=review_text,
					uploaded_image=uploaded_image,
					region=region,
					cuisine=cuisine,
					sample_image_path=sample_image_path,
					use_sample_image=use_sample_image,
				)
			except RuntimeError as exc:
				st.error(str(exc))
				return

		comparison_df = pd.DataFrame(comparison["rows"])
		comparison_df["Confidence"] = comparison_df["Confidence"].map(lambda value: f"{value:.0%}")
		st.dataframe(comparison_df, width="stretch")
		st.caption(
			f"Image source: {comparison['image_source']} | Predicted rating estimate: {comparison['predicted_rating']:.2f} | "
			f"Predicted cuisine/category: {comparison['predicted_category']}"
		)


def render_region_impact_demo() -> None:
	st.header("Region Impact Demo")
	st.caption("Explore how sentiment changes by location and why region context improves model behavior.")

	reference_data = _get_demo_reference_data()
	regions_df = reference_data.get("region_stats")
	cuisine_spread_df = reference_data.get("cuisine_spread")
	mismatch_df = reference_data.get("mismatch")
	ablation_df = load_csv(OUTPUT_DIR / "ablation_results.csv")

	cuisines = _get_demo_cuisines(reference_data)
	regions = _get_demo_regions(reference_data)
	selected_cuisine = st.selectbox("Cuisine", cuisines, key="region_impact_cuisine")
	selected_region = st.selectbox("Region", regions, key="region_impact_region")

	metric_col1, metric_col2, metric_col3 = st.columns(3)
	with metric_col1:
		if regions_df is not None and not regions_df.empty and {"region", "positive_rate"}.issubset(regions_df.columns):
			region_row = regions_df[regions_df["region"] == selected_region]
			if not region_row.empty:
				st.metric("Positive sentiment rate (selected region)", f"{float(region_row.iloc[0]['positive_rate']):.1%}")
			else:
				st.metric("Positive sentiment rate (selected region)", "N/A")
		else:
			st.metric("Positive sentiment rate (selected region)", "N/A")

	with metric_col2:
		if cuisine_spread_df is not None and not cuisine_spread_df.empty:
			match = cuisine_spread_df[cuisine_spread_df["cuisine_cluster"] == selected_cuisine]
			if not match.empty:
				st.metric("Region sentiment spread", f"{float(match.iloc[0]['spread']):.2f}")
			else:
				st.metric("Region sentiment spread", "N/A")
		else:
			st.metric("Region sentiment spread", "N/A")

	with metric_col3:
		if ablation_df is not None and not ablation_df.empty and {"Model", "F1"}.issubset(ablation_df.columns):
			ablation = _coerce_numeric_columns(ablation_df, ["F1"])
			row_it = ablation[ablation["Model"] == "Image + Text"]
			row_full = ablation[ablation["Model"] == "Image + Text + Region"]
			if not row_it.empty and not row_full.empty:
				delta = float(row_full.iloc[0]["F1"] - row_it.iloc[0]["F1"])
				st.metric("Region impact on F1", f"{delta:+.4f}")
			else:
				st.metric("Region impact on F1", "N/A")
		else:
			st.metric("Region impact on F1", "N/A")

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
				title="Positive sentiment rate by region",
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
					title=f"Top regions for {selected_cuisine} (text sentiment proxy)",
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
	st.caption("Inspect cases where image and text sentiment disagree, especially image-positive vs text-negative mismatches.")

	mismatch_df = load_csv(OUTPUT_DIR / "image_text_consistency_predictions.csv")
	if mismatch_df is None or mismatch_df.empty:
		st.info("No perception-gap outputs found. Run yelp.py to generate image-text consistency files.")
		return

	mismatch_df = _coerce_numeric_columns(
		mismatch_df,
		["image_prob", "text_prob", "mismatch", "img_pos_txt_neg", "img_neg_txt_pos"],
	)

	mode = st.radio(
		"Gap mode",
		["Image positive / Text negative", "Image negative / Text positive", "All mismatches"],
		horizontal=True,
	)
	regions = ["All"] + sorted(mismatch_df["region"].dropna().astype(str).unique().tolist())
	cuisines = ["All"] + sorted(mismatch_df["cuisine"].dropna().astype(str).unique().tolist())
	selected_region = st.selectbox("Region filter", regions, key="gap_region_filter")
	selected_cuisine = st.selectbox("Cuisine filter", cuisines, key="gap_cuisine_filter")

	filtered = mismatch_df.copy()
	if mode == "Image positive / Text negative":
		filtered = filtered[filtered["img_pos_txt_neg"] == 1]
	elif mode == "Image negative / Text positive":
		filtered = filtered[filtered["img_neg_txt_pos"] == 1]
	else:
		filtered = filtered[filtered["mismatch"] == 1]

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

	if filtered.empty:
		st.warning("No rows match the selected filters.")
		return

	scatter_df = filtered.head(800)
	fig = px.scatter(
		scatter_df,
		x="image_prob",
		y="text_prob",
		color="region",
		hover_data=["cuisine", "label"],
		title="Perception gap cases (image probability vs text probability)",
	)
	fig.add_hline(y=0.5, line_dash="dot")
	fig.add_vline(x=0.5, line_dash="dot")
	st.plotly_chart(fig, width="stretch", key="gap_scatter")

	st.markdown("**Examples**")
	st.dataframe(
		filtered[["region", "cuisine", "label", "image_prob", "text_prob", "image_pred", "text_pred"]].head(120),
		width="stretch",
	)

	st.info("Perception gap means the image looks appealing but the review tone is negative (or vice versa), which is exactly where multimodal reasoning matters.")


def render_example_use_cases() -> None:
	st.header("Example Use Cases")
	st.caption("Practical ways to position the model for restaurant operators, delivery platforms, and growth teams.")

	use_cases = [
		(
			"Restaurant Owner",
			"Understand customer perception",
			[
				"Spot dishes that look appealing but generate negative written feedback.",
				"Identify cuisines or locations with unusually high perception gap.",
				"Decide whether to fix plating, food quality, or service messaging first.",
			],
		),
		(
			"Food Delivery App",
			"Support better recommendations",
			[
				"Mix photo quality, review tone, and local market context in ranking logic.",
				"Personalize recommendations by region-specific customer taste patterns.",
				"Catch visually strong items whose text reviews signal disappointment.",
			],
		),
		(
			"Marketing Team",
			"Find cuisine trends",
			[
				"Compare cuisines with high regional volatility before launching promotions.",
				"Choose markets where hero imagery aligns with strong customer sentiment.",
				"Use explainability output to tell a more concrete story than a black-box score.",
			],
		),
	]

	cols = st.columns(3)
	for col, (title, subtitle, bullets) in zip(cols, use_cases):
		with col:
			with st.container(border=True):
				st.subheader(title)
				st.caption(subtitle)
				for bullet in bullets:
					st.markdown(f"- {bullet}")

	st.divider()
	st.subheader("Slide Content: Real-World Application")
	st.markdown("**Input**: food image + review + region")
	st.markdown("**Model**: CNN + LSTM + region embedding")
	st.markdown("**Output**: sentiment + confidence + explanation")
	st.markdown("**Use case**: restaurant insights and personalization")
	st.caption(
		"Speaker note: This Streamlit demo shows how the model can be used in a real-world setting. A user provides a food image, review text, and region, and the system predicts sentiment while explaining whether the result was driven more by visual or textual evidence."
	)


def render_demo_explainability() -> None:
	st.header("Explainability")
	st.caption("This page makes the model more transparent than a black-box classifier by combining attention, perception-gap, and region/cuisine variability signals.")

	reference_data = _get_demo_reference_data()
	att_no_region = reference_data["attention_no_region"]
	att_region = reference_data["attention_region"]
	region_gap = reference_data["region_gap"]
	cuisine_spread = reference_data["cuisine_spread"]

	attention_rows = []
	if att_no_region is not None and not att_no_region.empty and {"img_concentration", "txt_concentration"}.issubset(att_no_region.columns):
		attention_rows.append(
			{
				"Model": "Image + Text",
				"Image focus": float(pd.to_numeric(att_no_region["img_concentration"], errors="coerce").mean()),
				"Text focus": float(pd.to_numeric(att_no_region["txt_concentration"], errors="coerce").mean()),
			}
		)
	if att_region is not None and not att_region.empty and {"img_concentration", "txt_concentration"}.issubset(att_region.columns):
		attention_rows.append(
			{
				"Model": "Image + Text + Region",
				"Image focus": float(pd.to_numeric(att_region["img_concentration"], errors="coerce").mean()),
				"Text focus": float(pd.to_numeric(att_region["txt_concentration"], errors="coerce").mean()),
			}
		)

	left, right = st.columns((1, 1))
	with left:
		if attention_rows:
			attention_df = pd.DataFrame(attention_rows).melt(
				id_vars="Model",
				value_vars=["Image focus", "Text focus"],
				var_name="Signal",
				value_name="Focus",
			)
			fig = px.bar(
				attention_df,
				x="Model",
				y="Focus",
				color="Signal",
				barmode="group",
				text="Focus",
				title="Attention focus chart",
			)
			fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
			fig.update_yaxes(tickformat=".0%", range=[0, 1])
			st.plotly_chart(fig, width="stretch", key="demo_explain_attention")
		else:
			st.info("No attention outputs available.")
	with right:
		if region_gap is not None and not region_gap.empty:
			region_gap = _coerce_numeric_columns(region_gap, ["samples", "mismatch_rate"])
			fig = px.bar(
				region_gap.sort_values("mismatch_rate", ascending=False).head(10),
				x="region",
				y="mismatch_rate",
				color="mismatch_rate",
				text="mismatch_rate",
				title="Perception gap by region",
			)
			fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
			fig.update_yaxes(tickformat=".0%")
			st.plotly_chart(fig, width="stretch", key="demo_explain_region_gap")
		else:
			st.info("No region-level perception gap summary available.")

	if cuisine_spread is not None and not cuisine_spread.empty:
		st.subheader("Region/cuisine spread")
		cuisine_spread = _coerce_numeric_columns(cuisine_spread, ["min_rate", "max_rate", "region_count", "spread"])
		fig = px.bar(
			cuisine_spread.sort_values("spread", ascending=False).head(12),
			x="cuisine_cluster",
			y="spread",
			color="region_count",
			text="spread",
			title="Cuisine variability across regions",
		)
		fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
		st.plotly_chart(fig, width="stretch", key="demo_explain_cuisine_spread")
		st.caption("This makes the model more transparent than a black-box classifier by tying predictions back to attention behavior, mismatch hotspots, and regional volatility.")


def missing_pipeline_inputs() -> list[Path]:
	return [path for path in REQUIRED_PIPELINE_INPUTS if not path.exists()]


def sidebar_controls() -> None:
	st.sidebar.title("Controls")
	st.sidebar.caption("Run the training/analysis pipeline or inspect saved outputs.")
	st.sidebar.markdown("**Research Question**")
	st.sidebar.write(RESEARCH_QUESTION)

	if st.sidebar.button("Prepare data from URLs", width="stretch"):
		with st.sidebar:
			with st.spinner("Preparing runtime data files..."):
				logs, errors = prepare_pipeline_inputs_from_urls()
				st.session_state["data_prepare_logs"] = "\n".join(logs) if logs else ""
				st.session_state["data_prepare_errors"] = "\n".join(errors) if errors else ""
				if errors:
					st.error("Data preparation completed with issues. See logs below.")
				else:
					st.success("Runtime data prepared successfully.")

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

	with st.sidebar.expander("Data bootstrap settings"):
		for key in DATA_URL_ENV_VARS:
			configured = bool(_resolve_setting(key))
			st.write(f"{'Configured' if configured else 'Not set'}: {key}")
		if st.session_state.get("data_prepare_logs"):
			st.text_area("Prepare logs", st.session_state["data_prepare_logs"], height=180)
		if st.session_state.get("data_prepare_errors"):
			st.text_area("Prepare errors", st.session_state["data_prepare_errors"], height=160)










def render_retrieval() -> None:
	st.header("Cross-Modal Retrieval Demo")
	text_to_image = load_csv(OUTPUT_DIR / "retrieval_text_to_image.csv")
	image_to_text = load_csv(OUTPUT_DIR / "retrieval_image_to_text.csv")

	left, right = st.columns(2)
	with left:
		st.subheader("Text -> Image")
		if render_table_or_message(text_to_image, "No text->image retrieval results found."):
			query = str(text_to_image.iloc[0]["query"])
			st.caption(f"Query: {query}")
			for _, row in text_to_image.head(5).iterrows():
				st.image(row["image_path"], width=260)
				st.write(f"Similarity: {row['similarity']:.4f}")
				st.write(row["review_text"])
				st.divider()

	with right:
		st.subheader("Image -> Text")
		if render_table_or_message(image_to_text, "No image->text retrieval results found."):
			st.image(image_to_text.iloc[0]["query_image_path"], width=260)
			for _, row in image_to_text.head(5).iterrows():
				st.write(f"Similarity: {row['similarity']:.4f} | Region: {row['region']} | Sentiment: {int(row['sentiment'])}")
				st.write(row["review_text"])
				st.divider()





def render_presentation_plots() -> None:
	st.header("Presentation Plots")
	st.caption("Static presentation-ready figures exported by yelp.py for slides, reports, and final writeups.")

	plot_specs = [
		(
			"Model F1 Comparison",
			OUTPUT_DIR / "final_model_f1_bar.png",
			"Result: Image + Text + Region is the best model (F1 = 0.9661, Accuracy = 0.9422). Adding region improves over Image + Text (F1: 0.9567 -> 0.9661, +0.0094; Accuracy: 0.9260 -> 0.9422, +0.0162), showing that geographic context provides a measurable performance lift.",
		),
		(
			"Best-Model Confusion Matrix",
			OUTPUT_DIR / "final_best_model_confusion_matrix.png",
			"Result: The best model still makes more false negatives than false positives, consistent with Recall (0.9560) being lower than Precision (0.9765). In practice, it is slightly more conservative when predicting positive sentiment, which keeps false alarms low but misses some true positives.",
		),
		(
			"Perception Gap by Region",
			OUTPUT_DIR / "final_perception_gap_by_region.png",
			"Result: Perception gap is clearly location-dependent, with mismatch rates ranging from 21.43% to 44.83% across regions in this run. High-gap regions (for example, St Petersburg_FL at 44.83%) indicate much stronger image-text disagreement than lower-gap regions, supporting the value of geographic context.",
		),
		(
			"Perception Gap by Cuisine",
			OUTPUT_DIR / "final_perception_gap_by_cuisine.png",
			"Result: Perception gap also varies strongly by cuisine, with mismatch rates ranging from 15.15% (Jazz & Blues) to 50.72% (Italian) in this run. This indicates some cuisine categories are substantially harder for image and text signals to agree on than others.",
		),
		(
			"Attention Comparison",
			OUTPUT_DIR / "final_attention_comparison.png",
			"Result: Adding region shifts modality reliance toward image features: image concentration rises from 0.2224 to 0.4177 (+0.1953), while text concentration decreases slightly from 0.1767 to 0.1655 (-0.0112). This shift is accompanied by higher accuracy (0.9260 to 0.9422), suggesting region-aware fusion helps the model use visual evidence more effectively.",
		),
		(
			"Region/Cuisine Spread",
			OUTPUT_DIR / "final_region_cuisine_spread.png",
			"Result: The same cuisine shows large regional sentiment variability. For example, Seafood has the widest spread (min 0.35, max 0.94; spread 0.59 across 7 regions), and even broad categories like Restaurants vary from 0.7143 to 1.0000 across 21 regions. This directly supports region-aware modeling.",
		),
	]

	available = [(title, path, talk_track) for title, path, talk_track in plot_specs if path.exists()]
	missing = [title for title, path, _ in plot_specs if not path.exists()]

	if not available:
		st.info("No final presentation plots found. Run yelp.py to generate the PNG exports.")
		return

	if missing:
		st.warning("Some presentation plots are still missing: " + ", ".join(missing))

	for idx in range(0, len(available), 2):
		cols = st.columns(2)
		for col, (title, path, talk_track) in zip(cols, available[idx:idx + 2]):
			with col:
				st.subheader(title)
				st.image(str(path), caption=path.name, width="stretch")
				st.caption(talk_track)


def render_text_analysis() -> None:
	st.header("Text Analysis")
	terms_df = load_csv(OUTPUT_DIR / "text_tfidf_term_scores.csv")
	keyword_plot = OUTPUT_DIR / "text_tfidf_keywords.png"
	if terms_df is None or terms_df.empty:
		st.info("No TF-IDF term scores found.")
		return

	pos_terms = terms_df.nlargest(15, "pos_score")[["term", "pos_score"]]
	neg_terms = terms_df.nlargest(15, "neg_score")[["term", "neg_score"]]

	left, right = st.columns(2)
	with left:
		st.subheader("Top Positive Keywords")
		st.dataframe(pos_terms, width="stretch")
	with right:
		st.subheader("Top Negative Keywords")
		st.dataframe(neg_terms, width="stretch")

	if keyword_plot.exists():
		st.image(str(keyword_plot), caption="TF-IDF keyword importance")


def render_visual_features() -> None:
	st.header("Visual Features")
	tsne_df = load_csv(OUTPUT_DIR / "visual_features_tsne.csv")
	if tsne_df is None or tsne_df.empty:
		st.info("No visual feature embeddings found.")
		return

	color_mode = st.radio("Color points by", ["sentiment", "region"], horizontal=True)
	plot_df = tsne_df.copy()
	if color_mode == "region":
		top_regions = plot_df["region"].value_counts().head(10).index.tolist()
		plot_df["region_group"] = plot_df["region"].where(plot_df["region"].isin(top_regions), "Other")
		fig = px.scatter(
			plot_df,
			x="tsne_x",
			y="tsne_y",
			color="region_group",
			title="t-SNE of CNN Embeddings by Region",
			opacity=0.65,
		)
	else:
		plot_df["sentiment_label"] = plot_df["sentiment"].map({1: "Positive", 0: "Negative"})
		fig = px.scatter(
			plot_df,
			x="tsne_x",
			y="tsne_y",
			color="sentiment_label",
			title="t-SNE of CNN Embeddings by Sentiment",
			opacity=0.65,
		)
	fig.update_traces(marker=dict(size=5))
	st.plotly_chart(fig, width="stretch", key="visual_tsne_scatter")

	image_cols = st.columns(3)
	for column, image_name in zip(
		image_cols,
		[
			"visual_features_tsne_sentiment.png",
			"visual_features_tsne_region.png",
			"visual_features_pca_scree.png",
		],
	):
		image_path = OUTPUT_DIR / image_name
		with column:
			if image_path.exists():
				st.image(str(image_path), caption=image_name)


def _render_noise_analysis(noise_df: pd.DataFrame | None) -> None:
	"""Render a detailed, written + charted breakdown of the three noise axes."""
	if noise_df is None or noise_df.empty:
		st.info("No data quality analysis found. Run yelp.py to generate outputs/data_quality_noise_analysis.csv.")
		return

	# Normalise column names to lowercase so the function is robust to CSV case variation
	noise_df = noise_df.copy()
	noise_df.columns = [c.lower() for c in noise_df.columns]
	for col in ["f1", "accuracy", "samples"]:
		if col in noise_df.columns:
			noise_df[col] = pd.to_numeric(noise_df[col], errors="coerce")

	def get_group(analysis: str, group: str) -> dict:
		rows = noise_df[(noise_df["analysis"] == analysis) & (noise_df["group"] == group)]
		if rows.empty:
			return {}
		return rows.iloc[0].to_dict()

	# ── Overview bar chart ─────────────────────────────────────────────────
	fig_overview = px.bar(
		noise_df,
		x="group",
		y="f1",
		color="analysis",
		barmode="group",
		text=noise_df["f1"].map(lambda v: f"{v:.3f}" if pd.notna(v) else ""),
		title="F1 Score by Noise Subgroup (all axes)",
		labels={"f1": "F1 Score", "group": "Subgroup", "analysis": "Noise Axis"},
		color_discrete_sequence=px.colors.qualitative.Set2,
	)
	fig_overview.update_traces(textposition="outside")
	fig_overview.update_layout(yaxis_range=[0, 1.1])
	st.plotly_chart(fig_overview, width="stretch", key="noise_overview_f1_bar")

	col1, col2, col3 = st.columns(3)

	# ── Axis 1: Review Length ──────────────────────────────────────────────
	with col1:
		st.markdown("##### ✍️ Review Length")
		short = get_group("review_length", "short")
		long  = get_group("review_length", "long")

		if short and long:
			f1_s = short.get("f1"); f1_l = long.get("f1")
			acc_s = short.get("accuracy"); acc_l = long.get("accuracy")
			n_s = int(short.get("samples", 0)); n_l = int(long.get("samples", 0))

			st.metric("Short reviews F1",  f"{f1_s:.4f}" if f1_s is not None else "—",
					  delta=f"{f1_s - f1_l:+.4f} vs long" if (f1_s is not None and f1_l is not None) else None)
			st.metric("Long reviews F1",   f"{f1_l:.4f}" if f1_l is not None else "—")
			st.metric("Short Accuracy",    f"{acc_s:.4f}" if acc_s is not None else "—")
			st.metric("Long  Accuracy",    f"{acc_l:.4f}" if acc_l is not None else "—")

			fig = px.bar(
				x=["Short", "Long"],
				y=[f1_s, f1_l],
				text=[f"{f1_s:.3f}", f"{f1_l:.3f}"],
				labels={"x": "Review Length", "y": "F1"},
				title=f"Short (n={n_s}) vs Long (n={n_l})",
				color=["Short", "Long"],
				color_discrete_sequence=["#636EFA", "#EF553B"],
			)
			fig.update_traces(textposition="outside")
			fig.update_layout(showlegend=False, yaxis_range=[0, 1.1])
			st.plotly_chart(fig, width="stretch", key="noise_review_length_bar")

			# Written interpretation
			if f1_s is not None and f1_l is not None:
				delta = f1_l - f1_s
				if delta > 0.02:
					st.markdown(
						f"> **Long reviews are easier for the model** (+{delta:.4f} F1). "
						f"Longer text provides richer sentiment signal, so short reviews introduce noise — "
						f"the model has less information to work with."
					)
				elif delta < -0.02:
					st.markdown(
						f"> **Short reviews score higher** ({delta:+.4f} F1 vs long). "
						f"This may reflect that very short reviews are often unambiguous (e.g., 'Amazing!' or 'Terrible!'), "
						f"while long reviews contain nuanced or mixed sentiment that is harder to classify."
					)
				else:
					st.markdown(
						f"> **Review length has minimal impact** (F1 gap: {delta:+.4f}). "
						f"The LSTM may be capturing sentiment from key phrases regardless of total length."
					)
		else:
			st.info("review_length subgroups not found in CSV.")

	# ── Axis 2: Caption Availability ──────────────────────────────────────
	with col2:
		st.markdown("##### 🖼️ Caption Availability")
		with_cap    = get_group("caption_availability", "with_caption")
		without_cap = get_group("caption_availability", "without_caption")

		if with_cap and without_cap:
			f1_w = with_cap.get("f1"); f1_wo = without_cap.get("f1")
			acc_w = with_cap.get("accuracy"); acc_wo = without_cap.get("accuracy")
			n_w = int(with_cap.get("samples", 0)); n_wo = int(without_cap.get("samples", 0))

			st.metric("With caption F1",    f"{f1_w:.4f}"  if f1_w  is not None else "—",
					  delta=f"{f1_w - f1_wo:+.4f} vs no caption" if (f1_w is not None and f1_wo is not None) else None)
			st.metric("No caption F1",      f"{f1_wo:.4f}" if f1_wo is not None else "—")
			st.metric("With caption Acc",   f"{acc_w:.4f}"  if acc_w  is not None else "—")
			st.metric("No caption Acc",     f"{acc_wo:.4f}" if acc_wo is not None else "—")

			fig = px.bar(
				x=["With Caption", "No Caption"],
				y=[f1_w, f1_wo],
				text=[f"{f1_w:.3f}", f"{f1_wo:.3f}"],
				labels={"x": "Caption", "y": "F1"},
				title=f"With (n={n_w}) vs Without (n={n_wo})",
				color=["With Caption", "No Caption"],
				color_discrete_sequence=["#00CC96", "#AB63FA"],
			)
			fig.update_traces(textposition="outside")
			fig.update_layout(showlegend=False, yaxis_range=[0, 1.1])
			st.plotly_chart(fig, width="stretch", key="noise_caption_availability_bar")

			# Written interpretation
			if f1_w is not None and f1_wo is not None:
				delta = f1_w - f1_wo
				if delta > 0.02:
					st.markdown(
						f"> **Captions improve performance** (+{delta:.4f} F1). "
						f"Photo captions add an additional text signal on top of the review, "
						f"helping the full multimodal model align the visual and language branches."
					)
				elif delta < -0.02:
					st.markdown(
						f"> **Missing captions do not hurt significantly** ({delta:+.4f} F1). "
						f"The model may be treating missing captions as neutral padding without degrading the visual branch."
					)
				else:
					st.markdown(
						f"> **Caption availability has minimal impact** (F1 gap: {delta:+.4f}). "
						f"Caption text may be too short or repetitive to add meaningful signal beyond the review itself."
					)
		else:
			st.info("caption_availability subgroups not found in CSV.")

	# ── Axis 3: Image Quality ──────────────────────────────────────────────
	with col3:
		st.markdown("##### 📷 Image Quality")
		low_q  = get_group("image_quality", "low_quality")
		high_q = get_group("image_quality", "high_quality")

		if low_q and high_q:
			f1_lo = low_q.get("f1"); f1_hi = high_q.get("f1")
			acc_lo = low_q.get("accuracy"); acc_hi = high_q.get("accuracy")
			n_lo = int(low_q.get("samples", 0)); n_hi = int(high_q.get("samples", 0))

			st.metric("Low quality F1",     f"{f1_lo:.4f}" if f1_lo is not None else "—",
					  delta=f"{f1_lo - f1_hi:+.4f} vs high quality" if (f1_lo is not None and f1_hi is not None) else None)
			st.metric("High quality F1",    f"{f1_hi:.4f}" if f1_hi is not None else "—")
			st.metric("Low quality Acc",    f"{acc_lo:.4f}" if acc_lo is not None else "—")
			st.metric("High quality Acc",   f"{acc_hi:.4f}" if acc_hi is not None else "—")

			fig = px.bar(
				x=["Low Quality", "High Quality"],
				y=[f1_lo, f1_hi],
				text=[f"{f1_lo:.3f}", f"{f1_hi:.3f}"],
				labels={"x": "Image Quality", "y": "F1"},
				title=f"Low (n={n_lo}) vs High (n={n_hi})",
				color=["Low Quality", "High Quality"],
				color_discrete_sequence=["#FFA15A", "#19D3F3"],
			)
			fig.update_traces(textposition="outside")
			fig.update_layout(showlegend=False, yaxis_range=[0, 1.1])
			st.plotly_chart(fig, width="stretch", key="noise_image_quality_bar")

			# Written interpretation
			if f1_lo is not None and f1_hi is not None:
				delta = f1_hi - f1_lo
				if delta > 0.02:
					st.markdown(
						f"> **Higher image quality helps** (+{delta:.4f} F1). "
						f"Images with stronger edge energy (sharper, more detailed photos) carry more visual sentiment signal. "
						f"Blurry or dark photos are a real source of noise for the image branch."
					)
				elif delta < -0.02:
					st.markdown(
						f"> **Low-quality images score surprisingly higher** ({delta:+.4f} F1 on low-quality). "
						f"This could indicate that blurry images happen to belong to a skewed subset of the dataset "
						f"(e.g., a particular rating bucket), rather than reflecting true image quality signal."
					)
				else:
					st.markdown(
						f"> **Image quality has minimal impact** (F1 gap: {delta:+.4f}). "
						f"The image branch may be relying on coarser features (color, composition) that are robust to sharpness."
					)
		else:
			st.info("image_quality subgroups not found in CSV.")

	# ── Overall written summary ────────────────────────────────────────────
	st.divider()
	st.markdown("#### Overall Noise Assessment")

	short_d  = get_group("review_length",       "short")
	long_d   = get_group("review_length",       "long")
	with_d   = get_group("caption_availability", "with_caption")
	wo_d     = get_group("caption_availability", "without_caption")
	low_d    = get_group("image_quality",        "low_quality")
	high_d   = get_group("image_quality",        "high_quality")

	axes_info = []
	if short_d and long_d and short_d.get("f1") is not None and long_d.get("f1") is not None:
		gap = abs(long_d["f1"] - short_d["f1"])
		axes_info.append(("Review length",      gap, "longer reviews contain richer sentiment signal"))
	if with_d and wo_d and with_d.get("f1") is not None and wo_d.get("f1") is not None:
		gap = abs(with_d["f1"] - wo_d["f1"])
		axes_info.append(("Caption availability", gap, "captions add auxiliary text signal for the fusion model"))
	if low_d and high_d and low_d.get("f1") is not None and high_d.get("f1") is not None:
		gap = abs(high_d["f1"] - low_d["f1"])
		axes_info.append(("Image quality",       gap, "sharper photos carry cleaner visual sentiment signal"))

	if axes_info:
		axes_info_sorted = sorted(axes_info, key=lambda x: x[1], reverse=True)
		most_impactful = axes_info_sorted[0]
		least_impactful = axes_info_sorted[-1]

		bullets = []
		for name, gap, reason in axes_info_sorted:
			impact = "high" if gap > 0.05 else ("moderate" if gap > 0.02 else "low")
			bullets.append(f"- **{name}**: F1 gap of **{gap:.4f}** — {impact} sensitivity. _{reason.capitalize()}._")

		st.markdown("\n".join(bullets))
		st.markdown(
			f"\n**Most impactful noise source**: {most_impactful[0]} (F1 gap {most_impactful[1]:.4f}). "
			f"This is the axis where cleaning or filtering the data would likely yield the biggest performance gain.\n\n"
			f"**Least impactful noise source**: {least_impactful[0]} (F1 gap {least_impactful[1]:.4f}). "
			f"The model is relatively robust to this source of noise and may not need special handling."
		)
	else:
		st.info("Not enough subgroup data to generate a summary.")








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
	cuisines = _get_demo_cuisines(reference_data)
	sample_image_path = _load_sample_demo_image(reference_data)

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

	# ── scenario loader ───────────────────────────────────────────────────────
	scol1, scol2 = st.columns([3, 1])
	with scol1:
		scenario_name = st.selectbox(
			"Quick-load a scenario", list(DEMO_SCENARIOS.keys()), index=1, key="upload_scenario_select"
		)
	with scol2:
		st.markdown("<br>", unsafe_allow_html=True)
		if st.button("Load scenario", key="upload_load_scenario"):
			scenario = DEMO_SCENARIOS[scenario_name]
			st.session_state["input_review_text"] = scenario["review"]
			st.session_state["input_region"] = scenario["region"]
			st.session_state["input_cuisine"] = scenario["cuisine"]
			st.rerun()

	col1, col2 = st.columns((1.15, 0.85))
	with col1:
		uploaded_image = st.file_uploader(
			"Food image (optional — sample image used when none uploaded)",
			type=["png", "jpg", "jpeg"],
			key="input_uploader",
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
		st.selectbox("Cuisine", cuisines, index=default_cuisine_idx, key="input_cuisine")

		st.caption(DEMO_SCENARIOS.get(scenario_name, {}).get("insight", ""))

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
		if saved_review:
			st.markdown("**Review snippet:**")
			snippet = saved_review[:160] + ("…" if len(saved_review) > 160 else "")
			st.caption(snippet)

		st.markdown(f"**Region:** {st.session_state.get('input_region', '—')}")
		st.markdown(f"**Cuisine:** {st.session_state.get('input_cuisine', '—')}")

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
	m1, m2, m3, m4 = st.columns(4)
	with m1:
		st.metric("Sentiment", result["label"])
	with m2:
		st.metric("Confidence", f"{float(result['confidence']):.0%}")
	with m3:
		st.metric("Predicted rating", f"{float(result['predicted_rating']):.2f}")
	with m4:
		st.metric("Category prediction", str(result["predicted_category"]))
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

	# ── current inputs summary ─────────────────────────────────────────────
	st.markdown("**Current inputs (from Upload & Review tab):**")
	inp1, inp2, inp3 = st.columns(3)
	with inp1:
		st.markdown(f"**Region:** {region}  \n**Cuisine:** {cuisine}")
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
	if not run_pred and "last_prediction_result" in st.session_state:
		st.divider()
		st.caption("Showing last prediction result. Click the button above to re-run.")
		_display_prediction_result(st.session_state["last_prediction_result"])
		return

	if run_pred:
		image_file = _get_image_from_session_or_sample(
			image_bytes=image_bytes,
			sample_image_path=sample_image_path,
			use_sample=use_sample,
		)
		with st.spinner("Running checkpoint inference…"):
			try:
				result = _estimate_demo_prediction(
					review_text, image_file, region, cuisine,
					reference_data, sample_image_path, use_sample,
				)
			except RuntimeError as exc:
				st.error(str(exc))
				return
		st.session_state["last_prediction_result"] = result
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
		f"Category: {comparison['predicted_category']}"
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

	st.markdown("**Using inputs from Upload & Review tab:**")
	inp1, inp2, inp3 = st.columns(3)
	with inp1:
		st.markdown(f"**Region:** {region}  \n**Cuisine:** {cuisine}")
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

	if not run_compare and "last_comparison_result" in st.session_state:
		st.divider()
		st.caption("Showing last comparison result. Click the button above to re-run.")
		_display_comparison_result(st.session_state["last_comparison_result"])
		return

	if run_compare:
		image_file = _get_image_from_session_or_sample(
			image_bytes=image_bytes,
			sample_image_path=sample_image_path,
			use_sample=use_sample,
		)
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
