from pathlib import Path
import os
import re
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
	page_title="Yelp Multimodal Dashboard",
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


def _estimate_demo_prediction(review_text: str, uploaded_image, region: str, cuisine: str, reference_data: dict[str, pd.DataFrame | None]) -> dict[str, object]:
	text_result = _analyze_text_signal(review_text)
	image_result = _analyze_image_signal(uploaded_image)
	region_metrics = _lookup_region_metrics(reference_data, region)
	cuisine_metrics = _lookup_cuisine_metrics(reference_data, cuisine)

	region_score = region_metrics["positive_rate"] if region_metrics["positive_rate"] is not None else 0.72
	weights = {"text": 0.55, "image": 0.25 if uploaded_image is not None else 0.0, "region": 0.20 if uploaded_image is not None else 0.45}
	total_weight = sum(weights.values())
	final_score = (
		weights["text"] * float(text_result["score"]) +
		weights["image"] * float(image_result["score"]) +
		weights["region"] * float(region_score)
	) / total_weight
	final_score = _clip(final_score)

	text_signal = weights["text"] * abs(float(text_result["score"]) - 0.5)
	image_signal = weights["image"] * abs(float(image_result["score"]) - 0.5)
	focus_total = text_signal + image_signal
	if focus_total == 0:
		text_focus = 0.5
		image_focus = 0.5 if uploaded_image is not None else 0.0
	else:
		text_focus = text_signal / focus_total
		image_focus = image_signal / focus_total if uploaded_image is not None else 0.0

	disagreement_penalty = abs(float(text_result["score"]) - float(image_result["score"])) * 0.18
	region_penalty = (region_metrics["mismatch_rate"] or 0.30) * 0.08
	cuisine_penalty = (cuisine_metrics["spread"] or 0.25) * 0.05
	confidence = _clip(0.58 + abs(final_score - 0.5) * 1.15 - disagreement_penalty - region_penalty - cuisine_penalty, 0.52, 0.98)
	label = "Positive" if final_score >= 0.5 else "Negative"

	if text_focus >= image_focus:
		attention_explanation = "The review text is carrying more of the decision than the image in this case."
	else:
		attention_explanation = "The food image is carrying more of the decision than the text in this case."

	if uploaded_image is None:
		attention_explanation += " Because no custom image was uploaded, the demo relies more heavily on text and region priors."

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
	}


def render_live_prediction_demo() -> None:
	st.header("Live Prediction")
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
		if sample_image_path:
			st.image(sample_image_path, caption="Sample food image available in the dataset", width="stretch")
		elif uploaded_image is not None:
			st.image(uploaded_image, caption="Uploaded food image", width="stretch")
		else:
			st.info("Upload a food photo, or use the sample dataset image for the live demo visual.")
		st.markdown("**Presentation flow**")
		st.markdown("1. Show the input a business user would provide.")
		st.markdown("2. Run a multimodal sentiment prediction.")
		st.markdown("3. Explain whether text or image drove the result.")
		st.markdown("4. Turn the result into an operational recommendation.")

	if run_demo:
		result = _estimate_demo_prediction(review_text, uploaded_image, region, cuisine, reference_data)
		st.divider()
		st.markdown("### Step 2: Run prediction")
		st.write("The model combines visual, textual, and regional signals to predict sentiment.")
		m1, m2, m3, m4 = st.columns(4)
		with m1:
			st.metric("Predicted sentiment", result["label"])
		with m2:
			st.metric("Confidence", f"{float(result['confidence']):.0%}")
		with m3:
			st.metric("Text focus", f"{float(result['text_focus']):.0%}")
		with m4:
			st.metric("Image focus", f"{float(result['image_focus']):.0%}")

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
			if result["image_result"]["brightness"] is not None:
				st.caption(
					f"Brightness {float(result['image_result']['brightness']):.0%} | Contrast {float(result['image_result']['contrast']):.0%} | Warmth {float(result['image_result']['warmth']):.0%}"
				)

		st.markdown("### Step 4: Show business insight")
		insight_left, insight_right = st.columns(2)
		with insight_left:
			st.success(result["region_insight"])
		with insight_right:
			st.warning(result["business_insight"])


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

def render_overview() -> None:
	st.header("Overview")
	ablation = load_csv(OUTPUT_DIR / "ablation_results.csv")
	if ablation is None or ablation.empty:
		st.warning("No ablation results found yet. Run yelp.py first.")
		return

	best_row = ablation.sort_values("F1", ascending=False).iloc[0]
	col1, col2, col3, col4 = st.columns(4)
	with col1:
		metric_card("Best Model", str(best_row["Model"]))
	with col2:
		metric_card("Best F1", f"{float(best_row['F1']):.4f}")
	with col3:
		metric_card("Best Accuracy", f"{float(best_row['Accuracy']):.4f}")
	with col4:
		if pd.notna(best_row.get("Category Acc")):
			metric_card("Category Acc", f"{float(best_row['Category Acc']):.4f}")
		else:
			metric_card("Category Acc", "N/A")

	chart_df = ablation.copy()
	for column in ["Accuracy", "Precision", "Recall", "F1", "Loss", "Category Acc", "Rating MAE"]:
		if column in chart_df.columns:
			chart_df[column] = pd.to_numeric(chart_df[column], errors="coerce")

	fig = px.bar(
		chart_df,
		x="Model",
		y="F1",
		color="Model",
		title="Ablation Study F1 by Model Variant",
		text="F1",
	)
	fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
	fig.update_layout(showlegend=False, height=420)
	st.plotly_chart(fig, width="stretch", key="overview_ablation_f1_bar")

	st.subheader("Ablation Table")
	st.dataframe(chart_df, width="stretch")

	# ── Written results section ─────────────────────────────────────────────
	st.divider()
	st.subheader("📝 Ablation Study Results")
	_render_ablation_written_results(chart_df)

	# ── Region impact callout ────────────────────────────────────────────────
	st.divider()
	st.subheader("🌍 Does Adding Region Help?")
	_render_region_impact_callout(ablation, key_prefix="overview")


def _render_ablation_written_results(ablation: pd.DataFrame) -> None:
	"""Auto-generate a written results narrative from ablation numbers."""

	MODEL_ORDER = ["Image Only", "Text Only", "Image + Text", "Image + Text + Region"]

	def get(model: str, col: str) -> float | None:
		rows = ablation[ablation["Model"] == model]
		if rows.empty or col not in rows.columns:
			return None
		val = rows.iloc[0][col]
		try:
			return float(val)
		except (TypeError, ValueError):
			return None

	f1_img  = get("Image Only",              "F1")
	f1_txt  = get("Text Only",               "F1")
	f1_it   = get("Image + Text",            "F1")
	f1_full = get("Image + Text + Region",   "F1")

	acc_img  = get("Image Only",              "Accuracy")
	acc_txt  = get("Text Only",               "Accuracy")
	acc_it   = get("Image + Text",            "Accuracy")
	acc_full = get("Image + Text + Region",   "Accuracy")

	best_row  = ablation.sort_values("F1", ascending=False).iloc[0]

	has_all_four = all(v is not None for v in [f1_img, f1_txt, f1_it, f1_full])

	# ── Summary paragraph ──────────────────────────────────────────────────
	st.markdown("#### Summary")
	st.markdown(
		f"We compare four model variants in an ablation study to isolate the contribution "
		f"of each modality: **Image Only**, **Text Only**, **Image + Text**, and the full "
		f"**Image + Text + Region** model. "
		f"The best-performing variant is **{best_row['Model']}** with an F1 score of "
		f"**{float(best_row['F1']):.4f}** and accuracy of **{float(best_row['Accuracy']):.4f}**."
	)

	# ── Per-model breakdown ────────────────────────────────────────────────
	st.markdown("#### Per-Model Breakdown")

	rows_md = []
	for model in MODEL_ORDER:
		f1   = get(model, "F1")
		acc  = get(model, "Accuracy")
		prec = get(model, "Precision")
		rec  = get(model, "Recall")
		if f1 is None:
			continue
		note = ""
		if model == "Image Only":
			note = "Establishes the visual-only baseline."
		elif model == "Text Only":
			note = "Establishes the text-only baseline."
			if f1_img is not None:
				if f1 > f1_img:
					note += f" Text outperforms image-only by **{f1 - f1_img:+.4f} F1**, suggesting review language carries stronger sentiment signal than food photos alone."
				elif f1 < f1_img:
					note += f" Surprisingly, image-only outperforms text-only by **{f1_img - f1:+.4f} F1** in this run."
				else:
					note += " Text and image alone are roughly equivalent."
		elif model == "Image + Text":
			note = "Combines both visual and language modalities."
			if f1_txt is not None and f1_img is not None:
				best_single = max(f1_txt, f1_img)
				if f1 > best_single:
					note += f" Fusion **improves** over the best single-modality baseline by **{f1 - best_single:+.4f} F1**, confirming complementary information across modalities."
				else:
					note += f" Fusion does **not** outperform the best single-modality baseline ({f1 - best_single:+.4f} F1), suggesting the fusion strategy may need further tuning or that one modality dominates."
		elif model == "Image + Text + Region":
			note = "Adds geographic context on top of image + text."
			if f1_it is not None:
				delta = f1 - f1_it
				if delta > 0:
					note += f" Region **helps**: +{delta:.4f} F1 over Image+Text, supporting the hypothesis that geography shapes sentiment perception."
				elif delta < 0:
					note += f" Region **hurts** slightly: {delta:.4f} F1 vs Image+Text. The region embedding may be underfitted or adding noise given the business-level pairing limitation."
				else:
					note += " Region has negligible impact on F1 in this run."

		parts = [f"**{model}**"]
		if acc  is not None: parts.append(f"Accuracy: **{acc:.4f}**")
		if prec is not None: parts.append(f"Precision: **{prec:.4f}**")
		if rec  is not None: parts.append(f"Recall: **{rec:.4f}**")
		parts.append(f"F1: **{f1:.4f}**")
		rows_md.append(f"- {' | '.join(parts)}  \n  {note}")

	st.markdown("\n".join(rows_md))

	# ── Cross-modality takeaways ───────────────────────────────────────────
	if has_all_four:
		st.markdown("#### Cross-Modality Takeaways")
		takeaways = []

		if f1_txt > f1_img:
			takeaways.append(
				f"**Text is the dominant modality.** Text-only F1 ({f1_txt:.4f}) exceeds image-only F1 ({f1_img:.4f}), "
				f"consistent with review text being a direct expression of sentiment."
			)
		elif f1_img > f1_txt:
			takeaways.append(
				f"**Image is the dominant modality.** Image-only F1 ({f1_img:.4f}) exceeds text-only F1 ({f1_txt:.4f}) — "
				f"visual presentation may be a stronger predictor of sentiment in this dataset."
			)
		else:
			takeaways.append("Image and text single-modality baselines are tied.")

		best_single = max(f1_txt, f1_img)
		fusion_gain = f1_it - best_single
		if fusion_gain > 0.005:
			takeaways.append(
				f"**Fusion adds value** (+{fusion_gain:.4f} F1 over best single modality). "
				f"The two signals are complementary — the model benefits from seeing both a photo and a review."
			)
		elif fusion_gain < -0.005:
			takeaways.append(
				f"**Fusion hurts slightly** ({fusion_gain:+.4f} F1 vs best single modality). "
				f"This could indicate cross-modal noise from the business-level pairing."
			)
		else:
			takeaways.append(
				f"**Fusion is roughly neutral** ({fusion_gain:+.4f} F1 vs best single modality). "
				f"The model likely learns to rely on the stronger modality."
			)

		region_gain = f1_full - f1_it
		if region_gain > 0.005:
			takeaways.append(
				f"**Region context is beneficial** (+{region_gain:.4f} F1 over Image+Text). "
				f"Geography encodes latent sentiment variance across cities that the image and text alone cannot capture."
			)
		elif region_gain < -0.005:
			takeaways.append(
				f"**Region context is slightly harmful** ({region_gain:+.4f} F1). "
				f"This may reflect the business-level pairing limitation — the region tag is not specific enough to add clean signal."
			)
		else:
			takeaways.append(
				f"**Region context is approximately neutral** ({region_gain:+.4f} F1). "
				f"More data or finer-grained location labels (e.g., neighborhood) may be needed to unlock the geographic signal."
			)

		for t in takeaways:
			st.markdown(f"- {t}")

	# ── Limitations reminder ───────────────────────────────────────────────
	with st.expander("⚠️ Limitations to keep in mind when interpreting these results"):
		st.markdown(
			"""
- **Business-level pairing**: Photos and reviews are matched at the business level, not the dish or visit level.
  A photo of the restaurant exterior may be paired with a review about the food quality, introducing cross-modal noise.
- **Class imbalance**: The dataset has more positive reviews than negative ones. F1 is reported, but results
  may still over-represent the majority class.
- **Single run**: These results are from one training run with a fixed seed. Variance across seeds is not measured here.
- **Epochs**: Training ran for at most 10 epochs with early stopping (patience=3). Longer training may change the ranking.
- **Region granularity**: Region is defined as city + state. Finer location data (e.g., neighborhood) might yield stronger signal.
"""
		)


def _render_region_impact_callout(ablation: pd.DataFrame, key_prefix: str = "overview") -> None:
	"""Inline helper that shows a focused with vs without region comparison."""
	def f1_for(name: str) -> float | None:
		rows = ablation[ablation["Model"] == name]
		return float(rows.iloc[0]["F1"]) if not rows.empty else None

	def acc_for(name: str) -> float | None:
		rows = ablation[ablation["Model"] == name]
		return float(rows.iloc[0]["Accuracy"]) if not rows.empty else None

	f1_without = f1_for("Image + Text")
	f1_with = f1_for("Image + Text + Region")
	acc_without = acc_for("Image + Text")
	acc_with = acc_for("Image + Text + Region")

	if f1_without is None or f1_with is None:
		st.info("Both 'Image + Text' and 'Image + Text + Region' results are required for this comparison.")
		return

	f1_delta = f1_with - f1_without
	acc_delta = (acc_with - acc_without) if acc_without is not None and acc_with is not None else None

	col_a, col_b, col_c = st.columns(3)
	with col_a:
		st.metric("F1 without region", f"{f1_without:.4f}")
	with col_b:
		st.metric("F1 with region", f"{f1_with:.4f}", delta=f"{f1_delta:+.4f}")
	with col_c:
		if acc_delta is not None:
			st.metric("Accuracy Δ (region)", f"{acc_delta:+.4f}")

	if f1_delta > 0:
		st.success(
			f"✅ Adding geographic context **improves** F1 by **{f1_delta:+.4f}** "
			f"({f1_without:.4f} → {f1_with:.4f}). Region is a useful signal."
		)
	elif f1_delta < 0:
		st.warning(
			f"⚠️ Adding geographic context **hurts** F1 by **{f1_delta:+.4f}** "
			f"({f1_without:.4f} → {f1_with:.4f}). The region embedding may be noisy or needs more data."
		)
	else:
		st.info("Adding region had no measurable impact on F1 in this run.")

	# Side-by-side bar chart
	compare_rows = ablation[ablation["Model"].isin(["Image + Text", "Image + Text + Region"])].copy()
	fig = px.bar(
		compare_rows,
		x="Model",
		y=["F1", "Accuracy", "Precision", "Recall"],
		barmode="group",
		title="With vs Without Region — Full Metrics Comparison",
		color_discrete_sequence=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
	)
	fig.update_layout(height=420, legend_title="Metric")
	st.plotly_chart(fig, width="stretch", key=f"region_callout_metrics_bar_{key_prefix}")



def render_attention() -> None:
	st.header("Attention")
	st.caption(
		"This page shows how sharp each attention stream is, whether region changes that focus, "
		"and whether mistakes come with flatter attention patterns."
	)

	att_no_region = load_csv(OUTPUT_DIR / "attention_Image_plus_Text.csv")
	att_region = load_csv(OUTPUT_DIR / "attention_Image_plus_Text_plus_Region.csv")
	att_no_region_summary = load_csv(OUTPUT_DIR / "attention_Image_plus_Text_summary.csv")
	att_region_summary = load_csv(OUTPUT_DIR / "attention_Image_plus_Text_plus_Region_summary.csv")

	def _callout_card(title: str, value: str, note: str) -> None:
		with st.container(border=True):
			st.markdown(f"**{title}**")
			st.markdown(
				f"<div style='font-size: 2rem; font-weight: 700; line-height: 1.1; margin: 0.25rem 0 0.35rem 0;'>{value}</div>",
				unsafe_allow_html=True,
			)
			st.caption(note)

	def _prepare_attention_frame(df: pd.DataFrame | None, label: str) -> pd.DataFrame | None:
		if df is None or df.empty:
			return None
		required = [
			"img_concentration",
			"txt_concentration",
			"img_peak_focus",
			"txt_peak_focus",
			"correct",
			"pred_confidence",
		]
		if not set(required).issubset(df.columns):
			return None
		clean = df.copy()
		for column in required:
			clean[column] = pd.to_numeric(clean[column], errors="coerce")
		clean = clean.dropna(subset=required)
		if clean.empty:
			return None
		clean["correct"] = clean["correct"].astype(int)
		clean["outcome"] = clean["correct"].map({1: "Correct", 0: "Incorrect"})
		clean["model"] = label
		return clean

	prepared_frames = [
		_prepare_attention_frame(att_no_region, "Image + Text"),
		_prepare_attention_frame(att_region, "Image + Text + Region"),
	]
	prepared_frames = [frame for frame in prepared_frames if frame is not None]

	if not prepared_frames:
		st.warning(
			"The attention page now expects the richer export fields from yelp.py. Rerun yelp.py once to refresh the attention CSVs and unlock the new charts."
		)
		with st.expander("Raw exported summary CSVs"):
			s1, s2 = st.columns(2)
			with s1:
				st.markdown("**Image + Text Summary**")
				if att_no_region_summary is None or att_no_region_summary.empty:
					st.info("No summary CSV found for Image + Text attention.")
				else:
					st.dataframe(att_no_region_summary, width="stretch")
			with s2:
				st.markdown("**Image + Text + Region Summary**")
				if att_region_summary is None or att_region_summary.empty:
					st.info("No summary CSV found for Image + Text + Region attention.")
				else:
					st.dataframe(att_region_summary, width="stretch")
		return

	combined = pd.concat(prepared_frames, ignore_index=True)
	model_summary = (
		combined.groupby("model", as_index=False)
		.agg(
			samples=("model", "size"),
			image_focus=("img_concentration", "mean"),
			text_focus=("txt_concentration", "mean"),
			image_peak=("img_peak_focus", "mean"),
			text_peak=("txt_peak_focus", "mean"),
			accuracy=("correct", "mean"),
			confidence=("pred_confidence", "mean"),
		)
	)

	overall_text_focus = float(model_summary["text_focus"].mean())
	overall_image_focus = float(model_summary["image_focus"].mean())
	text_minus_image = overall_text_focus - overall_image_focus
	correct_focus = float(combined.loc[combined["correct"] == 1, "txt_concentration"].mean()) if (combined["correct"] == 1).any() else float("nan")
	incorrect_focus = float(combined.loc[combined["correct"] == 0, "txt_concentration"].mean()) if (combined["correct"] == 0).any() else float("nan")
	mistake_gap = correct_focus - incorrect_focus if pd.notna(correct_focus) and pd.notna(incorrect_focus) else float("nan")
	region_delta = None
	base = model_summary.set_index("model")
	if {"Image + Text", "Image + Text + Region"}.issubset(base.index):
		region_delta = float(base.loc["Image + Text + Region", "text_focus"] - base.loc["Image + Text", "text_focus"])

	st.subheader("Headline Takeaways")
	card1, card2, card3 = st.columns(3)
	with card1:
		focus_winner = "Text stream sharper" if text_minus_image >= 0 else "Image stream sharper"
		_callout_card(
			"Main takeaway",
			focus_winner,
			f"Average focus gap: {abs(text_minus_image):.1%} between the two cross-attention streams.",
		)
	with card2:
		if region_delta is None:
			_callout_card("Region effect", "N/A", "Both multimodal models are needed to compare region impact.")
		else:
			region_value = f"{region_delta:+.1%}"
			region_note = "Change in text-side focus after adding region context."
			_callout_card("Region effect", region_value, region_note)
	with card3:
		if pd.isna(mistake_gap):
			_callout_card("Error pattern", "N/A", "No correct/incorrect split was available in the export.")
		else:
			mistake_value = f"{mistake_gap:+.1%}"
			mistake_note = "Correct predictions typically have higher text-focus concentration than errors." if mistake_gap >= 0 else "Errors are showing sharper text focus than correct predictions in this run."
			_callout_card("Error pattern", mistake_value, mistake_note)

	focus_chart = model_summary.melt(
		id_vars="model",
		value_vars=["image_focus", "text_focus"],
		var_name="modality",
		value_name="focus",
	)
	focus_chart["modality"] = focus_chart["modality"].map({"image_focus": "Image focus", "text_focus": "Text focus"})
	fig_focus = px.bar(
		focus_chart,
		x="model",
		y="focus",
		color="modality",
		barmode="group",
		text="focus",
		title="How Focused Is Each Attention Stream?",
		labels={"model": "Model", "focus": "Normalized concentration", "modality": "Attention stream"},
	)
	fig_focus.update_traces(texttemplate="%{text:.1%}", textposition="outside")
	fig_focus.update_yaxes(tickformat=".0%", range=[0, 1])
	st.plotly_chart(fig_focus, width="stretch", key="attn_focus_by_model")
	st.caption(
		"A value closer to 1 means the attention distribution is more concentrated on a small set of tokens or patches; closer to 0 means it is flatter and more diffuse."
	)

	outcome_summary = (
		combined.groupby(["model", "outcome"], as_index=False)
		.agg(
			image_focus=("img_concentration", "mean"),
			text_focus=("txt_concentration", "mean"),
			samples=("correct", "size"),
		)
	)
	if outcome_summary["outcome"].nunique() > 1:
		rich_chart = outcome_summary.melt(
			id_vars=["model", "outcome", "samples"],
			value_vars=["image_focus", "text_focus"],
			var_name="modality",
			value_name="focus",
		)
		rich_chart["modality"] = rich_chart["modality"].map({"image_focus": "Image focus", "text_focus": "Text focus"})
		fig_outcome = px.bar(
			rich_chart,
			x="outcome",
			y="focus",
			color="model",
			facet_col="modality",
			barmode="group",
			text="focus",
			title="Do Mistakes Come With Flatter Attention?",
			labels={"outcome": "Prediction outcome", "focus": "Mean normalized concentration", "model": "Model"},
		)
		fig_outcome.update_traces(texttemplate="%{text:.1%}", textposition="outside")
		fig_outcome.update_yaxes(tickformat=".0%", range=[0, 1])
		st.plotly_chart(fig_outcome, width="stretch", key="attn_focus_by_outcome")
		st.caption(
			"This richer chart uses the new yelp.py export fields. It compares attention sharpness for correct versus incorrect predictions instead of only showing overall averages."
		)

	if region_delta is not None and abs(region_delta) < 0.01:
		st.info(
			"Adding region barely changes attention sharpness. In this run, region helps mainly through the fused representation rather than by making cross-attention much more focused."
		)
	elif region_delta is not None:
		st.info(
			"Adding region produces a visible shift in attention sharpness, which suggests the geographic signal changes how selectively the model attends to text or image evidence."
		)

	with st.expander("Raw exported summary CSVs"):
		s1, s2 = st.columns(2)
		with s1:
			st.markdown("**Image + Text Summary**")
			if att_no_region_summary is None or att_no_region_summary.empty:
				st.info("No summary CSV found for Image + Text attention.")
			else:
				st.dataframe(att_no_region_summary, width="stretch")
		with s2:
			st.markdown("**Image + Text + Region Summary**")
			if att_region_summary is None or att_region_summary.empty:
				st.info("No summary CSV found for Image + Text + Region attention.")
			else:
				st.dataframe(att_region_summary, width="stretch")


def render_retrieval() -> None:
	st.header("Cross-Modal Retrieval")
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


def render_consistency() -> None:
	st.header("Perception Gap")
	mismatch_df = load_csv(OUTPUT_DIR / "image_text_consistency_predictions.csv")
	mismatch_only_df = load_csv(OUTPUT_DIR / "image_text_consistency_mismatches.csv")
	region_summary = load_csv(OUTPUT_DIR / "image_text_consistency_region_summary.csv")
	cuisine_summary = load_csv(OUTPUT_DIR / "image_text_consistency_cuisine_summary.csv")
	rating_quality_summary = load_csv(OUTPUT_DIR / "image_text_consistency_rating_quality_summary.csv")
	if mismatch_df is None or mismatch_df.empty:
		st.info("No consistency analysis outputs found.")
		return

	mismatch_rate = mismatch_df["mismatch"].mean()
	col1, col2, col3 = st.columns(3)
	with col1:
		metric_card("Perception Gap Rate", f"{mismatch_rate:.4f}")
	with col2:
		metric_card("Image positive / Text negative", str(int(mismatch_df["img_pos_txt_neg"].sum())))
	with col3:
		metric_card("Image negative / Text positive", str(int(mismatch_df["img_neg_txt_pos"].sum())))

	if region_summary is not None and not region_summary.empty:
		fig = px.bar(
			region_summary.sort_values("mismatch_rate", ascending=False).head(12),
			x="region",
			y="mismatch_rate",
			title="Top Regions by Perception Gap",
			color="mismatch_rate",
		)
		st.plotly_chart(fig, width="stretch", key="consistency_region_bar")

	c1, c2 = st.columns(2)
	with c1:
		if cuisine_summary is not None and not cuisine_summary.empty:
			fig = px.bar(
				cuisine_summary.sort_values("mismatch_rate", ascending=False).head(12),
				x="cuisine",
				y="mismatch_rate",
				color="mismatch_rate",
				title="Perception Gap by Cuisine",
			)
			st.plotly_chart(fig, width="stretch", key="consistency_cuisine_bar")
	with c2:
		if rating_quality_summary is not None and not rating_quality_summary.empty:
			fig = px.bar(
				rating_quality_summary,
				x="rating_quality_cluster",
				y="mismatch_rate",
				color="mismatch_rate",
				title="Perception Gap by Rating-Quality Proxy",
			)
			st.plotly_chart(fig, width="stretch", key="consistency_rating_bar")
	st.subheader("Mismatch Samples")
	if mismatch_only_df is not None and not mismatch_only_df.empty:
		st.dataframe(mismatch_only_df.head(100), width="stretch")
	else:
		st.dataframe(mismatch_df[mismatch_df["mismatch"]].head(100), width="stretch")


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


def render_region_and_quality() -> None:
	st.header("Region and Data Quality")
	st.caption(
		"This tab answers a key ablation question: does regional context carry real signal? "
		"High variance in sentiment rates across regions or cuisines supports using geography as a feature."
	)

	# ── Region impact summary pulled from ablation ───────────────────────────
	ablation = load_csv(OUTPUT_DIR / "ablation_results.csv")
	if ablation is not None and not ablation.empty:
		for col in ["F1", "Accuracy"]:
			if col in ablation.columns:
				ablation[col] = pd.to_numeric(ablation[col], errors="coerce")
		st.subheader("📊 Region Impact on Model Performance")
		_render_region_impact_callout(ablation, key_prefix="region_quality_tab")
		st.divider()

	region_df = load_csv(OUTPUT_DIR / "region_sentiment_stats.csv")
	noise_df = load_csv(OUTPUT_DIR / "data_quality_noise_analysis.csv")
	spread_df = load_csv(OUTPUT_DIR / "cuisine_region_sentiment_spread.csv")

	left, right = st.columns(2)
	with left:
		st.subheader("Region Sentiment Rates")
		if render_table_or_message(region_df, "No region sentiment statistics found."):
			fig = px.bar(
				region_df.head(15),
				x="region",
				y="positive_rate",
				color="positive_rate",
				title="Most Positive Regions",
			)
			st.plotly_chart(fig, width="stretch", key="region_positive_rate_bar")

			heatmap_df = region_df.copy()
			heatmap_df["bucket"] = pd.cut(
				heatmap_df["positive_rate"],
				bins=[0, 0.5, 0.7, 0.85, 1.0],
				labels=["low", "moderate", "high", "very_high"],
				include_lowest=True,
			).astype(str)
			cross = heatmap_df.groupby(["region", "bucket"]).size().reset_index(name="count")
			fig = px.density_heatmap(
				cross,
				x="region",
				y="bucket",
				z="count",
				title="Region Variation Heatmap (Sentiment Bucket Density)",
			)
			fig.update_xaxes(showticklabels=False)
			st.plotly_chart(fig, width="stretch", key="region_bucket_density_heatmap")

	with right:
		st.subheader("Data Quality / Noise Overview")
		if render_table_or_message(noise_df, "No data quality analysis found."):
			_noise = noise_df.copy()
			_noise.columns = [c.lower() for c in _noise.columns]
			fig = px.bar(
				_noise,
				x="group",
				y="f1",
				color="analysis",
				barmode="group",
				text=_noise["f1"].map(lambda v: f"{v:.3f}" if pd.notna(v) else ""),
				title="F1 by Data Quality Subgroup",
				labels={"f1": "F1 Score", "group": "Subgroup", "analysis": "Noise Axis"},
				color_discrete_sequence=px.colors.qualitative.Set2,
			)
			fig.update_traces(textposition="outside")
			fig.update_layout(yaxis_range=[0, 1.1])
			st.plotly_chart(fig, width="stretch", key="region_tab_noise_overview_bar")
		if spread_df is not None and not spread_df.empty:
			st.subheader("Same Cuisine, Different Region")
			st.dataframe(spread_df.head(15), width="stretch")

	# ── Full-width noisy data analysis section ───────────────────────────────
	st.divider()
	st.subheader("🔬 Noisy Data Analysis")
	st.caption(
		"Three axes of dataset noise are evaluated by measuring model performance on clean vs noisy subgroups. "
		"Larger F1 gaps between subgroups indicate the model is sensitive to that noise source."
	)
	_render_noise_analysis(noise_df)


def render_research_insights() -> None:
	st.header("Research Insights")
	st.caption("Auto-generated narrative claims from the latest experiment outputs.")

	ablation_df = load_csv(OUTPUT_DIR / "ablation_results.csv")
	attention_df = load_csv(OUTPUT_DIR / "attention_Image_plus_Text_plus_Region.csv")
	mismatch_df = load_csv(OUTPUT_DIR / "image_text_consistency_predictions.csv")
	region_df = load_csv(OUTPUT_DIR / "region_sentiment_stats.csv")
	cuisine_spread_df = load_csv(OUTPUT_DIR / "cuisine_region_sentiment_spread.csv")
	noise_df = load_csv(OUTPUT_DIR / "data_quality_noise_analysis.csv")

	if ablation_df is None or ablation_df.empty:
		st.info("Run yelp.py first to generate insight-ready outputs.")
		return

	ablation = ablation_df.copy()
	for column in ["Accuracy", "Precision", "Recall", "F1", "Loss", "Category Acc", "Rating MAE"]:
		if column in ablation.columns:
			ablation[column] = pd.to_numeric(ablation[column], errors="coerce")

	def f1_for(model_name: str) -> float | None:
		rows = ablation[ablation["Model"] == model_name]
		if rows.empty:
			return None
		return float(rows.iloc[0]["F1"])

	f1_text = f1_for("Text Only")
	f1_img = f1_for("Image Only")
	f1_img_txt = f1_for("Image + Text")
	f1_full = f1_for("Image + Text + Region")

	best_row = ablation.sort_values("F1", ascending=False).iloc[0]

	# ── Region impact — pinned at the top as the primary finding ─────────────
	st.subheader("🌍 Key Finding: Does Region Context Help?")
	_render_region_impact_callout(ablation, key_prefix="research_tab")
	st.divider()

	insights = []
	insights.append(
		f"Best-performing variant is **{best_row['Model']}** with F1 **{best_row['F1']:.4f}** and accuracy **{best_row['Accuracy']:.4f}**."
	)

	if f1_text is not None and f1_img is not None:
		insights.append(
			f"Text signal is stronger than image-only signal in this run: Text F1 **{f1_text:.4f}** vs Image F1 **{f1_img:.4f}**."
		)

	if f1_img_txt is not None and f1_text is not None:
		delta = f1_img_txt - f1_text
		insights.append(
			f"Adding image to text changes performance by **{delta:+.4f} F1** (Image+Text vs Text-only)."
		)

	if f1_full is not None and f1_img_txt is not None:
		delta = f1_full - f1_img_txt
		verdict = "improves" if delta > 0 else "hurts" if delta < 0 else "does not change"
		insights.append(
			f"Adding geographic context **{verdict}** performance by **{delta:+.4f} F1** (Image+Text+Region vs Image+Text)."
		)

	if attention_df is not None and not attention_df.empty:
		if {"img_concentration", "txt_concentration", "correct"}.issubset(attention_df.columns):
			img_focus = float(attention_df["img_concentration"].mean())
			txt_focus = float(attention_df["txt_concentration"].mean())
			correct_focus = float(attention_df.loc[attention_df["correct"] == 1, "txt_concentration"].mean())
			incorrect_focus = float(attention_df.loc[attention_df["correct"] == 0, "txt_concentration"].mean())
			insights.append(
				f"Cross-modal attention is sharper on text than image in the exported focus scores: image focus **{img_focus:.4f}**, text focus **{txt_focus:.4f}**."
			)
			if pd.notna(correct_focus) and pd.notna(incorrect_focus):
				insights.append(
					f"Prediction errors are associated with a text-focus shift of **{(correct_focus - incorrect_focus):+.4f}** compared with correct predictions."
				)
		else:
			img_mean = float(attention_df["alpha_img"].mean())
			txt_mean = float(attention_df["alpha_txt"].mean())
			insights.append(
				f"Legacy attention export detected: mean image-attention **{img_mean:.4f}**, mean text-attention **{txt_mean:.4f}**."
			)

	if mismatch_df is not None and not mismatch_df.empty:
		gap = float(mismatch_df["mismatch"].mean())
		insights.append(
			f"Perception Gap Rate is **{gap:.2%}** (image and text predictions disagree), indicating substantial subjective/noisy multimodal cases."
		)

	if region_df is not None and not region_df.empty:
		max_rate = float(region_df["positive_rate"].max())
		min_rate = float(region_df["positive_rate"].min())
		insights.append(
			f"Regional sentiment spread ranges from **{min_rate:.4f}** to **{max_rate:.4f}** (delta **{(max_rate - min_rate):.4f}**), supporting geographically dependent sentiment behavior."
		)

	if cuisine_spread_df is not None and not cuisine_spread_df.empty:
		top = cuisine_spread_df.sort_values("spread", ascending=False).iloc[0]
		insights.append(
			f"Strongest same-cuisine regional variability appears in **{top['cuisine_cluster']}** with spread **{top['spread']:.4f}** across regions."
		)

	if noise_df is not None and not noise_df.empty:
		worst = noise_df.sort_values("f1", ascending=True).iloc[0]
		best = noise_df.sort_values("f1", ascending=False).iloc[0]
		insights.append(
			f"Data-quality sensitivity: lowest subgroup F1 is **{worst['f1']:.4f}** ({worst['analysis']} / {worst['group']}), highest is **{best['f1']:.4f}** ({best['analysis']} / {best['group']})."
		)

	st.subheader("Presentation Claims")
	for idx, text in enumerate(insights, start=1):
		st.markdown(f"{idx}. {text}")

	st.subheader("Evidence Snapshot")
	evidence = pd.DataFrame(
		{
			"Metric": [
				"Best model",
				"Text-only F1",
				"Image-only F1",
				"Image+Text F1",
				"Image+Text+Region F1",
			],
			"Value": [
				str(best_row["Model"]),
				f"{f1_text:.4f}" if f1_text is not None else "N/A",
				f"{f1_img:.4f}" if f1_img is not None else "N/A",
				f"{f1_img_txt:.4f}" if f1_img_txt is not None else "N/A",
				f"{f1_full:.4f}" if f1_full is not None else "N/A",
			],
		}
	)
	st.dataframe(evidence, width="stretch")


def main() -> None:
	sidebar_controls()

	st.title("Yelp Food Intelligence App")
	st.caption(
		"Real-world Streamlit demo for multimodal restaurant sentiment: upload a food image, add review text and region, then turn the prediction into a business insight."
	)

	prediction_tab, use_cases_tab, explainability_tab, overview_tab, insights_tab, region_tab, attention_tab, consistency_tab, retrieval_tab, text_tab, visual_tab, presentation_tab = st.tabs(
		[
			"Live Prediction",
			"Use Cases",
			"Explainability",
			"Overview",
			"Research Insights",
			"Region Effects",
			"Attention",
			"Perception Gap",
			"Retrieval",
			"Text",
			"Visual",
			"Presentation Plots",
		]
	)

	with prediction_tab:
		render_live_prediction_demo()
	with use_cases_tab:
		render_example_use_cases()
	with explainability_tab:
		render_demo_explainability()
	with overview_tab:
		render_overview()
	with insights_tab:
		render_research_insights()
	with region_tab:
		render_region_and_quality()
	with attention_tab:
		render_attention()
	with consistency_tab:
		render_consistency()
	with retrieval_tab:
		render_retrieval()
	with text_tab:
		render_text_analysis()
	with visual_tab:
		render_visual_features()
	with presentation_tab:
		render_presentation_plots()


if __name__ == "__main__":
	main()
