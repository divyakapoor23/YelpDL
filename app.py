from pathlib import Path
import os
import subprocess
import sys
import shutil

import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PYTHON_BIN = sys.executable


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


def missing_pipeline_inputs() -> list[Path]:
	return [path for path in REQUIRED_PIPELINE_INPUTS if not path.exists()]


def build_pipeline_env() -> dict[str, str]:
	env = os.environ.copy()
	for key in DATA_PATH_ENV_VARS:
		if key in env and env[key]:
			continue
		try:
			secret_value = st.secrets.get(key)
		except Exception:
			secret_value = None
		if secret_value:
			env[key] = str(secret_value)
	return env


def sidebar_controls() -> None:
	st.sidebar.title("Controls")
	st.sidebar.caption("Run the training/analysis pipeline or inspect saved outputs.")
	st.sidebar.markdown("**Research Question**")
	st.sidebar.write(RESEARCH_QUESTION)

	missing_inputs = missing_pipeline_inputs()
	if missing_inputs:
		st.sidebar.warning(
			"Pipeline inputs are missing in this environment. "
			"You can still explore the precomputed outputs below."
		)

	run_pipeline = st.sidebar.button(
		"Run yelp.py",
		width="stretch",
		disabled=bool(missing_inputs),
	)

	if run_pipeline:
		with st.sidebar:
			with st.spinner("Running yelp.py. This can take several minutes..."):
				pipeline_env = build_pipeline_env()
				if missing_inputs:
					st.session_state["pipeline_stdout"] = ""
					st.session_state["pipeline_stderr"] = (
						"Pipeline inputs are missing. Please add these paths:\n"
						+ "\n".join(str(path) for path in missing_inputs)
						+ "\n\nSet one or more of: "
						+ ", ".join(DATA_PATH_ENV_VARS)
					)
					st.session_state["pipeline_code"] = 127
					load_csv.clear()
					return

				python_exec = Path(PYTHON_BIN)
				script_path = PROJECT_ROOT / "yelp.py"
				resolved_python = str(python_exec) if python_exec.exists() else shutil.which("python3")

				if not resolved_python:
					st.session_state["pipeline_stdout"] = ""
					st.session_state["pipeline_stderr"] = (
						"No Python interpreter found for pipeline execution. "
						"Expected runtime interpreter or python3 in PATH."
					)
					st.session_state["pipeline_code"] = 127
				elif not script_path.exists():
					st.session_state["pipeline_stdout"] = ""
					st.session_state["pipeline_stderr"] = f"Missing pipeline script: {script_path}"
					st.session_state["pipeline_code"] = 127
				else:
					try:
						result = subprocess.run(
							[resolved_python, "-u", str(script_path)],
							cwd=PROJECT_ROOT,
							capture_output=True,
							text=True,
							env=pipeline_env,
						)
						st.session_state["pipeline_stdout"] = result.stdout
						st.session_state["pipeline_stderr"] = result.stderr
						st.session_state["pipeline_code"] = result.returncode
					except OSError as exc:
						st.session_state["pipeline_stdout"] = ""
						st.session_state["pipeline_stderr"] = str(exc)
						st.session_state["pipeline_code"] = 127
				load_csv.clear()

	if "pipeline_code" in st.session_state:
		code = st.session_state["pipeline_code"]
		if code == 0:
			st.sidebar.success("Last run completed successfully.")
		else:
			st.sidebar.error(f"Last run failed with exit code {code}.")

		with st.sidebar.expander("Last run logs"):
			stdout = st.session_state.get("pipeline_stdout", "")
			stderr = st.session_state.get("pipeline_stderr", "")
			if stdout:
				st.text_area("stdout", stdout[-12000:], height=260)
			if stderr:
				st.text_area("stderr", stderr[-12000:], height=180)


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
			"Talk track: start with the main result. Point out which model performs best and emphasize whether adding region improves over Image + Text.",
		),
		(
			"Best-Model Confusion Matrix",
			OUTPUT_DIR / "final_best_model_confusion_matrix.png",
			"Talk track: explain where the best model is still making mistakes. This helps you discuss false positives vs false negatives instead of only headline accuracy.",
		),
		(
			"Perception Gap by Region",
			OUTPUT_DIR / "final_perception_gap_by_region.png",
			"Talk track: use this to argue that sentiment is not uniform across locations. Some regions show much larger disagreement between image and text perception.",
		),
		(
			"Perception Gap by Cuisine",
			OUTPUT_DIR / "final_perception_gap_by_cuisine.png",
			"Talk track: highlight that the perception gap also depends on cuisine type, which suggests some food categories are visually or textually easier to judge than others.",
		),
		(
			"Attention Comparison",
			OUTPUT_DIR / "final_attention_comparison.png",
			"Talk track: explain how the model balances image vs text information. Use this figure to describe whether region changes modality reliance.",
		),
		(
			"Region/Cuisine Spread",
			OUTPUT_DIR / "final_region_cuisine_spread.png",
			"Talk track: close by showing that the same cuisine can receive different sentiment across regions, which is your strongest justification for region-aware modeling.",
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

	st.subheader("Presentation-Ready Claims")
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

	st.title("Yelp Multimodal Analysis App")
	st.caption(RESEARCH_QUESTION)

	overview_tab, presentation_tab, insights_tab, region_tab, attention_tab, consistency_tab, retrieval_tab, text_tab, visual_tab = st.tabs(
		[
			"Overview",
			"Presentation Plots",
			"Research Insights",
			"Region Effects",
			"Attention",
			"Perception Gap",
			"Retrieval",
			"Text",
			"Visual",
		]
	)

	with overview_tab:
		render_overview()
	with presentation_tab:
		render_presentation_plots()
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


if __name__ == "__main__":
	main()
