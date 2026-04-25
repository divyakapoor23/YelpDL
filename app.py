from pathlib import Path
import subprocess

import pandas as pd
import plotly.express as px
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PYTHON_BIN = "/opt/anaconda3/envs/YelpDL/bin/python"


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
	st.dataframe(df, use_container_width=True)
	return True


def sidebar_controls() -> None:
	st.sidebar.title("Controls")
	st.sidebar.caption("Run the training/analysis pipeline or inspect saved outputs.")

	if st.sidebar.button("Run yelp.py", use_container_width=True):
		with st.sidebar:
			with st.spinner("Running yelp.py. This can take several minutes..."):
				result = subprocess.run(
					[PYTHON_BIN, "-u", "yelp.py"],
					cwd=PROJECT_ROOT,
					capture_output=True,
					text=True,
				)
				st.session_state["pipeline_stdout"] = result.stdout
				st.session_state["pipeline_stderr"] = result.stderr
				st.session_state["pipeline_code"] = result.returncode
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
	st.plotly_chart(fig, use_container_width=True)

	st.subheader("Ablation Table")
	st.dataframe(chart_df, use_container_width=True)


def render_attention() -> None:
	st.header("Attention")
	attention_paths = {
		"Image + Text": OUTPUT_DIR / "attention_Image_plus_Text.csv",
		"Image + Text + Region": OUTPUT_DIR / "attention_Image_plus_Text_plus_Region.csv",
	}
	choice = st.selectbox("Attention file", list(attention_paths.keys()))
	attention_df = load_csv(attention_paths[choice])
	if attention_df is None or attention_df.empty:
		st.info("No attention weights found for this model.")
		return

	col1, col2 = st.columns(2)
	with col1:
		metric_card("Mean image attention", f"{attention_df['alpha_img'].mean():.4f}")
	with col2:
		metric_card("Mean text attention", f"{attention_df['alpha_txt'].mean():.4f}")

	melted = attention_df.melt(var_name="modality", value_name="weight")
	fig = px.histogram(
		melted,
		x="weight",
		color="modality",
		barmode="overlay",
		nbins=40,
		title=f"Attention Weight Distribution: {choice}",
	)
	st.plotly_chart(fig, use_container_width=True)
	st.dataframe(attention_df.head(50), use_container_width=True)


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
	st.header("Consistency")
	mismatch_df = load_csv(OUTPUT_DIR / "image_text_consistency_predictions.csv")
	region_summary = load_csv(OUTPUT_DIR / "image_text_consistency_region_summary.csv")
	if mismatch_df is None or mismatch_df.empty:
		st.info("No consistency analysis outputs found.")
		return

	mismatch_rate = mismatch_df["mismatch"].mean()
	col1, col2, col3 = st.columns(3)
	with col1:
		metric_card("Mismatch rate", f"{mismatch_rate:.4f}")
	with col2:
		metric_card("Image positive / Text negative", str(int(mismatch_df["img_pos_txt_neg"].sum())))
	with col3:
		metric_card("Image negative / Text positive", str(int(mismatch_df["img_neg_txt_pos"].sum())))

	if region_summary is not None and not region_summary.empty:
		fig = px.bar(
			region_summary.sort_values("mismatch_rate", ascending=False).head(12),
			x="region",
			y="mismatch_rate",
			title="Top Regions by Image-Text Disagreement",
			color="mismatch_rate",
		)
		st.plotly_chart(fig, use_container_width=True)

	st.subheader("Mismatch Samples")
	st.dataframe(mismatch_df[mismatch_df["mismatch"]].head(100), use_container_width=True)


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
		st.dataframe(pos_terms, use_container_width=True)
	with right:
		st.subheader("Top Negative Keywords")
		st.dataframe(neg_terms, use_container_width=True)

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
	st.plotly_chart(fig, use_container_width=True)

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


def render_region_and_quality() -> None:
	st.header("Region and Data Quality")
	region_df = load_csv(OUTPUT_DIR / "region_sentiment_stats.csv")
	noise_df = load_csv(OUTPUT_DIR / "data_quality_noise_analysis.csv")

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
			st.plotly_chart(fig, use_container_width=True)

	with right:
		st.subheader("Data Quality / Noise")
		if render_table_or_message(noise_df, "No data quality analysis found."):
			fig = px.bar(
				noise_df,
				x="group",
				y="f1",
				color="analysis",
				barmode="group",
				title="F1 by Data Quality Subgroup",
			)
			st.plotly_chart(fig, use_container_width=True)


def main() -> None:
	sidebar_controls()

	st.title("Yelp Multimodal Analysis App")
	st.caption("Streamlit dashboard for the multimodal Yelp pipeline in yelp.py")

	overview_tab, attention_tab, retrieval_tab, consistency_tab, text_tab, visual_tab, region_tab = st.tabs(
		[
			"Overview",
			"Attention",
			"Retrieval",
			"Consistency",
			"Text",
			"Visual",
			"Region + Quality",
		]
	)

	with overview_tab:
		render_overview()
	with attention_tab:
		render_attention()
	with retrieval_tab:
		render_retrieval()
	with consistency_tab:
		render_consistency()
	with text_tab:
		render_text_analysis()
	with visual_tab:
		render_visual_features()
	with region_tab:
		render_region_and_quality()


if __name__ == "__main__":
	main()
