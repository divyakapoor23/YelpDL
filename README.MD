
# Multimodal Yelp Restaurant Sentiment Analysis

An end-to-end deep learning system that predicts restaurant sentiment by fusing **review text, food images, and regional metadata**. Built to test whether combining modalities meaningfully improves prediction quality over any single signal alone — includes a full ablation study (Image Only / Text Only / Image + Text / Image + Text + Region) plus an interactive Streamlit demo for exploring model behavior and business insights.

## Why Multimodal?

Star ratings and review text alone miss a lot of context — a blurry photo of a half-eaten dish or a region's typical cuisine style can shift how a review should be interpreted. This project builds a fusion architecture that combines:

- **Text** — review content processed through NLP pipelines
- **Images** — food photos processed through a CNN
- **Regional metadata** — location/cuisine context

...into a single multitask model that predicts sentiment, category, and rating jointly, with an attention mechanism to help interpret which modality drives each prediction.

## Features

- **4-way ablation comparison** — isolates the contribution of each modality (Image Only, Text Only, Image+Text, full multimodal fusion) using Accuracy, Precision, Recall, and F1-score.
- **Multitask prediction head** — sentiment, category, and rating predicted jointly rather than as separate models.
- **Attention-based interpretability** — surfaces which modality (and which parts of it) most influenced a given prediction.
- **Interactive Streamlit app** — quick-load demo scenarios, live inference on sample inputs, and auto-generated business insights per prediction.
- **Regional/cuisine analytics** — explores how sentiment patterns vary by location and cuisine type.

## Tech Stack

| Layer | Tools |
|---|---|
| Deep Learning | PyTorch, TensorFlow/Keras |
| NLP | spaCy, VADER, scikit-learn |
| Computer Vision | CNN (PyTorch/torchvision) |
| Topic Modeling | CorEx |
| App/UI | Streamlit, Plotly, Seaborn |
| Data | Pandas, NumPy |

## Project Structure

```
YelpDL/
├── app.py              # Streamlit application — demo UI, inference, insights
├── yelp.py             # Model architecture & training pipeline
├── results.py           # Evaluation, metrics, ablation comparison
├── environment.yml      # Conda environment spec
├── requirements.txt     # Python dependencies
├── outputs/              # Saved results / model outputs
└── .devcontainer/        # Dev container config
```

## Getting Started

```bash
# Clone the repo
git clone https://github.com/divyakapoor23/YelpDL.git
cd YelpDL

# Set up environment
conda env create -f environment.yml
# — or —
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Model Approach

The fusion model combines modality-specific encoders (CNN for images, NLP pipeline for text) with a learned attention layer that weighs each modality's contribution before the joint multitask prediction head. Ablation variants strip out individual modalities to isolate their marginal contribution, evaluated head-to-head against the full fusion model on Accuracy, Precision, Recall, and F1.

## Status

Core model, ablation pipeline, and Streamlit demo are functional. Ongoing work includes expanding the regional/cuisine dataset coverage and refining the attention visualization.

## Author

**Divya Kapoor** — [GitHub](https://github.com/divyakapoor23) · [LinkedIn](https://www.linkedin.com/in/d-kapoor/)
