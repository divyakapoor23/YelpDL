# YelpDL — Multimodal Yelp Sentiment Analysis

This is a class project for my Deep Learning course. The idea was to go beyond just using text for sentiment analysis and also bring in photos and location information from Yelp to see if that makes predictions better.

The main question I was trying to answer:

> How do visual, textual, and regional signals interact to shape perceived food sentiment, and does incorporating geographic context improve multimodal sentiment prediction?

Basically — does showing a model a food photo AND a review AND where the restaurant is located help it understand if the experience was positive or negative better than any single signal alone?

---

## What the Project Does

- Compares 4 model variants in an ablation study: image only, text only, image + text, and image + text + region.
- Uses a pretrained ResNet18 for images and an LSTM for text.
- Adds cross-modal attention so the model can learn which modality to trust more.
- Looks at cases where the image and review disagree (I called this the "perception gap").
- Does some analysis on which regions/cuisines have the most disagreement.
- Runs cross-modal retrieval (find images from text descriptions and vice versa).
- Generates a bunch of plots and a Streamlit dashboard to explore everything interactively.

---

## Project Structure

```
YelpDL/
  yelp.py              # main training + analysis script
  app.py               # Streamlit dashboard
  environment.yml      # conda environment
  outputs/             # all generated CSVs and plots end up here
  Data/
    Yelp JSON/
      yelp_dataset/
        yelp_academic_dataset_business.json
        yelp_academic_dataset_review.json
        yelp_academic_dataset_checkin.json
        yelp_academic_dataset_tip.json
        yelp_academic_dataset_user.json
    Yelp Photos/
      yelp_photos/
        photos.json
        photos/          # JPEG files named <photo_id>.jpg
```

---

## Getting the Data

You need to download the Yelp Open Dataset yourself from https://www.yelp.com/dataset and place the files in the exact folder structure above. The dataset is not included in this repo because it's too large.

The files the code actually needs are:
- `Data/Yelp JSON/yelp_dataset/yelp_academic_dataset_business.json`
- `Data/Yelp JSON/yelp_dataset/yelp_academic_dataset_review.json`
- `Data/Yelp Photos/yelp_photos/photos.json`
- `Data/Yelp Photos/yelp_photos/photos/` (the actual photo JPEGs)

---

## Setup

I used conda to manage the environment. To recreate it:

```bash
conda env create -f environment.yml
conda activate YelpDL
```

If you want to use it in a Jupyter notebook too:

```bash
python -m ipykernel install --user --name YelpDL --display-name "Python (YelpDL)"
```

---

## Running the Training Pipeline

```bash
conda activate YelpDL
python -u yelp.py
```

This will take a while depending on your machine. It trains all 4 model variants and then runs all the analyses. Everything gets saved into the `outputs/` folder.

> **Note:** I tested this on a Mac with Apple Silicon (MPS). It should also work with CUDA or CPU but I haven't verified all three personally.

---

## Running the Dashboard

Once `yelp.py` has finished and the `outputs/` folder has files in it:

```bash
streamlit run app.py
```

Open the URL it prints (usually `http://localhost:8501`). The dashboard has these tabs:

| Tab | What it shows |
|-----|--------------|
| Overview | Ablation results, F1 bar chart |
| Attention | Cross-modal attention weight distributions |
| Retrieval | Text→Image and Image→Text nearest neighbors |
| Consistency | Perception gap analysis by region/cuisine |
| Text | TF-IDF keyword importance |
| Visual | t-SNE and PCA of CNN embeddings |
| Region + Quality | Regional sentiment rates, data quality subgroups |
| Research Insights | Auto-generated takeaways from the results |

---

## Outputs Generated

After a successful run, you should see files like these in `outputs/`:

**CSVs:**
- `ablation_results.csv` — F1/accuracy for all 4 models
- `attention_Image_plus_Text.csv` / `attention_Image_plus_Text_plus_Region.csv`
- `image_text_consistency_predictions.csv` — row-level mismatch labels
- `image_text_consistency_region_summary.csv`
- `image_text_consistency_cuisine_summary.csv`
- `image_text_consistency_rating_quality_summary.csv`
- `retrieval_text_to_image.csv` / `retrieval_image_to_text.csv`
- `text_tfidf_term_scores.csv`
- `visual_features_tsne.csv`
- `region_sentiment_stats.csv`
- `data_quality_noise_analysis.csv`
- `cuisine_region_sentiment_spread.csv`

**Final presentation plots:**
- `final_model_f1_bar.png`
- `final_best_model_confusion_matrix.png`
- `final_perception_gap_by_region.png`
- `final_perception_gap_by_cuisine.png`
- `final_attention_comparison.png`
- `final_region_cuisine_spread.png`

---

## Known Limitation

The photo-to-review matching is done at the **business level**, not at the dish or visit level. So a photo of the restaurant exterior might get paired with a review about the pasta. This is a pretty big limitation and something I'd want to fix if I had more time (maybe by using photo labels from the dataset).

---

## Training Config (defaults in `yelp.py`)

| Setting | Value |
|---------|-------|
| Epochs | 10 |
| Batch size | 64 (GPU/MPS), 32 (CPU) |
| Learning rate | 1e-4 |
| Early stopping patience | 3 epochs |
| Max reviews loaded | 120,000 |

---

## If Something Breaks

- **No charts in the app** → Run `yelp.py` first, the app only reads from `outputs/`.
- **Import errors** → Make sure you activated the `YelpDL` conda environment.
- **Running too slow** → Lower `MAX_REVIEWS` and `MAX_PHOTOS` at the top of `yelp.py` to use a smaller sample.
