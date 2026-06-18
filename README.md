# Mental Health News Analysis with NLP & Machine Learning

> Multi-class classification of mental health news coverage using sentence-transformers BERT embeddings, plus unsupervised pattern discovery and 5-year sentiment trend analysis.

---

## What This Is

I wanted to understand how mental health is covered in U.S. and international news — across topics like depression, anxiety, PTSD, bipolar, stress, and suicide. So I built an end-to-end NLP pipeline that:

1. Collects articles automatically from the **GNews API**
2. Generates rich semantic embeddings using **sentence-transformers BERT**
3. Trains and compares three supervised classifiers
4. Layers unsupervised methods (K-Means, DBSCAN, LDA) to find themes the labels alone miss
5. Tracks how mental-health coverage tone has shifted over 5 years (2021–2025)

---

## Results at a Glance

| Model | Accuracy | Macro F1 | ROC-AUC |
|---|---|---|---|
| **Logistic Regression** | **88%** | **0.88** | **0.9828** |
| Random Forest | 86% | 0.86 | 0.9736 |
| Keras Neural Network | 82% | 0.81 | 0.8920 |

**The classical model won.** Logistic Regression beat the deep neural network on 8-class categorization with BERT embeddings as input — a useful reminder that more complex models aren't always better, especially when the input representation (BERT) is already doing the heavy lifting.

---

## Dataset

- **~6,000 articles** pulled from the GNews API
- **8 keyword categories**: depression, anxiety, PTSD, bipolar, stress, suicidal, addiction, suicide
- **Time range**: 2021 to 2025
- **Sources**: CNN, Newsweek, New York Post, and other major outlets the API aggregates from
- **Fields**: Title, Description, Full Content, Source, URL, Timestamp, Keyword Label

---

## Approach

### 1. Data collection
Python ingestion script that queries the GNews API across the 8 keywords. Stored articles with title, body, source, publish date, and assigned label.

### 2. Preprocessing
- Removed duplicates using URL / title / content matching
- Filtered out underrepresented keywords
- Cleaned HTML tags, URLs, and punctuation
- Merged `title + description + content` into a single text column
- Added sentiment labels using TextBlob polarity scores

### 3. Embeddings
Used `sentence-transformers/all-mpnet-base-v2` — a sentence-level BERT variant fine-tuned for semantic similarity. Output: 768-dimensional vector per article.

### 4. Supervised classification
Trained and compared three models on the embeddings:
- **Logistic Regression** — fast, interpretable, strong on high-dimensional inputs
- **Random Forest** — ensemble of decision trees
- **Keras Neural Network** — feedforward NN with hidden layers

Evaluated on accuracy, macro F1, ROC-AUC, plus per-class precision/recall and confusion matrices.

### 5. Unsupervised pattern discovery
Tried three methods to surface themes the supervised labels alone wouldn't capture:
- **K-Means** (k=6) — silhouette score ~0.03, very weak separation
- **DBSCAN** — high noise ratio, poor clusters
- **LDA topic modeling** — **worked.** Surfaced 6 coherent latent themes:
  - Topic 1: Anxiety, stress
  - Topic 2: Bipolar, diagnosis
  - Topic 3: Depression, suicidal
  - Topic 4: Politics, stress
  - Topic 5: Suicidal, PTSD
  - Topic 6: Depression, PTSD, study

### 6. Time-series sentiment
Used TextBlob to score sentiment per article, then aggregated monthly across 2021–2025 to visualize how coverage tone shifted.

---

## What Worked vs What Didn't

**Worked:**
- BERT embeddings + classical Logistic Regression — best accuracy in the bunch
- LDA topic modeling for theme discovery
- Time-series sentiment as a coverage-trend signal

**Didn't work as expected:**
- The Keras neural network underperformed Logistic Regression. The embedding space was already linearly separable; adding non-linear capacity didn't help.
- K-Means and DBSCAN both struggled in the 768-D embedding space. Articles overlap too much for distance-based clustering to produce clean groups.

**Takeaway:** Start simple, validate carefully, and let the method match the geometry of the data. LDA worked because it models documents over topics directly, while clustering tried to impose hard partitions on a continuous semantic surface.

---

## Key Insights

- Logistic Regression was the most effective model for keyword classification
- Suicidal and bipolar articles had highly distinctive linguistic patterns
- Articles on stress often overlapped semantically with anxiety and depression
- News focus on mental health increased notably from 2023 onward
- Media often frames mental health with neutral or mildly negative tone

---

## Tech Stack

- **Python 3.11**
- **Pandas, NumPy** — data manipulation
- **scikit-learn** — Logistic Regression, Random Forest, K-Means, DBSCAN, LDA
- **sentence-transformers** — BERT embeddings (`all-mpnet-base-v2`)
- **Keras / TensorFlow** — neural network
- **TextBlob** — sentiment analysis
- **Matplotlib, Seaborn** — visualization
- **GNews API** — data source

---

## Repository Contents

| File | What it is |
|---|---|
| `project.ipynb` | Full Jupyter notebook — data collection, modeling, evaluation, visualization |
| `Report.pdf` | Detailed written report covering methodology and findings |
| `Presentation.pptx` | Summary slide deck |
| `Dataset.csv` | Preprocessed dataset used for modeling |
| `images/` | Charts and visualizations (add confusion matrix, sentiment trend, model comparison here) |

---

## How to Reproduce

```bash
git clone https://github.com/muni-prathap-goud/Mental-Health-News-Analysis-using-NLP-and-Machine-Learning.git
cd Mental-Health-News-Analysis-using-NLP-and-Machine-Learning

# Install dependencies
pip install pandas numpy scikit-learn sentence-transformers tensorflow textblob matplotlib seaborn

# Launch the notebook
jupyter notebook project.ipynb
```

To re-pull fresh data from GNews, set your API key:
```bash
export GNEWS_API_KEY="your_key_here"
```

---

## Future Improvements

- Fine-tune BERT on a smaller domain-specific corpus to see if classification accuracy improves
- Add a Streamlit UI for live article classification
- Expand to a broader keyword set (general mental wellness, therapy, medication)
- Cross-reference with public mental-health-services data to look for journalism-vs-reality gaps

---

*Author: Muniprathap Murari · [github.com/muni-prathap-goud](https://github.com/muni-prathap-goud)*
