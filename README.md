
# Mental Health News Analysis using NLP and Machine Learning

This project explores the landscape of mental health–related news coverage using Natural Language Processing (NLP) and machine learning techniques. It classifies articles, analyzes sentiment trends, and uncovers latent themes to better understand how mental health topics are portrayed in the media.

## Project Overview

Mental health issues like depression, anxiety, PTSD, and suicidal ideation are widely covered in digital media. Using data collected from thousands of news articles across reputable sources (CNN, Newsweek, New York Post, etc.), this project offers insights into:
- Keyword trends and publishing frequency
- Sentiment polarity and emotional framing
- Article classification using BERT-based models
- Thematic patterns using clustering and topic modeling

## Objectives

- Collect articles using targeted mental health keywords
- Clean and preprocess the text for modeling
- Perform sentiment analysis and exploratory data analysis (EDA)
- Build classification models to predict article categories
- Discover themes using clustering (KMeans, DBSCAN) and LDA topic modeling

## Dataset

- **Source**: GNews API  
- **Size**: ~6,000 articles  
- **Fields**: Title, Description, Full Content, Source, URL, Timestamp, Keyword Label  
- **Keywords**: depression, anxiety, PTSD, bipolar, stress, suicidal, addiction, suicide  

## Preprocessing Steps

- Removed duplicates using URL/title/content matching
- Filtered out underrepresented keywords
- Cleaned HTML tags, URLs, punctuation
- Merged `title + description + content` into a single text column
- Created sentiment labels using TextBlob polarity score

## Exploratory Data Analysis (EDA)

- Keyword and source frequency analysis
- Monthly trends from 2021 to 2025
- Sentiment distribution (positive, negative, neutral)
- Time-series analysis showing media focus shifts over time

## Model Building

### Supervised Learning (Classification)

- **Embeddings**: BERT (all-mpnet-base-v2)
- **Models**:
  - Logistic Regression
  - Random Forest
  - Neural Network (Keras)

| Model             | Accuracy | Macro F1 | ROC-AUC |
|------------------|----------|----------|---------|
| LogisticRegression | 88%      | 0.88     | 0.9828  |
| RandomForest      | 86%      | 0.86     | 0.9736  |
| Neural Network    | 82%      | 0.81     | 0.8920  |

### Unsupervised Learning (Clustering & Topics)

- **KMeans (K=6)**: Low silhouette score (~0.03), overlapping clusters
- **DBSCAN**: High noise, poor cluster separability
- **LDA Topic Modeling**:
  - Topic 1: Anxiety, stress
  - Topic 2: Bipolar, diagnosis
  - Topic 3: Depression, suicidal
  - Topic 4: Politics, stress
  - Topic 5: Suicidal, PTSD
  - Topic 6: Depression, PTSD, study

## Tools & Libraries

- Python, Pandas, NumPy, Scikit-learn
- BERT (sentence-transformers)
- Keras / TensorFlow
- TextBlob
- Matplotlib & Seaborn

## Key Insights

- Logistic Regression was most effective for keyword classification
- Suicidal and bipolar articles had highly distinctive patterns
- Articles on stress often overlapped semantically with anxiety and depression
- News focus increased notably from 2023 onward
- Media often frames mental health with neutral or mildly negative tone

## Repository Contents

- `DataMining_final_project.ipynb`: Jupyter notebook with full pipeline
- `DataMining_Final_Report.pdf`: Detailed project report
- `DataMining Final Presentation.pptx`: Summary slides
- `gnews_mental_health.csv`: Preprocessed dataset used for modeling

## Conclusion

This project offers a deep dive into mental health coverage in online media using interpretable NLP and ML methods. It demonstrates how language models, combined with statistical learning, can surface hidden patterns and help public health researchers, media analysts, and policy makers better understand how mental health is represented in the news.

## References

- Devlin et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
- TextBlob Documentation: https://textblob.readthedocs.io/en/dev/
- Scikit-learn: https://scikit-learn.org/
- Sentence Transformers: https://www.sbert.net/
- Keras: https://keras.io/
