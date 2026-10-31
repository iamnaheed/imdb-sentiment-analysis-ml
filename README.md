# Sentiment Analysis of Movie Reviews using Machine Learning

**A supervised text-classification study on the IMDb Movie Reviews dataset using TF-IDF features and Logistic Regression**

![Python](https://img.shields.io/badge/Python-3.11-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.1-orange)
![Status](https://img.shields.io/badge/Status-Academic%20Project-green)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Motivation](#2-motivation)
3. [Research Objectives](#3-research-objectives)
4. [Dataset](#4-dataset)
5. [Methodology](#5-methodology)
6. [Model Description](#6-model-description)
7. [Evaluation Metrics](#7-evaluation-metrics)
8. [Results](#8-results)
9. [Error Analysis and Limitations](#9-error-analysis-and-limitations)
10. [Future Work](#10-future-work)
11. [Project Structure](#11-project-structure)
12. [Installation and Usage](#12-installation-and-usage)
13. [Technologies Used](#13-technologies-used)
14. [Learning Outcomes](#14-learning-outcomes)
15. [References](#15-references)
16. [Author](#16-author)

---

## 1. Project Overview

Sentiment analysis is a core task in **Natural Language Processing (NLP)**. Its goal is to work out the emotional tone, or *opinion polarity*, of a piece of text automatically. This project builds and evaluates a complete machine learning pipeline that classifies movie reviews as **Positive** or **Negative**.

The project uses the publicly available **IMDb Large Movie Review Dataset** of 50,000 labelled reviews. Raw text is turned into numerical features with **TF-IDF (Term Frequency–Inverse Document Frequency)**, and a **Logistic Regression** classifier is trained on those features. Performance is measured on a held-out test set using standard classification metrics: precision, recall, F1-score, and accuracy.

This is an educational project with three aims:

- show the end-to-end workflow of a classical NLP classification system;
- set a strong, interpretable **baseline** for later comparison with deep learning methods;
- critically analyse where traditional bag-of-words methods fall short, which leads into future research on contextual language models.

---

## 2. Motivation

Every day, people write very large amounts of opinion text: product reviews, social media posts, customer feedback, and film critiques. No human team can read all of it. Automated sentiment analysis lets organisations:

- monitor public opinion and brand reputation;
- summarise customer feedback at scale;
- support recommendation systems and market research.

Movie reviews make a good benchmark for this problem. They are long, varied in vocabulary, and often contain mixed opinions, sarcasm, or contrasting statements. That makes them a realistic and challenging test for text classifiers.

---

## 3. Research Objectives

The project addresses the following objectives:

1. **Build** a reproducible machine learning pipeline for binary sentiment classification.
2. **Represent** unstructured text as numerical feature vectors using TF-IDF.
3. **Train** a Logistic Regression classifier and measure how well it generalises to unseen data.
4. **Evaluate** the model with precision, recall, F1-score, and accuracy.
5. **Analyse** the kinds of reviews the model misclassifies and identify the limits of the approach.
6. **Propose** directions for improvement, such as word embeddings and Transformer models.

---

## 4. Dataset

| Property            | Description                                                         |
|---------------------|---------------------------------------------------------------------|
| **Name**            | IMDb Dataset of 50K Movie Reviews                                   |
| **Source**          | [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) (derived from Maas et al., 2011) |
| **Size**            | 50,000 reviews                                                      |
| **Classes**         | 2: `positive`, `negative`                                           |
| **Class balance**   | Balanced, 25,000 reviews per class                                  |
| **Columns**         | `review` (raw text), `sentiment` (label)                            |
| **Format**          | CSV                                                                 |

**Why this dataset?**
It is a widely used benchmark in NLP research, so results can be compared with published work. Because the classes are balanced, accuracy is a meaningful metric and no resampling is needed.

> **Note:** The dataset is not included in this repository because of its size. Download `IMDB Dataset.csv` from Kaggle and place it in the `data/` folder (see [Installation and Usage](#12-installation-and-usage)).

---

## 5. Methodology

The pipeline has six sequential stages. Each one is explained below.

```
 Raw CSV  ->  Label Encoding  ->  Train/Test Split  ->  TF-IDF Vectorisation  ->  Logistic Regression  ->  Evaluation
```

### Step 1 — Data Loading

The dataset is read into a **Pandas DataFrame** with `pd.read_csv()`. Each row holds a review and its sentiment label. Pandas makes it easy to inspect the data and transform columns.

### Step 2 — Label Encoding

Machine learning algorithms need numerical targets, so the text labels are mapped to binary values:

| Original label | Encoded value |
|----------------|---------------|
| `positive`     | `1`           |
| `negative`     | `0`           |

### Step 3 — Train–Test Split

The data is split into two separate subsets with `train_test_split`:

- **Training set (80% — 40,000 reviews):** used to learn the model parameters.
- **Test set (20% — 10,000 reviews):** kept unseen during training and used only for the final evaluation.

A fixed `random_state=42` makes the split **reproducible**, so the same results can be obtained on every run. Splitting *before* feature extraction prevents **data leakage**, meaning no information from the test set reaches the training process.

### Step 4 — Text Preprocessing and Feature Extraction (TF-IDF)

Text cannot go straight into a classifier. It must first become a numerical vector. This project uses scikit-learn's `TfidfVectorizer`, which performs the following preprocessing internally:

- **Lowercasing**, so "Great" and "great" count as the same word;
- **Tokenisation**, splitting text into individual words;
- **Vocabulary building**, keeping only the **5,000 most frequent terms** (`max_features=5000`) to control dimensionality and reduce noise.

Each word is then weighted with **TF-IDF**:

$$
\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \log\left(\frac{N}{\text{DF}(t)}\right)
$$

where:

- **TF(t, d)** is how often term *t* appears in document *d*;
- **N** is the total number of documents;
- **DF(t)** is the number of documents that contain term *t*.

**Intuition:** words that appear often in one review but rarely across the whole corpus (e.g. *"masterpiece"*, *"boring"*) get **high weights**. Very common words (e.g. *"the"*, *"movie"*) get **low weights**. The result is a sparse feature matrix in which each review is a 5,000-dimensional vector.

> The vectoriser is **fitted only on the training data** and then used to **transform** the test data. This is essential for a fair, leakage-free evaluation.

### Step 5 — Model Training

A **Logistic Regression** classifier is trained on the TF-IDF vectors of the training set. The model learns one weight per vocabulary term, which shows how strongly that term points to positive or negative sentiment. See [Section 6](#6-model-description) for details.

### Step 6 — Model Evaluation

The trained model predicts sentiment labels for the 10,000 unseen test reviews. The predictions are compared with the true labels, and a full **classification report** is produced (see [Section 7](#7-evaluation-metrics)).

---

## 6. Model Description

### Logistic Regression

Logistic Regression is a linear model for binary classification. It estimates the probability that a review is positive by applying the **sigmoid function** to a weighted sum of the input features:

$$
P(y = 1 \mid \mathbf{x}) = \sigma(\mathbf{w}^\top \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^\top \mathbf{x} + b)}}
$$

If the probability is 0.5 or higher, the review is classified as **Positive**. Otherwise it is classified as **Negative**.

### Why Logistic Regression?

| Advantage              | Explanation                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| **Strong baseline**    | Performs very well on high-dimensional, sparse text features.               |
| **Interpretability**   | Each learned weight is tied to a word, so you can see which words drive each prediction. |
| **Efficiency**         | Trains in seconds on a standard CPU, with no GPU required.                  |
| **Probabilistic output** | Gives confidence scores, not just hard labels.                            |
| **Regularisation**     | Uses L2 regularisation by default, which reduces overfitting.               |

---

## 7. Evaluation Metrics

The model is evaluated with scikit-learn's `classification_report`. With **TP** = true positives, **TN** = true negatives, **FP** = false positives, and **FN** = false negatives:

| Metric        | Formula                                   | Meaning                                                              |
|---------------|-------------------------------------------|----------------------------------------------------------------------|
| **Accuracy**  | (TP + TN) / (TP + TN + FP + FN)           | Overall share of correct predictions.                                |
| **Precision** | TP / (TP + FP)                            | Of the reviews predicted positive, how many really were positive.    |
| **Recall**    | TP / (TP + FN)                            | Of the reviews that really were positive, how many the model found.  |
| **F1-Score**  | 2 × (Precision × Recall) / (Precision + Recall) | Harmonic mean of precision and recall; balances the two.       |

Because the dataset is **balanced**, accuracy is a reliable headline metric. Precision, recall, and F1 still matter because they show whether the model favours one class over the other.

---

## 8. Results

Run `python model/train_model.py` and record the classification report printed to the console in the table below.

| Class        | Precision | Recall | F1-Score | Support |
|--------------|-----------|--------|----------|---------|
| Negative (0) |           |        |          |         |
| Positive (1) |           |        |          |         |
| **Accuracy** |           |        |          | 10,000  |

**Expected performance:** published benchmarks for TF-IDF with Logistic Regression on this dataset typically report **about 88–90% accuracy**. That is competitive for a classical, non-deep-learning approach and confirms TF-IDF with Logistic Regression as a strong baseline.

---

## 9. Error Analysis and Limitations

The model performs well overall, but the TF-IDF representation has known structural weaknesses.

### 9.1 Loss of word order and context

TF-IDF is a **bag-of-words** model. It treats a review as an unordered collection of words and ignores grammar and sequence. So these two sentences can look almost identical to the model:

- *"The movie was **not good**, it was bad."*
- *"The movie was **not bad**, it was good."*

### 9.2 Sarcasm and irony

Sarcastic reviews often use positive words to express negative sentiment, for example: *"Oh great, another two hours of my life I'll never get back."* The model sees *"great"* and may predict Positive.

### 9.3 Contrasting and mixed opinions

Reviews whose sentiment changes partway through are hard to classify:

> *"I expected this movie to be bad... but it was surprisingly good."*

The early negative words (*"expected"*, *"bad"*) compete with the later positive conclusion, and the model has no way to know the final clause matters most.

### 9.4 Fixed vocabulary

Limiting the vocabulary to 5,000 terms means rare but highly informative words, misspellings, and new slang are ignored.

### 9.5 No semantic similarity

TF-IDF treats *"excellent"* and *"outstanding"* as completely unrelated features, even though they mean nearly the same thing.

---

## 10. Future Work

The limitations above point to several research directions:

1. **N-gram features:** add bigrams and trigrams (`ngram_range=(1, 2)`) to capture short phrases such as *"not good"*.
2. **Advanced preprocessing:** remove HTML tags (e.g. `<br />`), handle stop words, and apply lemmatisation.
3. **Hyperparameter tuning:** use `GridSearchCV` with k-fold cross-validation to tune the regularisation strength `C` and the vocabulary size.
4. **Model comparison:** benchmark against Naive Bayes, Linear SVM, and Random Forest.
5. **Word embeddings:** use dense semantic representations such as **Word2Vec**, **GloVe**, or **FastText**.
6. **Deep learning:** try sequence models such as **LSTM** and **BiLSTM**, which keep word order.
7. **Transformer models:** fine-tune pre-trained contextual models such as **BERT**, **RoBERTa**, or **DistilBERT**, which handle negation, contrast, and context far better.
8. **Explainability:** use **LIME** or **SHAP** to explain individual predictions.
9. **Deployment:** serve the trained model as a web application (e.g. Flask or Streamlit) for real-time predictions.

---

## 11. Project Structure

```
Sentiment-Analysis-ML/
│
├── data/
│   └── IMDB Dataset.csv      # Dataset (download from Kaggle)
│
├── model/
│   └── train_model.py        # Full pipeline: load, split, vectorise, train, evaluate
│
├── requirements.txt          # Pinned Python dependencies
└── README.md                 # Project documentation
```

---

## 12. Installation and Usage

### Prerequisites

- Python 3.9 or later (tested with Python 3.11)
- pip

### Step 1 — Get the project

```bash
git clone <repository-url>
cd Sentiment-Analysis-ML
```

### Step 2 — Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Add the dataset

Download **IMDB Dataset.csv** from [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) and save it as:

```
data/IMDB Dataset.csv
```

### Step 5 — Train and evaluate the model

Run this from the project root folder:

```bash
python model/train_model.py
```

The script prints a classification report with precision, recall, F1-score, and accuracy for each class.

---

## 13. Technologies Used

| Tool / Library     | Version | Purpose                                         |
|--------------------|---------|-------------------------------------------------|
| **Python**         | 3.11    | Programming language                            |
| **Pandas**         | 2.2.1   | Data loading and manipulation                   |
| **NumPy**          | 1.26.4  | Numerical computation                           |
| **scikit-learn**   | 1.4.1   | TF-IDF vectorisation, Logistic Regression, metrics |
| **SciPy**          | 1.12.0  | Sparse matrix support for TF-IDF features       |
| **joblib**         | 1.3.2   | Parallel computation used by scikit-learn       |

---

## 14. Learning Outcomes

Through this project I gained practical experience in:

- designing an **end-to-end machine learning pipeline**, from raw data to evaluation;
- **text representation** and feature engineering for NLP;
- applying sound experimental practice: **train/test separation**, **preventing data leakage**, and **reproducibility**;
- **interpreting evaluation metrics** beyond accuracy;
- **critically analysing model errors** and relating them to the model's underlying assumptions;
- identifying open research problems in NLP that motivate further study at **Master's level**, especially contextual language understanding with Transformer models.

---

## 15. References

1. Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011). *Learning Word Vectors for Sentiment Analysis*. Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics (ACL).
2. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825–2830.
3. Pang, B., & Lee, L. (2008). *Opinion Mining and Sentiment Analysis*. Foundations and Trends in Information Retrieval, 2(1–2), 1–135.
4. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL-HLT.
5. Jurafsky, D., & Martin, J. H. *Speech and Language Processing* (3rd ed. draft).

---

## 16. Author

**[Your Full Name]**
Bachelor of [Your Degree], [Your University]

- Email: [your.email@example.com]
- LinkedIn: [linkedin.com/in/your-profile]
- GitHub: [github.com/your-username]

*This project was developed for educational purposes and as part of my portfolio for Master's programme applications in Data Science, Artificial Intelligence, and Machine Learning.*
