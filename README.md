# Fake News Detection Using Text-Mining Models

**Authors**:  
Yuxing Liu – [yul351@ucsd.edu](mailto:yul351@ucsd.edu)  
Yangqi Huang – [yah031@ucsd.edu](mailto:yah031@ucsd.edu)  
Xiaomeng Sun – [rus006@ucsd.edu](mailto:rus006@ucsd.edu)  

## Abstract

This project explores fake news detection using natural language processing (NLP) and machine learning (ML) models. Leveraging a Kaggle dataset with 122,193 labeled news articles, we used TF-IDF, Word2Vec, and BERT embeddings to build classifiers such as Logistic Regression, XGBoost, a Neural Network, and BERT-based models. BERT and XGBoost (with Word2Vec) achieved state-of-the-art performance.

---

## Table of Contents

- [Introduction](#introduction)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Literature Review](#literature-review)
- [Approach](#approach)
- [Experiment](#experiment)
- [Future Suggestions and Conclusion](#future-suggestions-and-conclusion)
- [References](#references)
- [Appendix](#appendix)

---

## Introduction

We used a Kaggle dataset comprising real and fake online news articles. The dataset contains:
- 83,539 training samples (49% fake, 51% real)
- 38,654 test samples (45.18% fake, 54.82% real)

---

## Exploratory Data Analysis

### Text Length Analysis

- Avg. article length: ~402 words
- Bimodal length distribution
- Fake news showed higher variance in length

### Word Cloud Analysis

- Fake news: Emotional/sensational words (e.g., *Trump*, *people*)
- Real news: Factual words (e.g., *Reuters*, *statement*)
- Word frequency alone isn't sufficient—contextual embeddings are required

### Feature Extraction and Model Selection

We used:
- **TF-IDF**
- **Word2Vec**
- **BERT tokenization**

Models tested:
- Logistic Regression
- XGBoost
- Neural Network (PyTorch)
- BERTForSequenceClassification (PyTorch)

---

## Literature Review

- NLP techniques (e.g., sentiment analysis, syntax, fact-checking) are key in fake news detection.
- ML/DL methods like LSTM, BERT, Capsule Networks offer high accuracy.
- Hybrid models (e.g., XGBoost + Word2Vec) provide strong results.
- Context, external knowledge, and interpretability are crucial areas of focus.

---

## Approach

### Text Processing
- Lowercasing
- Removing punctuation/quotes
- Tokenization, stopword removal, lemmatization
- Label encoding (True → 1, False → 0)

### Text Vectorization
- TF-IDF and Word2Vec (average method with imputation)

### Model Training
- Logistic Regression, XGBClassifier, Neural Network, and BERT

### Model Evaluation Metrics
- Accuracy, Precision, Recall, F1 Score

---

## Experiment

### TF-IDF Vectorizer
- Sparse/high-dimensional
- Context-insensitive → lower performance

### Word2Vec
- High accuracy (~96%) with Logistic Regression
- Captures context well

### Logistic Regression
- Word2Vec > TF-IDF
- Grid search improved results slightly

### XGBClassifier
- Very high performance (up to 99% with Word2Vec)
- Tuned with grid search (n_estimators, max_depth, subsample)

### Neural Network
- Feedforward architecture (ReLU + Dropout)
- Trained only on TF-IDF
- Outperformed traditional models

### BERT
- Used `bert-base-uncased` with fine-tuning
- Trained on 20% of dataset due to resource constraints
- Achieved 99.6% accuracy, 99.63% F1 on test set

---

## Future Suggestions and Conclusion

### Future Suggestions

- **Multimodal detection**: Add image/source credibility
- **Adversarial training**: Improve robustness
- **Cross-domain generalization**: Extend beyond traditional news
- **Explainability**: Use SHAP, attention visualization
- **Real-time deployment**: Optimize for speed and scalability

### Conclusion

BERT and XGBoost with Word2Vec showed superior results. Hybrid methods proved highly effective. While deep learning offers high performance, challenges remain around generalizability, interpretability, and real-world deployment.

---

## Results Summary

| Model                        | Accuracy | F1 Score | Precision | Recall  |
|-----------------------------|----------|----------|-----------|---------|
| Logistic Regression (TF-IDF) | 0.4754   | 0.1942   | 0.6152    | 0.1153  |
| Logistic Regression (Word2Vec) | 0.9638 | 0.9670   | 0.9646    | 0.9695  |
| XGBClassifier (TF-IDF)       | 0.4617   | 0.0377   | 0.9466    | 0.0193  |
| XGBClassifier (Word2Vec)     | 0.9987   | 0.9990   | 1.0000    | 0.9979  |
| BERT                         | 0.9960   | 0.9963   | 1.0000    | 0.9927  |
| Neural Network (TF-IDF)      | 0.5290   | 0.3939   | 0.6687    | 0.2792  |

---

## References

Refer to the full paper for citation details including [OQW20], [SGK24], [DCLT19], and others.

---

## Appendix

### Code & Resources

- GitHub Repository: [415_Project](https://github.com/rsm-yah031/415_Project/tree/main)  
- Dataset: [Kaggle - Fake News Detection](https://www.kaggle.com/datasets/sadmansakibmahi/fake-news-detection-dataset-with-pre-trained-model)
