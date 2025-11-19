# 🧠 Fake News Detection — Project Summary

Fake news refers to intentionally misleading or false information presented as real news. Because it often resembles legitimate journalism, detecting it automatically is a challenging task.  
This project implements **multiple machine learning and deep learning models** to classify news as **Real** or **Fake**. It also uses an **ensemble voting technique** for improved reliability.

---

## 📌 Dataset
**[Kaggle Fake News Detection Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)**
Contains labeled news articles from multiple sources with two labels:

- **0 — Real**
- **1 — Fake**

---

## 🚀 Models Implemented

### **A) TF-IDF + Logistic Regression**
A classical ML baseline:
- Converts text into numerical features using **TF-IDF vectors**
- Trains a **Logistic Regression** classifier
- Fast, lightweight, and highly interpretable  
**Achieved accuracy: 0.98**

---

### **B) LSTM Neural Network**
A deep learning model designed to capture **sequential relationships** and **word order**:
- Uses tokenization + padded sequences  
- Embedding layer + stacked LSTMs  
- Trained with **EarlyStopping** to avoid overfitting  
**Achieved accuracy: 0.98**

---

### **C) Pretrained BERT (HuggingFace)**
A transformer-based language model tested in **inference mode** (no fine-tuning):
- Loaded using `transformers.pipeline("text-classification")`
- Model used: **[`dhruvpal/fake-news-bert`](https://huggingface.co/dhruvpal/fake-news-bert)**
- Demonstrated excellent contextual understanding  
**Achieved accuracy: 0.99**

---

## 🎯 Weighted Soft Voting Ensemble

A technique that combines the probabilities of multiple models.  
Each model contributes according to a **weight**, giving stronger models more influence.

This ensemble combines:

- **TF-IDF + Logistic Regression**
- **LSTM Neural Network**

Final prediction = weighted average of model probabilities for fake and real class,  
leading to smoother and more stable results than relying on one model alone.

---

## 🌐 Web Interface (FastAPI + HTML/CSS)

A lightweight web demo was built to showcase real-time predictions.

### Backend
✔ Implemented using **FastAPI**  
✔ Loads the simplest trained Logistic Regression TF-IDF model  
✔ Returns prediction + confidence score in JSON format  

### Frontend
✔ Clean HTML/CSS interface  
✔ Paste news → get prediction instantly  
✔ Outputs Real or Fake with Confidence Score.

---

## ✅ Summary

This project demonstrates a full workflow using multiple types of Models for Fake News Detection:

- Text cleaning and preprocessing  
- Classical ML, Deep Learning (LSTM), and Transformer-based models  
- Weighted ensemble for higher reliability  
- Deployed demo using FastAPI + modern frontend  

It showcases both practical ML engineering and modern NLP techniques.
