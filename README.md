# Email Spam Detection using Apache Spark

## Project Overview
This project implements an email spam detection system using Big Data technologies and machine learning techniques. The system processes email/SMS data and classifies messages as spam or ham.

---

## Technologies Used
- Apache Spark  
- Scala  
- PySpark (Databricks / Google Colab)  
- MLlib  

---

## Features
- Data preprocessing (cleaning, tokenization, stopword removal)  
- Feature extraction using TF-IDF  
- Classification using Logistic Regression and Naive Bayes  
- Model evaluation using accuracy  
- Data visualization (Spam vs Ham, Message Length, Word Frequency)  

---

## Dataset Details
The dataset used is the SMS Spam Collection Dataset.

- v1 → Label (spam / ham)  
- v2 → Message text  

---

## Model Output
The model classifies emails into:
- 0 → Ham (Non-Spam)  
- 1 → Spam  

Example output:
Accuracy: 1.0  

---

## Visualizations Included
- Spam vs Ham Distribution  
- Average Message Length Analysis  
- Message Length Distribution  
- Word Frequency Analysis  

---


## Project Structure
email-spam-detection/
│
├── build.sbt  
├── README.md  
└── src/
    └── main/
        └── scala/
            └── SpamDetection.scala  

---

## Future Scope
The system can be enhanced using advanced NLP and deep learning models for improved accuracy. Real-time deployment and better evaluation techniques such as train-test split can further improve performance.

---

## Author
Priyanshi Varshney
