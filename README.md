# Fraud Detection Project

This repository contains machine learning and deep learning models developed for fraud detection. The project is structured into two main parts: traditional ML models and DL-based approaches.

---

## Project Structure

```
Fraud_Detection/
├── DL_models
│   ├── Details
│   │   └── Details.docx
│   ├── Graphs
│   │   ├── cnn_confusion_matrix_percent.png
│   │   ├── graphs.ipynb
│   │   ├── lstm_confusion_matrix_percent.png
│   │   └── metrics_comparison.png
│   ├── models
│   │   ├── CNN_model.h5
│   │   └── LSTM_model.h5
│   └── Train_Scripts
│       ├── CNN.py
│       └── lstm.py
├── ML_Models
│   ├── ML_Models
│   │   ├── LogisticRegression_model.pkl
│   │   ├── RandomForest_model.pkl
│   │   └── XGBoost_model.pkl
│   ├── Notebook
│   │   └── Fraud_Detection.ipynb
│   ├── Plots
│   │   ├── class_distribution.png
│   │   ├── LogisticRegression_confusion_matrix_percent.png
│   │   ├── model_performance_comparison.png
│   │   ├── RandomForest_confusion_matrix_percent.png
│   │   └── XGBoost_confusion_matrix_percent.png
│   └── Top_Features
│       ├── top_features_mean_importance.png
│       └── top_feature_importance.csv
├── dataset/
│   ├── fraudTest.csv
│   ├── fraudTrain.csv
```

---

## Key Components

### DL_models
- **Train_Scripts**: Python scripts to train CNN and LSTM models.
- **models**: Saved `.h5` models.
- **Graphs**: Visualization of confusion matrices and performance.
- **Details**: Project documentation and model details.

### ML_Models
- **ML_Models**: Pickled models including Logistic Regression, Random Forest, and XGBoost.
- **Notebook**: Main Jupyter notebook for preprocessing, training, and evaluation.
- **Plots**: Visuals of performance metrics and class distributions.
- **Top_Features**: Feature importance analysis.

### dataset
- Sparkov data 
---

## Getting Started

1. Clone the repository:

2. Navigate to the project folder:

3. Run the notebooks or scripts as needed.

---

## Results

- Evaluation of ML and DL models using confusion matrices and performance metrics.
- Comparison of algorithm accuracy and feature importance.
- Visual analytics for improved interpretability.

---

## Models Used

- **Machine Learning**: Logistic Regression, Random Forest, XGBoost
- **Deep Learning**: CNN, LSTM

---

## Author

**Shahbaj Ahmad**  
