# Script 2: CNN Model for Fraud Detection
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from category_encoders import TargetEncoder
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import warnings
import os

warnings.filterwarnings("ignore")

# Function to Train & Evaluate CNN Model
def train_and_evaluate_cnn(model, name, X_train, y_train, X_test, y_test, tune=False):
    try:
        if tune:
            # Manual tuning (grid search impractical for neural networks)
            learning_rates = [0.0001, 0.0005]
            best_model = None
            best_f1 = 0
            for lr in learning_rates:
                temp_model = Sequential([
                    Conv1D(32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
                    BatchNormalization(),
                    Conv1D(64, kernel_size=3, activation='relu'),
                    Flatten(),
                    Dense(64, activation='relu'),
                    Dropout(0.5),
                    Dense(1, activation='sigmoid')
                ])
                temp_model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])
                temp_model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test), verbose=0)
                y_pred_proba = temp_model.predict(X_test, verbose=0)
                y_pred = (y_pred_proba > 0.5).astype(int).flatten()
                f1 = f1_score(y_test, y_pred)
                if f1 > best_f1:
                    best_f1 = f1
                    best_model = temp_model
                    print(f"Tuning {name}: Learning rate {lr} - F1 Score: {f1:.4f}")
            model = best_model
        else:
            model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test), verbose=1)

        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        print(f"\n=== {name} Model Results ===")
        print(f"{'Metric':<20} {'Value':>10}")
        print("-" * 35)
        print(f"{'Accuracy':<20} {accuracy:>10.4f}")
        print(f"{'Precision':<20} {precision:>10.4f}")
        print(f"{'Recall':<20} {recall:>10.4f}")
        print(f"{'F1 Score':<20} {f1:>10.4f}")
        print(f"{'ROC AUC Score':<20} {roc_auc:>10.4f}")
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(f"{'True Negatives (TN)':<20}: {cm[0, 0]}")
        print(f"{'False Positives (FP)':<20}: {cm[0, 1]}")
        print(f"{'False Negatives (FN)':<20}: {cm[1, 0]}")
        print(f"{'True Positives (TP)':<20}: {cm[1, 1]}")

        try:
            filename = f"{name}_model.h5"
            model.save(filename)
            if os.path.exists(filename):
                print(f"\nModel Saved: {os.path.abspath(filename)}")
            else:
                print(f"\nFailed to confirm {name} model was saved.")
        except Exception as e:
            print(f"\nError saving {name} model: {e}")

        print("=" * 35 + "\n")
        return accuracy, precision, recall, f1, roc_auc
    except Exception as e:
        print(f"\nError training {name}: {e}")
        return None, None, None, None, None

# Define data types to optimize memory usage
dtype_dict = {
    'trans_num': 'category',
    'merchant': 'category',
    'category': 'category',
    'amt': 'float32',
    'gender': 'category',
    'city': 'category',
    'state': 'category',
    'lat': 'float32',
    'long': 'float32',
    'zip': 'int32',
    'job': 'category',
    'dob': 'category',
    'trans_date_trans_time': 'category',
    'cc_num': 'int64',
    'unix_time': 'int64',
    'merch_lat': 'float32',
    'merch_long': 'float32',
    'is_fraud': 'int8'
}

# Load train and test data
df_train = pd.read_csv("FraudTrain.csv", dtype=dtype_dict)
df_test = pd.read_csv("FraudTest.csv", dtype=dtype_dict)

# Preprocess data
def preprocess_data(data):
    columns_to_drop = ['trans_date_trans_time', 'cc_num', 'merchant', 'first', 'last', 'street', 'dob', 'trans_num', 'unix_time']
    data = data.drop(columns=columns_to_drop, errors='ignore')

    categorical_cols = ['category', 'gender', 'city', 'state', 'job']
    target_encoder = TargetEncoder()

    for col in categorical_cols:
        if col in data.columns:
            data[col] = target_encoder.fit_transform(data[col], data['is_fraud'])

    return data

df_train = preprocess_data(df_train)
df_test = preprocess_data(df_test)

# Use a smaller subset for CNN to manage memory
subset_size = 20000
df_train_subset = df_train.sample(n=subset_size, random_state=42)

# Balance dataset using SMOTE on the subset
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(df_train_subset.drop(columns=['is_fraud']), df_train_subset['is_fraud'])
X_test, y_test = df_test.drop(columns=['is_fraud']), df_test['is_fraud']

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape data for CNN
X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Define and Train CNN Model
cnn_model = Sequential([
    Conv1D(32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    BatchNormalization(),
    Conv1D(64, kernel_size=3, activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
cnn_model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])

# Train and Evaluate CNN Model
results = train_and_evaluate_cnn(cnn_model, "CNN", X_train_cnn, y_train, X_test_cnn, y_test, tune=False)  # Set tune=True for tuning
accuracy = results[0] if results else None

# Plot model accuracy (single model)
if accuracy is not None:
    plt.figure(figsize=(6, 4))
    sns.barplot(x=["CNN"], y=[accuracy])
    plt.title("CNN Model Accuracy")
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.show()