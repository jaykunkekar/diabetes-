import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('diabetes.csv')

# Display basic information
print(f"Dataset shape: {df.shape}")
print(f"Column names: {df.columns.tolist()}")

# Separate features and target
X = df.drop(['PatientID', 'Diabetic'], axis=1)
y = df['Diabetic']

print(f"\nFeatures: {X.columns.tolist()}")
print(f"Target distribution:\n{y.value_counts()}")

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Validation set size: {X_val.shape[0]}")

# Train Random Forest Classifier
print("\nTraining Random Forest Classifier...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

# Make predictions
y_train_pred = rf_model.predict(X_train)
y_val_pred = rf_model.predict(X_val)

# Evaluate the model
train_accuracy = accuracy_score(y_train, y_train_pred)
val_accuracy = accuracy_score(y_val, y_val_pred)
val_precision = precision_score(y_val, y_val_pred)
val_recall = recall_score(y_val, y_val_pred)
val_f1 = f1_score(y_val, y_val_pred)

print(f"\n--- Model Performance ---")
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Validation Precision: {val_precision:.4f}")
print(f"Validation Recall: {val_recall:.4f}")
print(f"Validation F1-Score: {val_f1:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"\n--- Feature Importance ---")
print(feature_importance.to_string(index=False))

# Save the model
model_path = 'diabetes_model.joblib'
joblib.dump(rf_model, model_path)
print(f"\nModel saved to {model_path}")

# Save feature names for later use
feature_names_path = 'feature_names.joblib'
joblib.dump(X.columns.tolist(), feature_names_path)
print(f"Feature names saved to {feature_names_path}")
