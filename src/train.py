# Azure ML Training Script
import argparse
import os
from pathlib import Path

# Azure ML imports
import mlflow
import mlflow.sklearn

# ML imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.datasets import load_iris


def main():
    """Main training function"""
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_estimators', type=int, default=100, help='Number of trees')
    parser.add_argument('--max_depth', type=int, default=5, help='Max tree depth')
    args = parser.parse_args()
    
    print("🚀 Starting training...")
    
    # Enable autologging
    mlflow.sklearn.autolog()
    
    # Load data
    print("📊 Loading data...")
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name='species')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Train model
    print("🎯 Training model...")
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    
    print(f"✅ Training complete!")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    
    # MLflow will automatically log model and metrics due to autolog


if __name__ == "__main__":
    main()
