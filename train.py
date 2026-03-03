import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import os

# Set MLflow tracking URI (could be a remote server in prod)
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Customer_Churn_Prediction")

def simulate_data(n=1000):
    np.random.seed(42)
    # Generate some dummy data
    age = np.random.randint(18, 70, n)
    tenure = np.random.randint(1, 60, n)
    balance = np.random.uniform(100, 10000, n)
    support_calls = np.random.randint(0, 5, n)
    
    # Calculate churn likelihood based on a formula
    churn_prob = (support_calls * 0.1) + (1 / (tenure + 1)) - (balance * 0.0001) + (age * 0.001)
    # Threshold for churn
    y = [1 if prob > 0.4 else 0 for prob in churn_prob]
    
    # Create DataFrame
    X = pd.DataFrame({'age': age, 'tenure': tenure, 'balance': balance, 'support_calls': support_calls})
    return X, y

if __name__ == "__main__":
    print("Loading data...")
    X, y = simulate_data(5000)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define hyperparameters
    n_estimators = 100
    max_depth = 5
    
    # Start MLflow run
    with mlflow.start_run():
        print("Training model...")
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Log parameters and metrics
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        print(f"Model trained. Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        print(f"Model saved in run: {mlflow.active_run().info.run_id}")
