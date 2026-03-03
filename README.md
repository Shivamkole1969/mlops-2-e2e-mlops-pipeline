# End-to-End MLOps Pipeline Automation

## 🌟 STAR Summary

- **Situation:** Data Science models (like XGBoost or Random Forests) generated locally during experimentation frequently broke in production due to dependency mismatches, and it was nearly impossible to reproduce successful past experiments due to a lack of metric tracking.
- **Task:** Build a scalable, containerized, and fully automated Machine Learning lifecycle pipeline that bridges the gap between proof-of-concept models and reliable production-ready systems.
- **Action:** 
  1. **Experiment Tracking:** Integrated **MLflow** into the training scripts to log hyperparameters, model performance metrics (Accuracy, F1-Score), and serialize binary model artifacts.
  2. **Containerization:** Wrote a **Dockerfile** to package the application and dependencies explicitly to prevent "works on my machine" issues.
  3. **CI/CD Automation:** Engineered a continuous integration and continuous deployment pipeline using **GitHub Actions**. Upon every push or Pull Request, the CI pipeline provisions an environment, triggers model training, logs to MLflow, builds a Docker image, and readies it for deployment.
- **Result:** Cut model deployment times from days to minutes. Ensured 100% reproducibility of all experiments, laying a foundation for scalable, secure ML workloads in a cloud environment.

## 🛠 Tech Stack Used (MLOps Alignment)
*   **Version Control & CI/CD:** Git, GitHub Actions
*   **Experiment Tracking:** MLflow
*   **Containerization:** Docker
*   **Machine Learning:** Scikit-Learn

## 🚀 How to Run Locally

1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the training script manually:
   ```bash
   python train.py
   ```
3. Check the MLflow UI for your logged experiment:
   ```bash
   mlflow ui --backend-store-uri sqlite:///mlflow.db
   ```
4. Build the Docker container locally:
   ```bash
   docker build -t mlops-pipeline .
   ```
