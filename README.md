# Breast Cancer Prediction using K-Nearest Neighbors and Cross-Validation

This project implements a machine learning pipeline to classify breast cancer tumors as benign or malignant using the K-Nearest Neighbors (KNN) algorithm. It leverages cross-validation techniques to ensure robust model evaluation and prevent overfitting.

## 📌 Project Overview

Breast cancer is one of the most common cancers affecting women worldwide. Early detection can significantly improve prognosis. This project uses the popular Breast Cancer Wisconsin (Diagnostic) dataset to train and evaluate a KNN classifier.

## 🔍 Objectives
- Load and preprocess the dataset.
- Train a KNN classifier.
- Optimize hyperparameters (e.g., number of neighbors) using cross-validation.
- Evaluate model performance with metrics such as accuracy and confusion matrix.

## 🛠️ Technologies & Libraries
- Python
- scikit-learn
- pandas
- NumPy
- Matplotlib (for visualization)

## ⚙️ How to Run

1. Clone this repository:
    ```bash
    git clone <repository-url>
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the notebook or script:
    ```bash
    python breast_cancer_knn.py
    ```
   or open the Jupyter Notebook:
    ```bash
    jupyter notebook breast_cancer_knn.ipynb
    ```

## 📈 Results
- KNN achieved high classification accuracy.
- Cross-validation helped determine the optimal number of neighbors.
- The final model showed strong performance in distinguishing malignant from benign tumors.

## 🧠 Next Steps
- Compare with other classifiers such as Support Vector Machines or Random Forests.
- Implement grid search for hyperparameter tuning.
- Explore feature selection techniques.

## 📝 License
This project is for educational purposes.

## 🙌 Acknowledgments
- [scikit-learn Breast Cancer Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)
