# Breast Cancer Prediction using Multiple Classifiers
### Project Overview:
This project focuses on building machine learning models to predict breast cancer based on a given dataset. We explore various classification algorithms, including:

- Logistic Regression
- Support Vector Machines (SVM)
- K-Nearest Neighbors (K-NN)
- Naive Bayes
- Neural Networks (MLP)
- Random Forest
- Ensemble Learning (Voting Classifier)
The project evaluates the performance of each model using different metrics such as accuracy, confusion matrix, classification report, and ROC-AUC score. Additionally, cross-validation is applied to ensure robustness in model evaluation.

### Dataset:
The dataset used for this project contains breast cancer diagnostic data, where features represent various clinical attributes. The target variable is a binary classification indicating whether the tumor is benign or malignant.

### Dataset Structure:
* Features: Various clinical measurements of breast cancer tumors (e.g., radius, texture, smoothness, etc.).
* Target: Indicates whether the tumor is benign (2) or malignant (4).

### Models Implemented:
1. Logistic Regression
A linear model used for binary classification. It predicts the probability of the target class using the sigmoid function.
2. Support Vector Machine (SVM)
SVM aims to find a hyperplane that best separates the data into two classes by maximizing the margin between the classes.
3. K-Nearest Neighbors (K-NN)
K-NN is a non-parametric model that classifies data points based on the majority class among the nearest neighbors.
4. Naive Bayes (GaussianNB)
Naive Bayes is a probabilistic model based on Bayes' theorem, assuming independence between features.
5. Neural Network (MLP)
A multi-layer perceptron (MLP) with one hidden layer using the ReLU activation function and trained with the Adam optimizer.
6. Random Forest
An ensemble model that builds multiple decision trees and combines them to improve performance and reduce overfitting.
7. Voting Classifier (Ensemble Model)
Combines the predictions of multiple classifiers (Logistic Regression, SVM, K-NN, Random Forest) using hard voting to make a final decision.

### Model Evaluation:
- Confusion Matrix:
Used to evaluate the performance of the models in terms of true positives, true negatives, false positives, and false negatives.

- Classification Report:
Provides a summary of precision, recall, F1-score, and support for each class:

* Precision: How many of the predicted positives are actually positive.
* Recall: How many actual positives are correctly identified.
* F1-Score: Harmonic mean of precision and recall.

- ROC-AUC Score:
Measures the model's ability to distinguish between classes. A higher AUC score indicates better performance.

- Cross-Validation:
Performs 10-fold cross-validation to ensure that the model's performance is consistent across different subsets of the data.

### Conclusion:
This project demonstrates the application of various machine learning models to predict breast cancer. The `K-NN` model achieved the highest accuracy. The Random Forest model and Ensemble model also performed exceptionally well. Cross-validation confirmed the consistency of results across all models.