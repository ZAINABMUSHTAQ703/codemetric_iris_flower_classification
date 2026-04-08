# Iris Flower Classification - Codmetric Internship Task 1

## Project Overview
This project is a part of my Machine Learning Internship at **Codmetric**. The goal is to classify Iris flowers into three species: **Setosa, Versicolor, and Virginica** based on their sepal and petal measurements.

## Dataset
The dataset used is the classic **Iris Dataset** from `sklearn.datasets`. It contains 150 samples with 4 features:
1. Sepal Length (cm)
2. Sepal Width (cm)
3. Petal Length (cm)
4. Petal Width (cm)

## Methodology
- **Exploratory Data Analysis (EDA):** Used `Seaborn` pairplots to visualize the relationships between features.
- **Data Splitting:** Divided the data into 80% Training and 20% Testing sets.
- **Algorithm:** Applied the **k-Nearest Neighbors (k-NN)** classifier.
- **Evaluation:** Achieved **100% accuracy** on the test set, verified through a Confusion Matrix and Classification Report.

## Libraries Used
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## Results
The model successfully predicted all test samples with perfect precision and recall, as shown in the Confusion Matrix included in the repository.
