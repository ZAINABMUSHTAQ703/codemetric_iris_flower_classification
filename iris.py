import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load the Iris dataset
# Using the classic Scikit-learn dataset
iris_data = load_iris()
df = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)
df['species'] = iris_data.target

# Mapping numerical targets to actual species names for better readability
df['species_name'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print("--- Initial Data Snapshot ---")
print(df.head())

# 2. Exploratory Data Analysis (EDA)
# Visualizing relationships between features using a pairplot
print("\nGenerating Pairplot for EDA...")
sns.pairplot(df.drop('species', axis=1), hue='species_name', palette='husl')
plt.suptitle("Exploratory Data Analysis: Iris Feature Relationships", y=1.02)
plt.show()

# 3. Data Splitting
# Separating features (X) and target labels (y)
X = iris_data.data
y = iris_data.target

# Splitting dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Model Building (k-Nearest Neighbors)
# Initializing the k-NN classifier with 3 neighbors
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)

# 5. Model Evaluation
# Making predictions on the test set
y_pred = knn_model.predict(X_test)

# Calculating and displaying accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n--- Model Performance ---")
print(f"Accuracy Score: {accuracy * 100:.2f}%")

# 6. Visualization: Confusion Matrix
# Plotting a heatmap to visualize true vs predicted labels
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', 
            xticklabels=iris_data.target_names, 
            yticklabels=iris_data.target_names)
plt.xlabel('Predicted Species')
plt.ylabel('Actual Species')
plt.title('Confusion Matrix - Iris Classification')
plt.show()

# Professional Summary Output
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris_data.target_names))