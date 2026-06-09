import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load Dataset
data = pd.read_csv("iris.csv")

# Step 2: Display First 5 Rows
print("First 5 Rows of Dataset:")
print(data.head())

# Features and Target
X = data.drop(["Id", "Species"], axis=1)
y = data["Species"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 5: Train Model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Step 6: Make Predictions
y_pred = model.predict(X_test)

# Step 7: Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:")
print(accuracy)

# Step 8: Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualization
sns.pairplot(data, hue="Species")
plt.suptitle("Iris Flower Classification Visualization", y=1.02)
plt.show()