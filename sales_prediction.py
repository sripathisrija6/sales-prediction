import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Load Dataset
data = pd.read_csv("advertising.csv")

# Display first rows
print(data.head())

# Features and Target
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Accuracy
print("\nR2 Score:", r2_score(y_test, y_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))

# Predict New Sales
new_data = [[150, 25, 30]]
predicted_sales = model.predict(new_data)

print("\nPredicted Sales:", predicted_sales[0])

# Plot
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Sales Prediction")
plt.show()