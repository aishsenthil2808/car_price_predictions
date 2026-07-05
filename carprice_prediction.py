# ==========================================
# TASK 3: CAR PRICE PREDICTION
# ==========================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_csv(r"C:\Users\aishs\Downloads\car data.csv")

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

print("========== DATASET ==========")
print(df.head())

print("\nColumns:")
print(df.columns.tolist())

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

# ==========================================
# Feature Engineering
# ==========================================

# Create Car Age Feature
df["Car_Age"] = 2025 - df["Year"]

# Remove Year Column
df.drop("Year", axis=1, inplace=True)

# ==========================================
# Encode Categorical Columns
# ==========================================

encoder = LabelEncoder()

categorical_columns = [
    "Car_Name",
    "Fuel_Type",
    "Selling_type",
    "Transmission"
]

for col in categorical_columns:
    df[col] = encoder.fit_transform(df[col])

print("\nProcessed Dataset:")
print(df.head())

# ==========================================
# Feature Selection
# ==========================================

X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

# ==========================================
# Train Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ==========================================
# Random Forest Regression
# ==========================================

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# ==========================================
# Prediction
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# Evaluation
# ==========================================

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n========== MODEL EVALUATION ==========")
print("Mean Absolute Error :", mae)
print("Mean Squared Error :", mse)
print("Root Mean Squared Error :", rmse)
print("R2 Score :", r2)

# ==========================================
# Feature Importance
# ==========================================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance")
print(importance)

plt.figure(figsize=(8,5))
plt.barh(importance["Feature"], importance["Importance"])
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.grid(True)
plt.show()

# ==========================================
# Actual vs Predicted Scatter Plot
# ==========================================

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred)

plt.plot(
    [y.min(), y.max()],
    [y.min(), y.max()],
    color="red"
)

plt.title("Actual vs Predicted Selling Price")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.grid(True)
plt.show()

# ==========================================
# Residual Plot
# ==========================================

residuals = y_test - y_pred

plt.figure(figsize=(6,4))
plt.scatter(y_pred, residuals)

plt.axhline(y=0, color="red")

plt.title("Residual Plot")
plt.xlabel("Predicted Price")
plt.ylabel("Residual")

plt.grid(True)
plt.show()

# ==========================================
# Predict Sample Car Price
# ==========================================

sample = X.iloc[[0]]

prediction = model.predict(sample)

print("\nPredicted Selling Price:")
print(round(prediction[0],2), "Lakhs")

# ==========================================
# GRAPHICAL REPRESENTATIONS
# ==========================================

# 1 Distribution of Selling Price
plt.figure(figsize=(8,5))
plt.hist(df["Selling_Price"], bins=15)
plt.title("Distribution of Selling Price")
plt.xlabel("Selling Price")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# 2 Distribution of Present Price
plt.figure(figsize=(8,5))
plt.hist(df["Present_Price"], bins=15)
plt.title("Distribution of Present Price")
plt.xlabel("Present Price")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# 3 Fuel Type Distribution
fuel = df["Fuel_Type"].value_counts()

plt.figure(figsize=(6,4))
plt.bar(fuel.index.astype(str), fuel.values)
plt.title("Fuel Type Distribution")
plt.xlabel("Fuel Type")
plt.ylabel("Count")
plt.grid(axis="y")
plt.show()

# 4 Selling Type Distribution
selling = df["Selling_type"].value_counts()

plt.figure(figsize=(6,4))
plt.bar(selling.index.astype(str), selling.values)
plt.title("Selling Type Distribution")
plt.xlabel("Selling Type")
plt.ylabel("Count")
plt.grid(axis="y")
plt.show()

# 5 Transmission Distribution
trans = df["Transmission"].value_counts()

plt.figure(figsize=(6,4))
plt.bar(trans.index.astype(str), trans.values)
plt.title("Transmission Distribution")
plt.xlabel("Transmission")
plt.ylabel("Count")
plt.grid(axis="y")
plt.show()

# 6 Present Price vs Selling Price
plt.figure(figsize=(7,5))
plt.scatter(df["Present_Price"], df["Selling_Price"])
plt.title("Present Price vs Selling Price")
plt.xlabel("Present Price")
plt.ylabel("Selling Price")
plt.grid(True)
plt.show()

# 7 Driven Kilometers vs Selling Price
plt.figure(figsize=(7,5))
plt.scatter(df["Driven_kms"], df["Selling_Price"])
plt.title("Driven Kilometers vs Selling Price")
plt.xlabel("Driven Kilometers")
plt.ylabel("Selling Price")
plt.grid(True)
plt.show()

# 8 Car Age vs Selling Price
plt.figure(figsize=(7,5))
plt.scatter(df["Car_Age"], df["Selling_Price"])
plt.title("Car Age vs Selling Price")
plt.xlabel("Car Age")
plt.ylabel("Selling Price")
plt.grid(True)
plt.show()

# 9 Actual vs Predicted Line Graph
plt.figure(figsize=(10,5))
plt.plot(y_test.values, label="Actual Price", marker="o")
plt.plot(y_pred, label="Predicted Price", marker="x")

plt.title("Actual vs Predicted Selling Price")
plt.xlabel("Test Samples")
plt.ylabel("Selling Price")
plt.legend()
plt.grid(True)
plt.show()
