import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
df = pd.read_csv(r"c:\Users\aishs\Downloads\Car data.csv")
print(df.columns.tolist())
print("First 5 Rows:")
print(df.head())
print("\nDataset Shape:")
print(df.shape)
print("\nDataset Information:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())
df["Car_Age"] = 2025 - df["Year"]
df.drop("Year", axis=1, inplace=True)
le = LabelEncoder()
categorical_columns = ["Fuel_Type", "Selling_Type", "Transmission"]
for col in categorical_columns:
    df[col] = le.fit_transform(df[col])
df["Car_Name"] = LabelEncoder().fit_transform(df["Car_Name"])
print("\nProcessed Dataset:")
print(df.head())
X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print("\n========== Model Evaluation ==========")
print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})
importance = importance.sort_values(by="Importance", ascending=False)
print("\nFeature Importance:")
print(importance)
plt.figure(figsize=(8,5))
plt.barh(importance["Feature"], importance["Importance"])
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.show()
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs Predicted Price")
plt.plot([y.min(), y.max()],
         [y.min(), y.max()],
         color='red')
plt.show()
residuals = y_test - y_pred
plt.figure(figsize=(6,4))
plt.scatter(y_pred, residuals)
plt.axhline(y=0, color='red')
plt.xlabel("Predicted Price")
plt.ylabel("Residual")
plt.title("Residual Plot")
plt.show()
sample = X.iloc[[0]]
prediction = model.predict(sample)
print("\nPredicted Selling Price:")
print(prediction[0], "Lakhs")