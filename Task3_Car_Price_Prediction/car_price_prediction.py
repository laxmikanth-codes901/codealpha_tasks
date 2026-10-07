import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = "data/car_data.csv"
TARGET = "Selling_Price"

# Step 1: Load and inspect
print("=== Step 1: Load data ===")
df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)
print(df.head())

# Step 2: Data cleaning
print("\n=== Step 2: Data cleaning ===")
print("Missing values:\n", df.isnull().sum())
print("Duplicates:", df.duplicated().sum())
df = df.drop_duplicates()
df = df.dropna()

# Step 3: Feature engineering
#  - 'age': how old the car is (resale price falls with age)
#  - drop Car_Name: too many unique brands; keep as is for non-technical simplicity
print("\n=== Step 3: Feature engineering ===")
current_year = df["Year"].max()
df["Car_Age"] = current_year - df["Year"]
df = df.drop(columns=["Year"])
print("Created Car_Age using current year", current_year)
print("Categorical columns:", df.select_dtypes(include="str").columns.tolist())

# Step 4: EDA
print("\n=== Step 4: Exploratory data analysis ===")
print(df.describe())

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.scatterplot(data=df, x="Present_Price", y=TARGET, ax=axes[0])
axes[0].set_title("Selling Price vs Present Price")
sns.scatterplot(data=df, x="Driven_kms", y=TARGET, ax=axes[1])
axes[1].set_title("Selling Price vs Kilometers Driven")
plt.tight_layout()
plt.savefig("car_eda.png", dpi=120)
plt.close()
print("Saved car_eda.png")

# Step 5: Preprocessing pipeline
#  - One-hot encode categorical features (Car_Name, Fuel_Type, Selling_type, Transmission)
#  - Scale numeric features
print("\n=== Step 5: Preprocessing pipeline ===")
X = df.drop(columns=[TARGET])
y = df[TARGET]

categorical_cols = X.select_dtypes(include="str").columns.tolist()
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
print("Numeric:", numeric_cols)
print("Categorical:", categorical_cols)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# Step 6: Train regression models
print("\n=== Step 6: Model training and evaluation ===")
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
}

results = {}
for name, model in models.items():
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2}
    print(f"\n--- {name} ---")
    print(f"MAE : {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R^2 : {r2:.4f}")

# Step 7: Comparison summary
print("\n=== Step 7: Model comparison ===")
summary = pd.DataFrame(results).T
print(summary.round(4))
best_model = summary["R2"].idxmax()
print(f"\nBest model: {best_model} with R^2 = {summary.loc[best_model, 'R2']:.4f}")

# Step 8: Visualize best model predictions
print("\n=== Step 8: Predictions vs actual (best model) ===")
pipe = Pipeline(steps=[("preprocessor", preprocessor),
                       ("model", models[best_model])])
pipe.fit(X_train, y_train)
preds = pipe.predict(X_test)
plt.figure(figsize=(8, 6))
plt.scatter(y_test, preds, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title(f"Model: {best_model}  (R^2 = {summary.loc[best_model, 'R2']:.3f})")
plt.tight_layout()
plt.savefig("car_predictions.png", dpi=120)
plt.close()
print("Saved car_predictions.png")