import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_PATH = "data/Iris.csv"

# Step 1: Load and inspect the data
df = pd.read_csv(DATA_PATH)
print("=== Step 1: Load data ===")
print(df.head())
print("\nShape:", df.shape)
print("\nInfo:")
print(df.info())

# Step 2: Check missing values and clean
print("\n=== Step 2: Data cleaning ===")
print("Missing values:\n", df.isnull().sum())
df = df.dropna()

# Drop the Id column (not a feature) and fix the Species name if needed
if "Id" in df.columns:
    df = df.drop(columns=["Id"])
df["Species"] = df["Species"].astype(str).str.strip().str.lower().str.replace("iris-", "")
print("\nUnique species:", df["Species"].unique())
print(df["Species"].value_counts())

# Step 3: Exploratory data analysis (visual summaries)
print("\n=== Step 3: Exploratory data analysis ===")
print(df.describe())

sns.pairplot(df, hue="Species")
plt.savefig("iris_pairplot.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved iris_pairplot.png")

# Step 4: Prepare features and target
print("\n=== Step 4: Train/test split + scaling ===")
X = df.drop(columns=["Species"]).values
y = df["Species"].values

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
print("Classes (encoded):", dict(zip(label_encoder.classes_, range(len(label_encoder.classes_)))))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 5: Train and evaluate multiple models
print("\n=== Step 5: Model training and evaluation ===")
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    print(f"\n--- {name} ---")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, target_names=label_encoder.classes_))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))

# Step 6: Summary
print("\n=== Step 6: Model comparison ===")
for name, acc in results.items():
    print(f"{name}: {acc:.4f}")

best_model = max(results, key=results.get)
print(f"\nBest model: {best_model} with accuracy {results[best_model]:.4f}")


