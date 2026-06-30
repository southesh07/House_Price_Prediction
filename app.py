# Import libraries
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.linear_model    import Ridge
from sklearn.metrics         import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("house_prices_practice.csv").drop(columns=["Id"])
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("Nulls:\n", df.isnull().sum())
print("Duplicates:", df.duplicated().sum())

# Missing value handling
for col in df.columns:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)

# Outlier removal on target
target = "SalePrice"
Q1, Q3 = df[target].quantile(0.25), df[target].quantile(0.75)
IQR    = Q3 - Q1
df     = df[(df[target] >= Q1 - 1.5 * IQR) & (df[target] <= Q3 + 1.5 * IQR)]
print("Shape after outlier removal:", df.shape)

# Feature Engineering
df["HouseAge"] = 2025 - df["YearBuilt"]
df["TotalSF"]  = df["GrLivArea"] + df["TotalBsmtSF"]
df["QualArea"] = df["OverallQual"] * df["GrLivArea"]

# Train-Test Split
X = df.drop(target, axis=1)
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling
scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# Train Model
model = Ridge(alpha=1.0)
model.fit(X_train_s, y_train)

# Evaluate
y_pred = model.predict(X_test_s)
mae    = mean_absolute_error(y_test, y_pred)
rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
r2     = r2_score(y_test, y_pred)

print(f"\nMAE  : ${mae:,.2f}")
print(f"RMSE : ${rmse:,.2f}")
print(f"R2   : {r2:.4f}")

# Compare Actual vs Predicted
result = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
print(result.head(10))

# Save model bundle
bundle = {"model": model, "scaler": scaler, "feature_cols": list(X.columns)}
with open("model_bundle.pkl", "wb") as f:
    pickle.dump(bundle, f)

print("\nSaved model_bundle.pkl")
print("Run the app with:  streamlit run streamlit_app.py")
