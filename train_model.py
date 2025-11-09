import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

print("🚀 Training model using processed_data.csv...")

# 1️⃣ Load dataset
df = pd.read_csv("data/processed_data.csv")
print("✅ Dataset loaded successfully:", df.shape)

# 2️⃣ Select only the relevant features
features = [
    "year",
    "population",
    "gdp",
    "primary_energy_consumption",
    "fossil_fuel_consumption",
    "renewables_consumption",
    "greenhouse_gas_emissions"
]

# Make sure the target column exists
target = "electricity_demand"

# Keep only selected columns
df = df[features + [target]].dropna()

# 3️⃣ Split data
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4️⃣ Train model
model = RandomForestRegressor(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

# 5️⃣ Evaluate model
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

print(f"📊 Model Performance → MAE: {mae:.2f}, R²: {r2:.3f}")

# 6️⃣ Save model
joblib.dump(model, "models/energy_model.pkl")
print("✅ Model saved at: models/energy_model.pkl")
