import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Load data
df = pd.read_csv("data/data.csv")
df=df.dropna()

X = df[
    ["season", "design", "batch_size", "room_temp",
     "plant_type", "part", "condition"]
]

y = df[["yield_ml_per_kg", "volume"]]

# Columns
cat_cols = ["season", "design", "plant_type", "part", "condition"]
num_cols = ["batch_size", "room_temp"]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols)
    ]
)

# Model
model = MultiOutputRegressor(
    RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
pipeline.fit(X_train, y_train)

# Evaluate
preds = pipeline.predict(X_test)
print("R2 Yield:", r2_score(y_test["yield_ml_per_kg"], preds[:, 0]))
print("R2 Volume:", r2_score(y_test["volume"], preds[:, 1]))

# Save
joblib.dump(pipeline, "model/model.pkl")

print("✅ Model saved successfully")
