# ml/train_model.py
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from preprocess import preprocess

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "train.csv"
MODEL_PATH = BASE_DIR / "ml" / "pipeline.pkl"


def train_model():
    try:
        print("🚀 Loading training data...")
        if not DATA_PATH.exists():
            raise FileNotFoundError(f"Training data not found at {DATA_PATH}")

        df = pd.read_csv(DATA_PATH)

        print("🔧 Preprocessing data...")
        df = preprocess(df)

        if "Sale_Amount" not in df.columns:
            raise ValueError(
                "Sale_Amount column is missing after preprocessing. "
                "Check preprocessing steps."
            )

        print("🧹 Cleaning target variable...")
        df = df.dropna(subset=['Sale_Amount'])

        X = df.drop("Sale_Amount", axis=1)
        y = df["Sale_Amount"]

        if X.empty:
            raise ValueError("Feature matrix X is empty after preprocessing.")
        if y.empty:
            raise ValueError("Target vector y is empty after preprocessing.")

        print("📊 Splitting train/validation...")
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        numeric_features = [
            "Clicks",
            "Impressions",
            "Cost",
            "Leads",
            "Conversions",
            "Conversion Rate",
            "Ad_Year",
            "Ad_Month",
            "Ad_DayOfWeek",
        ]

        categorical_features = [
            "Campaign_Name",
            "Location",
            "Device",
            "Keyword",
        ]

        print("🔄 Creating preprocessing pipeline...")
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
                ("num", SimpleImputer(strategy="median"), numeric_features),
            ],
            remainder="drop",
        )

        print("🤖 Initializing model...")
        model = RandomForestRegressor(
            n_estimators=350,
            random_state=42,
            n_jobs=-1,
        )

        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ])

        print("🏋️ Training model...")
        pipeline.fit(X_train, y_train)

       

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"💾 Saving pipeline to {MODEL_PATH} ...")
        joblib.dump(pipeline, MODEL_PATH)

        print("\n🎉 Training complete! Model saved as ml/pipeline.pkl")
    except Exception as e:
        print("❌ Error during training:", e, file=sys.stderr)
        raise


if __name__ == "__main__":
    train_model()
