"""
train.py

Top-level entry point: load data -> engineer features -> train baseline
model -> evaluate.

Run with:  python train.py
(after activating your venv and installing requirements.txt)

This is the file an interviewer will open first to understand your whole
pipeline at a glance - keep it readable, resist cramming logic in here
that belongs in src/.
"""

from pathlib import Path

from src.data_loader import load_raw_data, add_rul_labels
from src.features import add_rolling_features, clip_rul, get_feature_columns
from src.models import train_baseline_model
from src.evaluate import evaluate_predictions

DATA_DIR = Path(__file__).parent / "data" / "raw"


def main():
    # 1. Load
    train_path = DATA_DIR / "train_FD001.txt"
    if not train_path.exists():
        raise FileNotFoundError(
            f"Expected data at {train_path}. Download CMAPSS dataset and "
            "place train_FD001.txt in data/raw/."
        )

    df = load_raw_data(train_path)
    df = add_rul_labels(df)

    # 2. Feature engineering
    df = add_rolling_features(df)
    df = clip_rul(df, max_rul=125)
    feature_cols = get_feature_columns(df)

    # 3. Train baseline
    model, (X_val, y_val) = train_baseline_model(df, feature_cols)

    # 4. Evaluate
    y_pred = model.predict(X_val)
    metrics = evaluate_predictions(y_val, y_pred)

    print("\n--- Baseline model results ---")
    print(f"RMSE:         {metrics['rmse']:.2f} cycles")
    print(f"CMAPSS score: {metrics['cmapss_score']:.2f}")


if __name__ == "__main__":
    main()
