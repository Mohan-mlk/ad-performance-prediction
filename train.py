"""Train and persist the Iris random forest model."""

import argparse
import json
import pickle
from pathlib import Path

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def train_model(model_path: str | Path = "model.pkl") -> float:
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.2, random_state=42, stratify=data.target
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    destination = Path(model_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as model_file:
        pickle.dump(model, model_file)

    accuracy = float(model.score(X_test, y_test))
    (destination.parent / "metrics.json").write_text(
        json.dumps({"accuracy": accuracy}, indent=2) + "\n"
    )
    print("Model trained successfully!")
    print(f"Accuracy: {accuracy:.4f}")
    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="model.pkl")
    train_model(parser.parse_args().model_path)
