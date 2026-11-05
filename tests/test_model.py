import pickle

from fastapi.testclient import TestClient
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

import app as iris_app
from train import train_model


def test_training_saves_loadable_model_and_clears_quality_gate(tmp_path):
    model_path = tmp_path / "model.pkl"
    accuracy = train_model(model_path)
    assert accuracy >= 0.90
    assert model_path.is_file()
    with model_path.open("rb") as model_file:
        model = pickle.load(model_file)
    assert model.predict([[5.1, 3.5, 1.4, 0.2]]).shape == (1,)


def test_prediction_endpoint(monkeypatch):
    iris = load_iris()
    model = RandomForestClassifier(random_state=42).fit(iris.data, iris.target)
    monkeypatch.setattr(iris_app, "load_model", lambda: model)
    client = TestClient(iris_app.app)
    response = client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})
    assert response.status_code == 200
    assert response.json()["class_name"] == "setosa"


def test_prediction_rejects_wrong_feature_count():
    client = TestClient(iris_app.app)
    response = client.post("/predict", json={"features": [5.1, 3.5]})
    assert response.status_code == 422
