# Iris ML CI/CD

A small scikit-learn Iris classifier with automated tests, model validation, and a GitHub Actions release pipeline.

## Local run

```bash
python -m pip install -r requirements.txt
pytest -q
python train.py --model-path model.pkl
uvicorn app:app --host 0.0.0.0 --port 8000
```

The training script uses a fixed seed, saves a pickle model, and prints holdout accuracy. `/health` reports readiness; `/predict` accepts four Iris measurements in centimeters and returns the predicted class.

## GitHub Actions

Pull requests and pushes run unit tests and model validation. Pushes to `main` also build and publish a Docker image to GitHub Container Registry (GHCR) after the quality gate passes. The image is the deployable release artifact; a live hosting target can be connected separately.

In the repository settings, allow GitHub Actions to create and write packages if GHCR publishing is disabled by default. The workflow uses the built-in `GITHUB_TOKEN`; no personal access token is required.
