import pickle
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = PROJECT_ROOT / "models"


def load_pickle(run: str) -> any:
    file_path = MODEL_DIR / f"{run}.pkl"

    with file_path.open("rb") as file:
        model = pickle.load(file)

    print(f"Loaded model from file '{file_path}'.")
    return model


def save_pickle(model: any, run: str) -> None:
    file_path = MODEL_DIR / f"{run}.pkl"
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open("wb") as file:
        pickle.dump(model, file)

    print(f"Saved model to file '{file_path}'.")
