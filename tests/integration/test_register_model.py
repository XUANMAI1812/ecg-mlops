import importlib.util
import json
from pathlib import Path

import mlflow
import pytest
from torch import nn

from ecg.train.tracking import RunTracker

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "register_model.py"
_spec = importlib.util.spec_from_file_location("register_model", SCRIPT)
register_model = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(register_model)

THRESHOLDS = {"NORM": 0.44, "MI": 0.71, "STTC": 0.6, "CD": 0.7, "HYP": 0.68}


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(12, 4, 3)
        self.fc = nn.Linear(4, 5)

    def forward(self, x):
        return self.fc(self.conv(x).mean(dim=-1))


@pytest.fixture
def uri(tmp_path):
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.create_experiment("t", artifact_location=(tmp_path / "artifacts").as_uri())
    return tracking_uri


def _logged_run(uri: str, with_model: bool = True) -> str:
    with RunTracker(uri, "t", "run") as tracker:
        if with_model:
            tracker.log_model(TinyNet(), input_shape=(12, 1000))
        return tracker.run_id


def _thresholds_file(tmp_path: Path, run_id: str) -> Path:
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps({"run_id": run_id, "thresholds": THRESHOLDS}))
    return path


def test_register_creates_version_alias_tag_and_thresholds_artifact(uri, tmp_path):
    run_id = _logged_run(uri)
    client = mlflow.MlflowClient()

    version = register_model.register(
        client, run_id, _thresholds_file(tmp_path, run_id)
    )

    assert str(version.version) == "1"
    assert version.run_id == run_id
    assert version.tags["thresholds_artifact"] == "serving/thresholds.json"
    assert (
        str(client.get_model_version_by_alias("ecg_classifier", "staging").version)
        == "1"
    )

    local = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="serving/thresholds.json", dst_path=str(tmp_path)
    )
    content = json.loads(Path(local).read_text(encoding="utf-8"))
    assert content["run_id"] == run_id
    assert content["thresholds"] == THRESHOLDS
    assert content["model_id"] == register_model.find_logged_model_id(client, run_id)

    model = mlflow.pyfunc.load_model("models:/ecg_classifier@staging")
    assert model.metadata.run_id == run_id


def test_register_again_reuses_version(uri, tmp_path):
    run_id = _logged_run(uri)
    client = mlflow.MlflowClient()
    path = _thresholds_file(tmp_path, run_id)

    register_model.register(client, run_id, path)
    version = register_model.register(client, run_id, path, alias="production")

    assert str(version.version) == "1"
    assert len(client.search_model_versions("name = 'ecg_classifier'")) == 1
    assert set(client.get_registered_model("ecg_classifier").aliases) == {
        "staging",
        "production",
    }


def test_register_rejects_thresholds_of_another_run(uri, tmp_path):
    run_id = _logged_run(uri)
    client = mlflow.MlflowClient()

    with pytest.raises(ValueError, match="run_id"):
        register_model.register(client, run_id, _thresholds_file(tmp_path, "khac"))

    assert client.search_registered_models() == []


def test_register_requires_exactly_one_logged_model(uri, tmp_path):
    run_id = _logged_run(uri, with_model=False)

    with pytest.raises(SystemExit, match="đúng 1 logged model"):
        register_model.register(
            mlflow.MlflowClient(), run_id, _thresholds_file(tmp_path, run_id)
        )


def test_register_rejects_unknown_alias(uri, tmp_path):
    with pytest.raises(ValueError, match="alias"):
        register_model.register(
            mlflow.MlflowClient(), "x", _thresholds_file(tmp_path, "x"), alias="dev"
        )
