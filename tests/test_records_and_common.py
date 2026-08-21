import json

import pytest
import torch

from gated_residuals.artifacts import RunMetadata, save_manifest, save_records_csv
from gated_residuals.memory_activation import recall_sparsity_curve
from gated_residuals.records import ProbeCapture


def test_probe_capture_rejects_location_mismatch_and_nonfinite_values():
    state = torch.ones(1, 2, 4)
    capture = ProbeCapture(0, state, state, state * 0.5, state * 1.5, gate=torch.full((1, 2, 2, 1), 0.5))
    capture.validate()
    with pytest.raises(ValueError, match="shape"):
        ProbeCapture(0, state, state[..., :2], state, state).validate()
    bad = state.clone()
    bad[0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="NaN"):
        ProbeCapture(0, state, bad, state, state).validate()


def test_pra_recall_sparsity_copy_preserves_exact_kv_fraction():
    result = recall_sparsity_curve(
        [["a", "b", "c"], ["c", "b", "a"]],
        [{"a"}, {"a"}],
        candidate_token_lengths=[[2, 3, 5], [4, 3, 3]],
        require_complete_endpoint=True,
    )
    assert result["endpoint_complete"] is True
    assert result["kv_fraction_exact"] is True
    assert result["curve"][-1]["recall"] == pytest.approx(1.0)


def test_artifact_writers_persist_metadata_and_union_schema(tmp_path):
    metadata = RunMetadata.collect(
        run_id="test",
        model="tiny",
        model_variant="gated",
        dataset="analytic",
        seed=0,
    )
    manifest = save_manifest(tmp_path / "manifest.json", metadata, {"a": 1})
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["metadata"]["run_id"] == "test"
    table = save_records_csv(tmp_path / "metrics.csv", [{"a": 1}, {"a": 2, "b": 3}])
    text = table.read_text(encoding="utf-8")
    assert text.splitlines()[0] == "a,b"
