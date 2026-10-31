import json

import pytest

from pytorch_examples.cli import main
from pytorch_examples.examples import EXAMPLES


def test_list_prints_every_example(capsys):
    assert main(["list"]) == 0
    out = capsys.readouterr().out
    for name in EXAMPLES:
        assert name in out


def test_info_prints_json(capsys):
    assert main(["info"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert {"torch", "device", "cuda_available"} <= set(info)


def test_run_prints_metrics_as_json(capsys):
    assert main(["run", "autograd_basics"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["grads_match"] is True


def test_run_accepts_overrides(capsys):
    assert main(["run", "linear_regression", "--epochs", "2", "--device", "cpu"]) == 0
    assert json.loads(capsys.readouterr().out)["epochs"] == 2


def test_run_rejects_unknown_example():
    with pytest.raises(SystemExit) as excinfo:
        main(["run", "does_not_exist"])
    assert excinfo.value.code == 2


def test_a_command_is_required():
    with pytest.raises(SystemExit):
        main([])
