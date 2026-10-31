import json
import threading
import urllib.error
import urllib.request

import pytest

from pytorch_examples.examples import mlp_classifier
from pytorch_examples.serve import Predictor, build_predictor, make_server


@pytest.fixture(scope="module")
def predictor():
    # An untrained model is enough to test the HTTP and tensor plumbing quickly.
    return Predictor(mlp_classifier.build_model())


@pytest.fixture()
def base_url(predictor):
    server = make_server(predictor, "127.0.0.1", 0)  # port 0: the OS picks a free port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _post(url, payload):
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return response.status, json.loads(response.read())


def test_predictor_returns_a_probability_distribution(predictor):
    (prediction,) = predictor.predict([[0.1, 0.2]])
    assert prediction["class"] in range(mlp_classifier.NUM_CLASSES)
    assert sum(prediction["probabilities"]) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("bad", [[], [[1.0]], [[1.0, 2.0, 3.0]], [1.0, 2.0]])
def test_predictor_rejects_wrong_shapes(predictor, bad):
    with pytest.raises(ValueError):
        predictor.predict(bad)


def test_health_endpoint(base_url):
    with urllib.request.urlopen(f"{base_url}/health", timeout=10) as response:
        assert json.loads(response.read()) == {"status": "ok"}


def test_predict_endpoint(base_url):
    status, body = _post(f"{base_url}/predict", {"inputs": [[0.0, 0.0], [0.5, -0.5]]})
    assert status == 200
    assert len(body["predictions"]) == 2


def test_predict_endpoint_rejects_bad_input_with_400(base_url):
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _post(f"{base_url}/predict", {"inputs": [[1.0]]})
    assert excinfo.value.code == 400


def test_predict_endpoint_requires_inputs_key(base_url):
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _post(f"{base_url}/predict", {"wrong": 1})
    assert excinfo.value.code == 400


def test_unknown_path_is_404(base_url):
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(f"{base_url}/nope", timeout=10)
    assert excinfo.value.code == 404


def test_checkpoint_round_trip(tmp_path):
    path = tmp_path / "model.pt"
    first = build_predictor(path, epochs=1)  # trains, then saves
    assert path.exists()
    second = build_predictor(path)  # loads without training
    sample = [[0.3, -0.2]]
    assert first.predict(sample) == second.predict(sample)
