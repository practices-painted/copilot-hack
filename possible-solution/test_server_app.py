import os
import importlib.util

class FakeModel:
    def predict_proba(self, arr):
        # Return a bytes-like object so str(...) looks like "b'0.85 12.3'"
        return [b"0.85 12.3"]


def load_app():
    here = os.path.dirname(__file__)
    server_path = os.path.join(here, 'server', 'app.py')
    spec = importlib.util.spec_from_file_location("app_mod", server_path)
    app_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app_mod)
    return app_mod


def test_predict():
    app_mod = load_app()
    # override the loaded model with a predictable fake
    app_mod.model = FakeModel()
    client = app_mod.app.test_client()
    resp = client.get('/predict?day_of_week=3&airport_id=14771')
    assert resp.status_code == 200
    data = resp.get_json()
    assert isinstance(data.get('certainty'), float)
    assert abs(data['certainty'] - 0.85) < 1e-6
    assert isinstance(data.get('delay'), float)
    assert abs(data['delay'] - 12.3) < 1e-6


def test_airports(tmp_path, monkeypatch):
    here = os.path.dirname(__file__)
    server_dir = os.path.join(here, 'server')
    # ensure the endpoint reads the CSV from the server directory
    monkeypatch.chdir(server_dir)
    csv_path = os.path.join(server_dir, 'airports.csv')
    content = "id,name\n2,B Airport\n1,A Airport\n3,C Airport\n"
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write(content)

    app_mod = load_app()
    client = app_mod.app.test_client()
    resp = client.get('/airports')
    assert resp.status_code == 200
    data = resp.get_json()
    # should be sorted by name A,B,C
    assert [a['id'] for a in data] == [1, 2, 3]
    assert [a['name'] for a in data] == ['A Airport', 'B Airport', 'C Airport']

    # cleanup
    try:
        os.remove(csv_path)
    except Exception:
        pass
