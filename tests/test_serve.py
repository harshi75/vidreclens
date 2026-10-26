import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from vidreclens.models.twotower import TwoTower
from vidreclens.serve.app import create_app
from vidreclens.serve.engine import Recommender
from vidreclens.serve.export import export_bundle

N, D = 80, 16


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    feats = {"a": torch.randn(N, 24), "b": torch.randn(N, 10)}
    model = TwoTower(feats, N, d=D, use_id=True)
    with torch.no_grad():
        model.id.weight.normal_(0, 0.5)
    item_ids = rng.permutation(np.arange(1000, 1000 + N))
    titles = [f"video {i}, with comma" for i in item_ids]
    pop = rng.integers(0, 50, N)
    out = export_bundle(model, item_ids, titles, pop, tmp_path_factory.mktemp("bundle"), alpha=0.75, max_len=20)
    return model, item_ids, pop, out


def torch_top(model, item_ids, pop, hist_iids, k, alpha, scale):
    model.eval()
    with torch.no_grad():
        IR = model.encode_items()
        h = torch.as_tensor(hist_iids)[None]
        u = model.encode_users(IR, h, torch.ones_like(h, dtype=torch.bool))[0]
        s = scale * (IR[:-1] @ u) - alpha * torch.log1p(torch.as_tensor(pop, dtype=torch.float32))
        s[torch.as_tensor(hist_iids)] = -float("inf")
        top = torch.topk(s, k)
    return [int(item_ids[i]) for i in top.indices], top.values.numpy()


def test_engine_matches_torch(setup):
    model, item_ids, pop, out = setup
    eng = Recommender(out)
    rng = np.random.default_rng(1)
    for alpha in (0.0, 0.75, 1.5):
        for _ in range(15):
            hist = rng.choice(N, size=int(rng.integers(1, 12)), replace=False)
            ref_items, ref_scores = torch_top(model, item_ids, pop, hist, 10, alpha, eng.scale)
            res = eng.recommend([int(item_ids[i]) for i in hist], 10, alpha)
            assert [r["item_id"] for r in res["items"]] == ref_items
            assert np.allclose([r["score"] for r in res["items"]], ref_scores, atol=1e-4)


def test_history_excluded_and_unknown_reported(setup):
    _, item_ids, _, out = setup
    eng = Recommender(out)
    hist = [int(item_ids[0]), int(item_ids[1]), 424242]
    res = eng.recommend(hist, 15)
    got = {r["item_id"] for r in res["items"]}
    assert int(item_ids[0]) not in got and int(item_ids[1]) not in got
    assert res["unknown_items"] == [424242]
    assert len(res["items"]) == 15 and not res["cold_start"]


def test_cold_start_returns_most_popular(setup):
    _, item_ids, pop, out = setup
    eng = Recommender(out)
    res = eng.recommend([], 5)
    assert res["cold_start"] and len(res["items"]) == 5
    assert res["items"][0]["item_id"] == int(item_ids[int(np.argmax(pop))])
    assert eng.recommend([999999], 3)["cold_start"]


def test_api(setup):
    _, item_ids, _, out = setup
    with TestClient(create_app(out)) as c:
        h = c.get("/health").json()
        assert h["status"] == "ok" and h["items"] == N
        r = c.post("/recommend", json={"history": [int(item_ids[3]), int(item_ids[4])], "k": 7})
        assert r.status_code == 200 and len(r.json()["items"]) == 7
        assert c.post("/recommend", json={"history": [int(item_ids[3])], "k": 0}).status_code == 422
        assert c.post("/recommend", json={"history": [], "alpha": 5}).status_code == 422
        assert c.get(f"/items/{int(item_ids[0])}").json()["title"].startswith("video")
        assert c.get("/items/1").status_code == 404
        a0 = c.post("/recommend", json={"history": [int(item_ids[3])], "k": 20, "alpha": 0.0}).json()["items"]
        a2 = c.post("/recommend", json={"history": [int(item_ids[3])], "k": 20, "alpha": 2.0}).json()["items"]
        assert [x["item_id"] for x in a0] != [x["item_id"] for x in a2]
