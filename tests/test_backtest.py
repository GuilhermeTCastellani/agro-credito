"""Testes do backtest trimestral (src/backtest.py).  Rodar: python -m pytest tests/test_backtest.py -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

import backtest as bt  # noqa: E402


def test_recortes_respeitam_o_embargo_e_o_trimestre():
    datas = pd.Series(pd.date_range("2023-01-31", "2026-02-28", freq="ME"))
    treino, prev = bt.recortes(datas, "2025-03-31")
    assert datas[treino].max() == pd.Timestamp("2024-09-30")  # resultado de set/2024 sai em mar/2025
    assert list(datas[prev].dt.strftime("%Y-%m")) == ["2025-03", "2025-04", "2025-05"]
    _, prev_fim = bt.recortes(datas, "2025-12-31")
    assert list(datas[prev_fim].dt.strftime("%Y-%m")) == ["2025-12", "2026-01", "2026-02"]


def test_teto_do_peso_vem_so_do_treino():
    df = pd.DataFrame({"peso": np.r_[np.arange(1, 101, dtype=float), [1e6]], "x": 0})
    mascara = np.r_[np.ones(100, bool), [False]]
    d = bt.treino_da_origem(df, mascara)
    assert d["peso"].max() == pytest.approx(np.quantile(np.arange(1, 101), 0.99))
    assert df["peso"].max() == 1e6  # não altera a base original


def test_ajuste_de_nivel_acerta_a_media():
    rng = np.random.default_rng(0)
    lg, w = rng.normal(-3, 1, 1000), rng.uniform(1, 5, 1000)
    d = bt.ajuste_de_nivel(lg, w, 0.08)
    assert np.average(bt.sigmoide(lg + d), weights=w) == pytest.approx(0.08, abs=1e-6)


def _dados_sinteticos(seed=0):
    rng = np.random.default_rng(seed)
    linhas = []
    for origem in bt.ORIGENS:
        for k in range(bt.MESES_POR_ORIGEM):
            data = bt.fim_do_mes(origem, k)
            n = 400
            risco = rng.uniform(0.01, 0.2, n)
            linhas.append(pd.DataFrame({"data_base": data, "origem": pd.Timestamp(origem), "alvo_inad": rng.binomial(20, risco) / 20,
                                        "peso": rng.uniform(1, 10, n), "uf": rng.choice(["RS", "SP"], n), "submodalidade": "Custeio",
                                        "perfil_inad_atual_pct": risco * 100,
                                        **{m: np.clip(risco * rng.uniform(0.7, 1.3, n), 1e-4, 0.99) for m in bt.MODELOS}}))
    d = pd.concat(linhas, ignore_index=True)
    d["peso_teto"] = d["peso"]
    return d


def test_formas_moveis_nao_olham_resultados_ainda_desconhecidos():
    d = _dados_sinteticos()
    coef = {m: 0.25 for m in bt.MODELOS}
    a = bt.combinar(d, coef, 0.0)
    origem = pd.Timestamp("2025-03-31")
    # muda o resultado de tudo o que ainda não era conhecido em mar/2025 (data-base depois de set/2024)
    futuro = d["data_base"] > bt.fim_do_mes(origem, -bt.HORIZONTE)
    d2 = d.copy()
    d2.loc[futuro, "alvo_inad"] = 0.9
    b = bt.combinar(d2, coef, 0.0)
    alvo = (d["origem"] == origem).to_numpy()
    for col in ("empilhado_movel", "media_nivel_movel", "media_calibrada_movel", "empilhado_fixo_calibrado"):
        assert np.allclose(a[col].to_numpy()[alvo], b[col].to_numpy()[alvo]), col
    # as duas primeiras origens não têm histórico conhecido: as formas móveis ficam vazias
    assert a.loc[d["origem"] == pd.Timestamp(bt.ORIGENS[0]), "empilhado_movel"].isna().all()
    assert np.allclose(a["media"], bt.sigmoide(np.mean([np.log(d[m] / (1 - d[m])) for m in bt.MODELOS], axis=0)))


def test_escolha_respeita_a_tolerancia():
    r = {"empilhado_fixo": {"pr_auc": 0.27, "alertas": {"10%": {"f1": 0.66}}, "erro_nivel_abs_pp": 0.9},
         "media": {"pr_auc": 0.265, "alertas": {"10%": {"f1": 0.655}}, "erro_nivel_abs_pp": 0.4},
         "ruim": {"pr_auc": 0.20, "alertas": {"10%": {"f1": 0.66}}, "erro_nivel_abs_pp": 0.1}}
    assert bt.escolher(r) == "media"


def test_ponta_a_ponta_numa_amostra(tmp_path):
    base = pd.read_parquet(RAIZ / "dados" / "processados" / "base_modelagem.parquet")
    amostra = base.sample(frac=0.02, random_state=1).sort_values("data_base").reset_index(drop=True)
    del base
    caminho = tmp_path / "base.parquet"
    amostra.to_parquet(caminho, index=False)
    pasta = tmp_path / "bt"
    # treino real só de um modelo e uma origem, para conferir o formato do arquivo
    tempos = bt.rodar(caminho, pasta, modelos=("xgboost",), origens=("2025-12-31",), gpu=False)
    assert len(tempos) == 1
    with np.load(pasta / "2025-12_xgboost.npz") as z:
        linha, p = z["linha"], z["p"]
        assert str(z["corte_treino"]) == "2025-06-30"
    assert (amostra.loc[linha, "data_base"] >= "2025-12-31").all() and np.isfinite(p).all()
    # rodar de novo não refaz o que já está salvo
    assert bt.rodar(caminho, pasta, modelos=("xgboost",), origens=("2025-12-31",)).empty
    # previsões sintéticas para as demais origens e modelos, para testar a análise de ponta a ponta
    rng = np.random.default_rng(0)
    for origem in bt.ORIGENS:
        _, m_prev = bt.recortes(amostra["data_base"], origem)
        linhas = np.flatnonzero(m_prev).astype(np.int32)
        for modelo in bt.MODELOS:
            arq = pasta / f"{origem[:7]}_{modelo}.npz"
            if not arq.exists():
                q = np.clip(amostra.loc[linhas, "perfil_inad_atual_pct"].to_numpy() / 100 * rng.uniform(0.8, 1.4, len(linhas)), 1e-4, 0.99)
                np.savez_compressed(arq, linha=linhas, p=q.astype(np.float32), segundos=np.float32(0))
    saida = bt.analisar(pasta, caminho)
    assert saida["vencedor"] in saida["resultados"]
    assert {"empilhado_fixo", "media", "empilhado_movel", "media_nivel_movel"} <= set(saida["resultados"])
    assert (pasta / "resultados.json").exists()
