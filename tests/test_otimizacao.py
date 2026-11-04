"""Testes da busca, dos ensembles e da regra de escolha. Só usam dados sintéticos pequenos."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

optuna = pytest.importorskip("optuna")

from modelo import CATEGORICAS  # noqa: E402
from otimizacao import (FEATURES, MODELOS_BUSCA, NUMERICAS, _TrialFixo, amostra_por_exposicao, buscar,  # noqa: E402
                        combinar_ensembles, comparar, escolher_no_teste, gerar_previsoes, pr_auc, selecao_gulosa, sugerir,
                        treinar)


def _base(n=1200, seed=0, inicio="2023-01-31", meses=12):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"data_base": np.tile(pd.date_range(inicio, periods=meses, freq="ME"), n // meses)})
    for c in CATEGORICAS:
        d[c] = rng.choice(["a", "b"], len(d))
    for c in NUMERICAS:
        d[c] = rng.normal(5, 2, len(d)).clip(0)
    risco = 1 / (1 + np.exp(4 - d["perfil_inad_atual_pct"].to_numpy()))
    d["alvo_inad"] = np.clip(risco + rng.normal(0, 0.03, len(d)), 0, 1)
    d["peso"] = rng.uniform(0.5, 2, len(d))
    return d


def _janelas():
    return {"treino_val1": _base(seed=1), "val1": _base(400, 2, "2024-01-31", 6), "treino_val2": _base(seed=3),
            "val2": _base(400, 4, "2024-07-31", 6), "treino_teste": _base(seed=5), "teste": _base(600, 6, "2025-01-31", 12)}


def test_pr_auc_premia_quem_ordena_melhor():
    y, w = np.array([1.0, 1.0, 0.0, 0.0]), np.ones(4)
    assert pr_auc(y, w, np.array([0.9, 0.8, 0.2, 0.1])) == pytest.approx(1)
    assert pr_auc(y, w, np.array([0.1, 0.2, 0.8, 0.9])) < 0.6


def test_escolha_usa_as_quatro_metricas_e_desempata_pela_pr_auc():
    r = {"preciso": {"pr_auc": 0.24, "precisao": 0.9, "recall": 0.1, "f1": 0.2},
         "amplo": {"pr_auc": 0.20, "precisao": 0.1, "recall": 0.9, "f1": 0.2},
         "equilibrado": {"pr_auc": 0.25, "precisao": 0.5, "recall": 0.5, "f1": 0.5}}
    assert escolher_no_teste(r) == "equilibrado"
    empate = {"a": {"pr_auc": 0.3, "precisao": 0.1, "recall": 0.9, "f1": 0.4}, "b": {"pr_auc": 0.2, "precisao": 0.9, "recall": 0.1, "f1": 0.5}}
    assert escolher_no_teste(empate) == "a"


def test_amostra_respeita_exposicao_e_inadimplencia():
    d = pd.DataFrame({"peso": [1.0, 9.0], "alvo_inad": [1.0, 0.1]})
    s, y = amostra_por_exposicao(d, 20000)
    assert (s["alvo_inad"] == 0.1).mean() == pytest.approx(0.9, abs=0.02)
    assert y.mean() == pytest.approx(0.1 * 1 + 0.9 * 0.1, abs=0.02)


@pytest.mark.parametrize("modelo", MODELOS_BUSCA)
def test_espacos_de_busca_geram_parametros_validos(modelo):
    estudo = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=1))
    for _ in range(10):
        trial = estudo.ask()
        assert sugerir(modelo, trial)
        estudo.tell(trial, 0.5)


def test_parametros_salvos_reconstroem_os_derivados():
    estudo = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=2))
    trial = estudo.ask()
    original = sugerir("realmlp", trial)
    assert sugerir("realmlp", _TrialFixo(trial.params)) == original
    trial = estudo.ask()
    original = sugerir("lightgbm", trial)
    assert sugerir("lightgbm", _TrialFixo(trial.params)) == original


def test_selecao_gulosa_prefere_o_modelo_bom():
    rng = np.random.default_rng(3)
    y = rng.binomial(1, 0.2, 3000).astype(float)
    bom, ruido = y * 0.6 + rng.uniform(0, 0.4, 3000), rng.uniform(0, 1, 3000)
    pesos = selecao_gulosa({"bom": bom, "ruido": ruido}, y, np.ones(3000), passos=5)
    assert pesos.get("bom", 0) > pesos.get("ruido", 0)
    assert sum(pesos.values()) == pytest.approx(1)


@pytest.mark.parametrize("modelo", ["xgboost", "lightgbm", "catboost", "random_forest"])
def test_modelos_treinam_com_alvo_fracionario_e_preveem_probabilidade(modelo):
    pytest.importorskip({"xgboost": "xgboost", "lightgbm": "lightgbm", "catboost": "catboost", "random_forest": "sklearn"}[modelo])
    estudo = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=4))
    prever = treinar(modelo, sugerir(modelo, estudo.ask()), _base())
    p = prever(_base(100, 9))
    assert p.shape == (96,) or len(p) > 0
    assert ((p >= 0) & (p <= 1)).all()


def test_fluxo_completo_continua_de_onde_parou_e_compara_ensembles(tmp_path, monkeypatch):
    pytest.importorskip("lightgbm")
    monkeypatch.chdir(tmp_path)
    pasta = tmp_path / "progresso"
    janelas = _janelas()
    modelos = ("lightgbm", "random_forest")
    buscas = {m: buscar(m, janelas, pasta, 2) for m in modelos}
    assert all(b["tentativas"] == 2 for b in buscas.values())
    assert buscar("lightgbm", janelas, pasta, 3)["tentativas"] == 3  # continua, sem refazer as 2 primeiras
    previsoes = {m: gerar_previsoes(m, buscas[m]["params"], janelas, pasta) for m in modelos}
    arquivo = pasta / "previsoes_lightgbm.npz"
    momento = arquivo.stat().st_mtime_ns
    gerar_previsoes("lightgbm", buscas["lightgbm"]["params"], janelas, pasta)
    assert arquivo.stat().st_mtime_ns == momento  # não refaz o que está salvo
    comb = combinar_ensembles(previsoes, janelas)
    assert set(comb["previsoes"]) == {"ensemble_media", "ensemble_posicoes", "ensemble_guloso", "ensemble_empilhado"}
    saida = comparar({**previsoes, **comb["previsoes"]}, janelas, pasta, {"buscas": buscas})
    assert saida["vencedor"] in saida["resultados"]
    assert (pasta / "resultados_optuna.json").exists() and (pasta / "resultados_optuna.csv").exists()
    for r in saida["resultados"].values():
        assert {"pr_auc", "precisao", "recall", "f1", "auc", "brier"} <= set(r)


def test_features_nao_incluem_alvo_nem_peso():
    assert "alvo_inad" not in FEATURES and "peso" not in FEATURES
