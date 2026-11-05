"""Ensemble empilhado: salvar e carregar os componentes, prever e explicar, e o treino de produção em dados sintéticos."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

optuna = pytest.importorskip("optuna")
pytest.importorskip("lightgbm")
pytest.importorskip("catboost")

from modelo import Calibrador, Componente, ModeloEnsemble, logit  # noqa: E402
from otimizacao import MODELOS_ENSEMBLE, buscar, sugerir, treinar, treinar_producao  # noqa: E402
from test_otimizacao import _base, _janelas  # noqa: E402


@pytest.mark.parametrize("nome", MODELOS_ENSEMBLE)
def test_componente_salva_carrega_e_explica(nome, tmp_path):
    comp = treinar(nome, sugerir(nome, optuna.create_study(sampler=optuna.samplers.TPESampler(seed=1)).ask()), _base())
    novo = _base(120, 7)
    antes = comp.prever(novo)
    comp.salvar(tmp_path / nome)
    lido = Componente.carregar(tmp_path / nome)
    np.testing.assert_allclose(lido.prever(novo), antes, rtol=1e-5, atol=1e-6)
    contrib = lido.contribuicoes(novo)
    assert len(contrib) == len(novo)
    assert "perfil_inad_atual_pct" in contrib.columns


def _ensemble():
    comps = {n: treinar(n, sugerir(n, optuna.create_study(sampler=optuna.samplers.TPESampler(seed=2)).ask()), _base())
             for n in MODELOS_ENSEMBLE}
    return ModeloEnsemble(comps, {"xgboost": 0.4, "catboost": 0.3, "random_forest": 0.3, "lightgbm": 0.2}, -0.1)


def test_ensemble_preve_explica_e_volta_igual_do_disco(tmp_path):
    m = _ensemble()
    novo = _base(120, 8)
    bruto = m.prever_bruto(novo)
    esperado = 1 / (1 + np.exp(-(-0.1 + sum(c * logit(m.componentes[n].prever(novo)) for n, c in m.coeficientes.items()))))
    np.testing.assert_allclose(bruto, esperado)
    m.calibrador = Calibrador("platt").fit(bruto, novo["alvo_inad"].to_numpy(), novo["peso"].to_numpy())
    m.limiar, m.fatia_pct = 0.2, 5.0
    m.preparar_explicador(_base(240, 9))
    ex = m.explicar(novo)
    assert "calibracao" in ex.columns
    # sem o Random Forest, a soma fecha com a previsão; com ele, é aproximada
    soma = m.valor_base + ex.drop(columns="calibracao").sum(axis=1).to_numpy()
    assert np.median(np.abs(soma - logit(bruto))) < 0.5
    m.salvar(tmp_path / "ens")
    lido = ModeloEnsemble.carregar(tmp_path / "ens")
    np.testing.assert_allclose(lido.prever(novo), m.prever(novo), rtol=1e-5, atol=1e-6)
    assert (lido.limiar, lido.fatia_pct, lido.valor_base) == (m.limiar, m.fatia_pct, m.valor_base)


def test_treino_de_producao_salva_tudo_e_retoma(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    janelas = _janelas()
    base = pd.concat([janelas["treino_teste"], janelas["teste"]], ignore_index=True)
    caminho = tmp_path / "base.parquet"
    base.to_parquet(caminho)
    atual = janelas["teste"][janelas["teste"]["data_base"] == janelas["teste"]["data_base"].max()]
    caminho_atual = tmp_path / "atual.parquet"
    atual.to_parquet(caminho_atual)
    pasta = tmp_path / "progresso"
    import otimizacao
    monkeypatch.setattr(otimizacao, "carregar_janelas", lambda _c: janelas)
    buscas = {n: buscar(n, janelas, pasta, 2) for n in MODELOS_ENSEMBLE}
    (pasta / "resultados_optuna.json").write_text(json.dumps({"buscas": buscas}, default=float), encoding="utf-8")
    info = treinar_producao(caminho, caminho_atual, pasta)
    assert 0 < info["fatia_pct"] < 100
    final = ModeloEnsemble.carregar(pasta / "producao" / "final")
    p = final.prever(atual)
    assert ((p >= 0) & (p <= 1)).all()
    momento = (pasta / "producao" / "final" / "lightgbm" / "modelo.txt").stat().st_mtime_ns
    treinar_producao(caminho, caminho_atual, pasta)
    assert (pasta / "producao" / "final" / "lightgbm" / "modelo.txt").stat().st_mtime_ns == momento  # não refaz
