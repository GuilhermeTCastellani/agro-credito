"""
Treina, compara e calibra os modelos de risco na base oficial.

Janelas fora do tempo, sempre com 6 meses de embargo entre o último alvo usado e o período avaliado:
  validação 1: treino com perfis até jun/2023; avaliação em perfis de jan/2024 a jun/2024
              -> busca hiperparâmetros e ajusta o calibrador e o threshold usados no teste
              -> o threshold de melhor F1 vira uma fatia da carteira (ex.: os 8% de maior risco)
  validação 2: treino com perfis até dez/2023; avaliação em perfis de jul/2024 a dez/2024
              -> aplica a fatia da validação 1 mês a mês; o vencedor é o maior F1 nessa aplicação
  teste     : treino com perfis até jun/2024; avaliação em perfis a partir de jan/2025
              -> só mede; recebe o calibrador e a fatia prontos da validação 1, porque os alvos da
                 validação 2 (até jun/2025) ainda não seriam conhecidos no início do teste
  produção  : mesmo procedimento, deslocado para o fim da base (calibrador e fatia vêm dos 6 meses
              mais recentes, previstos por um modelo treinado 12 meses antes); o threshold exibido é o
              que produz essa fatia nos perfis do último mês

O alvo é uma fração (carteira inadimplente / carteira ativa). Cada perfil entra como um conjunto
de reais emprestados — fração y inadimplente, (1 - y) adimplente — ponderado pela exposição.
AUC, KS, Gini, Brier, precisão, recall e F1 são calculados nessa representação, ou seja,
medem reais de carteira, não número de produtores.

Candidatos:
  1. Regressão Logística, Random Forest e XGBoost (todas as variáveis, sem histórico do perfil)
  2. XGBoost relativo à UF: parte da log-chance da inadimplência atual da UF (offset) e aprende
     quanto cada perfil desvia desse patamar, sem as variáveis macro
  3. Versões com histórico do perfil e, numa delas, mais peso nos meses recentes
  4. As mesmas versões com histórico, com hiperparâmetros buscados pelo Brier
  5. Versões com o SCR ampliado (tendências, ativo problemático, vencimentos, crescimento, perfis
     parecidos e mês do alvo), com e sem busca de hiperparâmetros
  6. Versões com SCR ampliado e preços (cesta da UF, fertilizantes, boi e relação de troca)
  7. Versões com SCR ampliado e custos da Conab (margem estimada e variação do custo)

Critério de escolha (--criterio): F1 e AUC-ROC juntos (padrão, média das posições nas duas métricas), só F1 ou benefício
líquido médio, sempre na validação 2. A comparação entre modelos usa várias métricas. Os modelos finais de cada critério
ficam em modelos/por_criterio/ e src/trocar_criterio.py alterna entre eles.

As árvores aprendem com a perda logística, e a parada antecipada usa o log-loss. A busca de hiperparâmetros minimiza
o Brier na validação 1, porque ele mede a qualidade das probabilidades em vez de um único threshold. O teste não
participa de nenhuma dessas etapas.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from modelo import (CATEGORICAS, CUSTOS, MACRO, NUMERICAS_TODAS, PERFIL, PERFIL_AMPLIADO, PRECOS, Calibrador,
                    ModeloRisco, logit, offset_uf, preprocessador)
from trocar_criterio import CRITERIOS, trocar

RAIZ = Path(__file__).resolve().parents[1]
PROC = RAIZ / "dados" / "processados"
MODELOS = RAIZ / "modelos"
SEED = 42

FIM_TREINO_VAL = "2023-06-30"
INI_VAL, FIM_VAL = "2024-01-01", "2024-06-30"
FIM_TREINO_VAL2 = "2023-12-31"
INI_VAL2, FIM_VAL2 = "2024-07-01", "2024-12-31"
FIM_TREINO = "2024-06-30"
INI_TESTE = "2025-01-01"
NUMERICAS_SEM_MACRO = [c for c in NUMERICAS_TODAS if c not in MACRO]
MEIA_VIDA_MESES = 12
N_TENTATIVAS = 20
ALARMES_POR_ACERTO = (3, 5, 10, 20)  # sem dados de perda, testa quantos alarmes falsos valem um inadimplente achado
N_BOOTSTRAP = 200
HP_PADRAO = dict(learning_rate=0.05, max_depth=7, min_child_weight=0.5, subsample=0.8,
                 colsample_bytree=0.7, reg_lambda=2.0)


# ------------------------------------------------------------------ métricas
def expandir(y_frac, w):
    y = np.r_[np.ones_like(y_frac), np.zeros_like(y_frac)]
    return y, np.r_[w * y_frac, w * (1 - y_frac)]


def metricas(y_frac, w, p) -> dict:
    y, peso = expandir(y_frac, w)
    s = np.r_[p, p]
    ok = peso > 0
    y, peso, s = y[ok], peso[ok], s[ok]
    auc = roc_auc_score(y, s, sample_weight=peso)
    fpr, tpr, _ = roc_curve(y, s, sample_weight=peso)
    sc = np.clip(s, 1e-6, 1 - 1e-6)
    return {
        "auc": float(auc), "gini": float(2 * auc - 1), "ks": float(np.max(tpr - fpr)),
        "brier": float(np.average((s - y) ** 2, weights=peso)),
        "log_loss": float(-np.average(y * np.log(sc) + (1 - y) * np.log(1 - sc), weights=peso)),
        "inad_media_real_pct": float(np.average(y_frac, weights=w) * 100),
        "inad_media_prevista_pct": float(np.average(p, weights=w) * 100),
    }


def metricas_limiar(y_frac, w, p, limiar: float) -> dict:
    """Precisão, recall e F1 em reais de carteira, sinalizando os perfis com p >= limiar."""
    return metricas_sinal(y_frac, w, p >= limiar)


def fatia_do_limiar(p, w, limiar: float) -> float:
    """Fração da exposição que um limiar sinaliza."""
    return float(np.sum(w[p >= limiar]) / np.sum(w))


def sinalizar_por_fatia(p, w, datas, fatia: float) -> np.ndarray:
    """Em cada mês, sinaliza os perfis de maior risco até completar a fração `fatia` da exposição do mês."""
    datas = np.asarray(datas)
    sinal = np.zeros(len(p), dtype=bool)
    for mes in np.unique(datas):
        idx = np.flatnonzero(datas == mes)
        ordem = idx[np.argsort(-p[idx], kind="stable")]
        acum = np.cumsum(w[ordem])
        sinal[ordem[acum <= fatia * acum[-1]]] = True
    return sinal


def limiar_da_fatia(p, w, fatia: float) -> float:
    """Probabilidade que, sozinha, sinaliza aproximadamente a fração `fatia` da exposição."""
    ordem = np.argsort(-p, kind="stable")
    acum = np.cumsum(w[ordem])
    k = min(int(np.searchsorted(acum, fatia * acum[-1])), len(p) - 1)
    return float(p[ordem[k]])


def metricas_sinal(y_frac, w, sinal) -> dict:
    """Precisão, recall, F1 e MCC em reais de carteira para um conjunto de perfis sinalizados."""
    acerto = float(np.sum(w[sinal] * y_frac[sinal]))
    sinalizado, real, total = float(np.sum(w[sinal])), float(np.sum(w * y_frac)), float(np.sum(w))
    prec = acerto / sinalizado if sinalizado > 0 else 0.0
    rec = acerto / real if real > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    falso_pos, falso_neg = sinalizado - acerto, real - acerto
    verdadeiro_neg = total - acerto - falso_pos - falso_neg
    denominador = np.sqrt((acerto + falso_pos) * (acerto + falso_neg) * (verdadeiro_neg + falso_pos) * (verdadeiro_neg + falso_neg))
    mcc = (acerto * verdadeiro_neg - falso_pos * falso_neg) / denominador if denominador > 0 else 0.0
    return {"precisao": prec, "recall": rec, "f1": f1, "mcc": float(mcc), "pct_carteira_sinalizada": sinalizado / total * 100}


def beneficio_liquido(y_frac, w, sinal, alarmes_por_acerto: float) -> float:
    """Reais inadimplentes achados menos alarmes falsos descontados, por real de carteira (Vickers & Elkin, 2006)."""
    acerto = np.sum(w[sinal] * y_frac[sinal])
    falso = np.sum(w[sinal] * (1 - y_frac[sinal]))
    return float((acerto - falso / alarmes_por_acerto) / np.sum(w))


def curva_beneficio(y_frac, w, p) -> list[dict]:
    """Cada custo equivale ao limiar p >= 1 / (1 + alarmes), então a medida depende da calibração."""
    linhas = []
    for alarmes in ALARMES_POR_ACERTO:
        limiar = 1 / (1 + alarmes)
        linhas.append({"alarmes_por_acerto": alarmes, "limiar_pct": limiar * 100,
                       "modelo": beneficio_liquido(y_frac, w, p >= limiar, alarmes),
                       "todos": beneficio_liquido(y_frac, w, np.ones_like(p, dtype=bool), alarmes)})
    return linhas


def beneficio_medio(y_frac, w, p) -> float:
    return float(np.mean([linha["modelo"] for linha in curva_beneficio(y_frac, w, p)]))


def h_measure(y_frac, w, p, pontos_custo: int = 400) -> float:
    """H-measure de Hand (2009), com custos relativos numa Beta(2, 2)."""
    y, peso = expandir(y_frac, w)
    s, ok = np.r_[p, p], peso > 0
    fpr, tpr, _ = roc_curve(y[ok], s[ok], sample_weight=peso[ok])
    idx = np.unique(np.linspace(0, len(fpr) - 1, min(2000, len(fpr))).astype(int))
    fpr, tpr = fpr[idx][None, :], tpr[idx][None, :]
    pi1 = np.sum(w * y_frac) / np.sum(w)
    c = ((np.arange(pontos_custo) + 0.5) / pontos_custo)[:, None]
    densidade = 6 * c * (1 - c)
    perda = np.min(c * pi1 * (1 - tpr) + (1 - c) * (1 - pi1) * fpr, axis=1, keepdims=True)
    perda_sem_modelo = np.minimum(c * pi1, (1 - c) * (1 - pi1))
    return float(1 - np.sum(densidade * perda) / np.sum(densidade * perda_sem_modelo))


def calibracao_logistica(y_frac, w, p) -> dict:
    """Intercepto (alvo 0; negativo = superestima) e inclinação (alvo 1; < 1 = riscos extremos demais); Van Calster et al. (2019)."""
    x = logit(p)
    y, peso = expandir(y_frac, w)
    ok = peso > 0
    x2 = np.r_[x, x][ok].reshape(-1, 1)
    inclinacao = LogisticRegression(C=1e6, max_iter=1000).fit(x2, y[ok], sample_weight=peso[ok]).coef_[0, 0]
    intercepto = 0.0
    for _ in range(50):  # Newton com a previsão como offset
        q = 1 / (1 + np.exp(-(intercepto + x)))
        gradiente, curvatura = np.sum(w * (y_frac - q)), np.sum(w * q * (1 - q))
        passo = gradiente / curvatura
        intercepto += passo
        if abs(passo) < 1e-8:
            break
    return {"calibracao_intercepto": float(intercepto), "calibracao_inclinacao": float(inclinacao)}


def decompor_brier(y_frac, w, p, faixas: int = 10) -> dict:
    """Decomposição de Murphy (1973), Brier Skill Score e ECE, com faixas de mesma exposição."""
    ordem = np.argsort(p, kind="stable")
    acum = np.cumsum(w[ordem]) / np.sum(w)
    faixa = np.zeros(len(p), dtype=int)
    faixa[ordem] = np.minimum((acum * faixas).astype(int), faixas - 1)
    total = np.sum(w)
    media = np.sum(w * y_frac) / total
    confiabilidade = resolucao = ece = 0.0
    for k in np.unique(faixa):
        m = faixa == k
        wk = np.sum(w[m])
        if wk == 0:
            continue
        pk, yk = np.sum(w[m] * p[m]) / wk, np.sum(w[m] * y_frac[m]) / wk
        confiabilidade += wk * (pk - yk) ** 2 / total
        resolucao += wk * (yk - media) ** 2 / total
        ece += wk * abs(pk - yk) / total
    incerteza = media * (1 - media)
    brier = np.sum(w * (y_frac * (1 - p) ** 2 + (1 - y_frac) * p ** 2)) / total
    return {"brier_confiabilidade": float(confiabilidade), "brier_resolucao": float(resolucao),
            "brier_incerteza": float(incerteza), "brier_skill": float(1 - brier / incerteza), "ece": float(ece)}


def metricas_extras(y_frac, w, p) -> dict:
    y, peso = expandir(y_frac, w)
    ok = peso > 0
    return {"auprc": float(average_precision_score(y[ok], np.r_[p, p][ok], sample_weight=peso[ok])),
            "h_measure": h_measure(y_frac, w, p), **calibracao_logistica(y_frac, w, p), **decompor_brier(y_frac, w, p),
            "beneficio_medio": beneficio_medio(y_frac, w, p)}


def distribuicao_por_resultado(y_frac, w, p, faixas: int = 50) -> dict:
    """Distribuição das probabilidades entre os reais que ficaram inadimplentes e os demais."""
    limites = np.linspace(0, 1, faixas + 1)
    inad, _ = np.histogram(p, bins=limites, weights=w * y_frac)
    adim, _ = np.histogram(p, bins=limites, weights=w * (1 - y_frac))
    return {"centro_pct": ((limites[:-1] + limites[1:]) / 2 * 100).round(1).tolist(),
            "inadimplentes": (inad / inad.sum()).round(5).tolist(), "adimplentes": (adim / adim.sum()).round(5).tolist()}


def intervalos_bootstrap(y_frac, w, datas, ufs, previsoes: dict, referencia: str, n: int = N_BOOTSTRAP, seed: int = SEED) -> dict:
    """Intervalos de 95% reamostrando grupos de UF e mês, para respeitar parte da dependência entre os perfis.

    `previsoes` guarda, por modelo, (previsão sem calibrar, previsão calibrada, fatia). Inclui a diferença para `referencia`.
    """
    grupo = pd.factorize(pd.Series(np.asarray(datas)).astype(str) + "|" + pd.Series(np.asarray(ufs)).astype(str))[0]
    n_grupos = grupo.max() + 1
    rng = np.random.default_rng(seed)
    valores = {nome: [] for nome in previsoes}
    for _ in range(n):
        wb = w * rng.multinomial(n_grupos, np.full(n_grupos, 1 / n_grupos))[grupo]
        for nome, (bruto, calibrado, fatia) in previsoes.items():
            m = metricas(y_frac, wb, calibrado)
            sinal = sinalizar_por_fatia(bruto, wb, datas, fatia)
            valores[nome].append([m["auc"], m["brier"], metricas_sinal(y_frac, wb, sinal)["f1"], beneficio_medio(y_frac, wb, calibrado)])
    rotulos = ["auc", "brier", "f1", "beneficio_medio"]
    saida = {}
    for nome, linhas in valores.items():
        matriz = np.array(linhas)
        diferenca = matriz - np.array(valores[referencia])
        saida[nome] = {r: {"baixo": float(np.percentile(matriz[:, i], 2.5)), "alto": float(np.percentile(matriz[:, i], 97.5)),
                           "dif_baixo": float(np.percentile(diferenca[:, i], 2.5)), "dif_alto": float(np.percentile(diferenca[:, i], 97.5))}
                       for i, r in enumerate(rotulos)}
    return saida


def _varredura(y_frac, w, p):
    """Precisão, recall e F1 para cada limiar possível (só nas trocas de valor de p, para respeitar empates)."""
    ordem = np.argsort(-p, kind="stable")
    ps, acerto, sinalizado = p[ordem], np.cumsum((w * y_frac)[ordem]), np.cumsum(w[ordem])
    fim_de_empate = np.r_[ps[1:] != ps[:-1], True]
    ps, acerto, sinalizado = ps[fim_de_empate], acerto[fim_de_empate], sinalizado[fim_de_empate]
    real = float(np.sum(w * y_frac))
    prec = np.divide(acerto, sinalizado, out=np.zeros_like(acerto), where=sinalizado > 0)
    rec = acerto / real if real > 0 else np.zeros_like(acerto)
    f1 = np.divide(2 * acerto, sinalizado + real, out=np.zeros_like(acerto), where=sinalizado + real > 0)
    return ps, prec, rec, f1


def ajustar_limiar(y_frac, w, p) -> float:
    """Limiar que maximiza o F1 ponderado pela exposição."""
    ps, _, _, f1 = _varredura(y_frac, w, p)
    return float(ps[int(np.argmax(f1))])


def f1_maximo(y_true, y_score, sample_weight=None) -> float:
    """Melhor F1 entre todos os limiares. Não muda com transformações monotônicas, como a calibração."""
    y, p = np.asarray(y_true, dtype=float), np.asarray(y_score, dtype=float).ravel()
    w = np.ones_like(y) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    return float(np.max(_varredura(y, w, p)[3]))


def curva_limiar(y_frac, w, p, n=60) -> list[dict]:
    ps, prec, rec, f1 = _varredura(y_frac, w, p)
    idx = np.unique(np.linspace(0, len(ps) - 1, min(n, len(ps))).astype(int))
    return [{"limiar_pct": round(float(ps[i]) * 100, 3), "precisao": round(float(prec[i]), 4),
             "recall": round(float(rec[i]), 4), "f1": round(float(f1[i]), 4)} for i in idx]


def tabela_calibracao(y_frac, w, p, n=10) -> list[dict]:
    df = pd.DataFrame({"y": y_frac, "w": w, "p": p})
    df["faixa"] = pd.qcut(df["p"].rank(method="first"), n, labels=False)
    out = df.groupby("faixa").apply(lambda g: pd.Series({
        "prev_pct": np.average(g.p, weights=g.w) * 100,
        "real_pct": np.average(g.y, weights=g.w) * 100,
        "exposicao_mi": g.w.sum(),
    }), include_groups=False).reset_index()
    return out.round(3).to_dict("records")


# ------------------------------------------------------------------ calibração
def escolher_calibrador(p, y_frac, w, datas) -> tuple[str, dict]:
    """Ajusta na primeira metade do período e compara o Brier na segunda; devolve o método de menor Brier."""
    datas = pd.Series(datas).reset_index(drop=True)
    corte = datas.sort_values().iloc[len(datas) // 2]
    a = (datas < corte).to_numpy()
    if not a.any() or a.all():  # período com um único mês: não há como separar
        a = np.arange(len(datas)) < len(datas) // 2
    b = ~a
    notas = {}
    for metodo in Calibrador.METODOS:
        cal = Calibrador(metodo).fit(p[a], y_frac[a], w[a])
        notas[metodo] = metricas(y_frac[b], w[b], cal.transform(p[b]))["brier"]
    return min(notas, key=notas.get), notas


# ------------------------------------------------------------------ modelos
def peso_recencia(datas: pd.Series, meia_vida: float) -> np.ndarray:
    """Peso que cai pela metade a cada `meia_vida` meses, contados a partir do mês mais recente."""
    meses = (datas.max() - datas).dt.days / 30.4375
    return (0.5 ** (meses / meia_vida)).to_numpy()


def _pesos(d, meia_vida):
    return d["peso"].to_numpy() * (1.0 if meia_vida is None else peso_recencia(d["data_base"], meia_vida))


def params_xgb(n, hp=None):
    return dict(objective="reg:logistic", n_estimators=n, **(hp or HP_PADRAO),
                tree_method="hist", n_jobs=-1, random_state=SEED)


def sortear_hiperparametros(n: int, seed: int = SEED) -> list[dict]:
    """A primeira tentativa é a configuração padrão; as demais são sorteadas."""
    rng = np.random.default_rng(seed)
    sorteios = [dict(learning_rate=float(10 ** rng.uniform(np.log10(0.02), np.log10(0.15))),
                     max_depth=int(rng.integers(3, 10)),
                     min_child_weight=float(10 ** rng.uniform(np.log10(0.5), np.log10(50))),
                     subsample=float(rng.uniform(0.6, 1.0)),
                     colsample_bytree=float(rng.uniform(0.5, 1.0)),
                     reg_lambda=float(10 ** rng.uniform(np.log10(0.5), np.log10(20))))
                for _ in range(n - 1)]
    return [dict(HP_PADRAO)] + sorteios


def treinar_logistica(d, numericas):
    pre = preprocessador(numericas, escalar=True).fit(d[CATEGORICAS + numericas])
    X = pre.transform(d[CATEGORICAS + numericas])
    y, peso = expandir(d["alvo_inad"].to_numpy(), d["peso"].to_numpy())
    X2 = np.vstack([X, X])
    ok = peso > 0
    mod = LogisticRegression(C=1.0, max_iter=2000).fit(X2[ok], y[ok], sample_weight=peso[ok])
    return ModeloRisco("logistica", pre, mod, numericas)


def treinar_rf(d, numericas, n_amostra=200_000):
    d = d.sample(min(n_amostra, len(d)), random_state=SEED)
    pre = preprocessador(numericas, escalar=False).fit(d[CATEGORICAS + numericas])
    mod = RandomForestRegressor(n_estimators=150, max_depth=18, min_samples_leaf=20, max_features=0.35,
                                n_jobs=-1, random_state=SEED)
    mod.fit(pre.transform(d[CATEGORICAS + numericas]), d["alvo_inad"], sample_weight=d["peso"])
    return ModeloRisco("rf", pre, mod, numericas)


def rodadas_xgb(d, numericas, com_offset, meia_vida=None, hp=None, parar_por_f1=False):
    """Early stopping com os 6 últimos meses do próprio treino como validação (log-loss ou F1)."""
    corte = d["data_base"].max() - pd.DateOffset(months=6)
    pesos = _pesos(d, meia_vida)
    marca = (d["data_base"] <= corte).to_numpy()
    a, b = d[marca], d[~marca]
    f = CATEGORICAS + numericas
    pre = preprocessador(numericas, escalar=False).fit(a[f])
    if parar_por_f1:
        parada = xgb.callback.EarlyStopping(rounds=60, metric_name="f1_maximo", maximize=True, save_best=False)
        m = xgb.XGBRegressor(**params_xgb(2000, hp), eval_metric=f1_maximo, callbacks=[parada])
    else:
        m = xgb.XGBRegressor(**params_xgb(2000, hp), early_stopping_rounds=60, eval_metric="logloss")
    m.fit(pre.transform(a[f]), a["alvo_inad"], sample_weight=pesos[marca],
          base_margin=offset_uf(a) if com_offset else None,
          eval_set=[(pre.transform(b[f]), b["alvo_inad"])], sample_weight_eval_set=[b["peso"]],
          base_margin_eval_set=[offset_uf(b)] if com_offset else None, verbose=False)
    return int(m.best_iteration) + 1


def treinar_xgb(d, numericas, com_offset, meia_vida=None, hp=None, parar_por_f1=False):
    """As rodadas saem só dos dados de `d`, para nenhuma janela enxergar o período que vai avaliar."""
    n = rodadas_xgb(d, numericas, com_offset, meia_vida, hp, parar_por_f1)
    f = CATEGORICAS + numericas
    pre = preprocessador(numericas, escalar=False).fit(d[f])
    mod = xgb.XGBRegressor(**params_xgb(n, hp))
    mod.fit(pre.transform(d[f]), d["alvo_inad"], sample_weight=_pesos(d, meia_vida),
            base_margin=offset_uf(d) if com_offset else None)
    return ModeloRisco("xgb_offset" if com_offset else "xgb", pre, mod, numericas)


def melhor_por_brier(tentativas: list[dict]) -> dict:
    return min(tentativas, key=lambda t: t["brier_validacao"])


def buscar_hiperparametros(nome, tr_val, va, numericas, com_offset, meia_vida=None, n=N_TENTATIVAS) -> dict:
    """Busca aleatória: treina em tr_val e escolhe o menor Brier na validação 1."""
    yva, wva = va["alvo_inad"].to_numpy(), va["peso"].to_numpy()
    tentativas = []
    for i, hp in enumerate(sortear_hiperparametros(n)):
        t0 = time.time()
        p = treinar_xgb(tr_val, numericas, com_offset, meia_vida, hp).prever(va)
        tentativas.append({"hp": hp, "brier_validacao": metricas(yva, wva, p)["brier"]})
        print(f"    {nome} [{i + 1}/{n}] Brier {tentativas[-1]['brier_validacao']:.5f} ({time.time() - t0:.0f}s)", flush=True)
    melhor = melhor_por_brier(tentativas)
    return {"melhor": melhor["hp"], "brier_validacao": melhor["brier_validacao"], "tentativas": tentativas}


def carregar_sintonia(caminho: Path, nomes) -> dict:
    """Devolve as buscas já salvas nos metadados, para as versões que tiverem uma."""
    if not caminho.exists():
        return {}
    salva = json.loads(caminho.read_text(encoding="utf-8")).get("sintonia", {})
    return {n: salva[n] for n in nomes if n in salva}


def escolher_por_fatia_transferida(validacao: dict, nomes) -> str:
    """Modelo de maior F1 na validação 2 usando a fatia ajustada na validação 1."""
    return max(nomes, key=lambda n: validacao[n]["f1_fatia_janela2"])


def escolher_por_f1_e_auc(validacao: dict, nomes) -> str:
    """Menor média das posições em F1 e AUC-ROC na validação 2; em empate, a melhor pior posição e depois o maior F1."""
    nomes = list(nomes)
    f1 = {n: validacao[n]["f1_fatia_janela2"] for n in nomes}
    auc = {n: validacao[n]["auc_janela2"] for n in nomes}
    posicao = lambda valores: {n: i for i, n in enumerate(sorted(nomes, key=lambda x: -valores[x]))}
    pos_f1, pos_auc = posicao(f1), posicao(auc)
    return min(nomes, key=lambda n: ((pos_f1[n] + pos_auc[n]) / 2, max(pos_f1[n], pos_auc[n]), -f1[n]))


def escolher_por_beneficio(validacao: dict, nomes, chave: str = "beneficio_medio_janela2") -> str:
    """Modelo de maior benefício líquido na validação 2, com a calibração ajustada na validação 1."""
    return max(nomes, key=lambda n: validacao[n][chave])


def caminho_checkpoint(pasta: Path, nome: str, assinatura: str) -> Path:
    """O nome muda com a base de dados, para não reaproveitar resultado de outra versão."""
    return pasta / f"{hashlib.sha1(f'{nome}|{assinatura}'.encode()).hexdigest()[:16]}.joblib"


def salvar_checkpoint(caminho: Path, conteudo: dict) -> None:
    """Grava num arquivo temporário e troca no fim, para uma interrupção não deixar arquivo pela metade."""
    temporario = caminho.with_suffix(".tmp")
    joblib.dump(conteudo, temporario, compress=3)
    os.replace(temporario, caminho)


def _partes(d):
    return d["alvo_inad"].to_numpy(), d["peso"].to_numpy()


# ------------------------------------------------------------------ principal
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refazer-busca", action="store_true",
                    help="refaz a busca de hiperparâmetros em vez de reaproveitar a salva em modelos/metadados.json")
    ap.add_argument("--criterio", choices=CRITERIOS, default="f1_auc", help="critério do modelo ativo no dashboard")
    args = ap.parse_args()
    MODELOS.mkdir(exist_ok=True)
    df = pd.read_parquet(PROC / "base_modelagem.parquet")
    df["peso"] = df["peso"].clip(upper=df["peso"].quantile(0.99))  # limita a influência das maiores carteiras

    tr_val = df[df["data_base"] <= FIM_TREINO_VAL].reset_index(drop=True)
    va = df[(df["data_base"] >= INI_VAL) & (df["data_base"] <= FIM_VAL)].reset_index(drop=True)
    tr_val2 = df[df["data_base"] <= FIM_TREINO_VAL2].reset_index(drop=True)
    va2 = df[(df["data_base"] >= INI_VAL2) & (df["data_base"] <= FIM_VAL2)].reset_index(drop=True)
    tr = df[df["data_base"] <= FIM_TREINO].reset_index(drop=True)
    te = df[df["data_base"] >= INI_TESTE].reset_index(drop=True)
    print(f"validação 1: treino {len(tr_val):,} | avaliação {len(va):,}\n"
          f"validação 2: treino {len(tr_val2):,} | avaliação {len(va2):,}\n"
          f"teste: treino {len(tr):,} | avaliação {len(te):,}")
    print(f"teste: perfis {te['data_base'].min().date()} a {te['data_base'].max().date()}")
    yva, wva = _partes(va)
    yva2, wva2 = _partes(va2)
    yte, wte = _partes(te)

    com_perfil = NUMERICAS_TODAS + PERFIL
    sem_macro_perfil = NUMERICAS_SEM_MACRO + PERFIL
    com_ampliado = com_perfil + PERFIL_AMPLIADO
    sem_macro_ampliado = sem_macro_perfil + PERFIL_AMPLIADO
    com_precos = com_ampliado + PRECOS
    sem_macro_precos = sem_macro_ampliado + PRECOS
    com_custos = com_ampliado + CUSTOS
    receitas = {
        "Regressão Logística": lambda d: treinar_logistica(d, NUMERICAS_TODAS),
        "Random Forest": lambda d: treinar_rf(d, NUMERICAS_TODAS),
        "XGBoost": lambda d: treinar_xgb(d, NUMERICAS_TODAS, False),
        "XGBoost relativo à UF": lambda d: treinar_xgb(d, NUMERICAS_SEM_MACRO, True),
        "XGBoost + histórico do perfil": lambda d: treinar_xgb(d, com_perfil, False),
        "XGBoost relativo à UF + histórico": lambda d: treinar_xgb(d, sem_macro_perfil, True),
        "XGBoost relativo à UF + histórico + recência": lambda d: treinar_xgb(d, sem_macro_perfil, True, MEIA_VIDA_MESES),
        "XGBoost + SCR ampliado": lambda d: treinar_xgb(d, com_ampliado, False),
        "XGBoost relativo à UF + SCR ampliado": lambda d: treinar_xgb(d, sem_macro_ampliado, True),
        "XGBoost + SCR ampliado + preços": lambda d: treinar_xgb(d, com_precos, False),
        "XGBoost + SCR ampliado + Conab": lambda d: treinar_xgb(d, com_custos, False),
    }

    # versões com busca de hiperparâmetros pelo Brier, feita só com as janelas de validação
    otimizar = {
        "XGBoost + histórico (Brier)": (com_perfil, False, None),
        "XGBoost relativo à UF + histórico (Brier)": (sem_macro_perfil, True, None),
        "XGBoost + SCR ampliado (Brier)": (com_ampliado, False, None),
        "XGBoost relativo à UF + SCR ampliado (Brier)": (sem_macro_ampliado, True, None),
        "XGBoost + SCR ampliado + Conab (Brier)": (com_custos, False, None),
    }
    # uma interrupção perde no máximo o modelo em andamento: cada resultado fica em modelos/checkpoint até o fim do treino
    pasta_checkpoint = MODELOS / "checkpoint"
    if args.refazer_busca:
        shutil.rmtree(pasta_checkpoint, ignore_errors=True)
    pasta_checkpoint.mkdir(exist_ok=True)
    arq_sintonia = pasta_checkpoint / "sintonia.json"
    sintonia = {} if args.refazer_busca else {**carregar_sintonia(MODELOS / "metadados.json", otimizar),
                                              **carregar_sintonia(arq_sintonia, otimizar)}
    print("hiperparâmetros: buscas reaproveitadas:", list(sintonia), flush=True)
    for nome, (numericas, com_offset, meia_vida) in otimizar.items():
        if nome not in sintonia:
            sintonia[nome] = buscar_hiperparametros(nome, tr_val, va, numericas, com_offset, meia_vida)
            arq_sintonia.write_text(json.dumps({"sintonia": sintonia}, ensure_ascii=False), encoding="utf-8")
        print(f"  {nome}: melhor Brier na validação {sintonia[nome]['brier_validacao']:.5f} {sintonia[nome]['melhor']}", flush=True)
        receitas[nome] = (lambda d, numericas=numericas, com_offset=com_offset, meia_vida=meia_vida, hp=sintonia[nome]["melhor"]:
                          treinar_xgb(d, numericas, com_offset, meia_vida, hp))
    recortes = {"PF": te["cliente"] == "PF", "PJ": te["cliente"] == "PJ",
                "Cooperativas": te["segmento"] == "Cooperativa", "Custeio": te["submodalidade"] == "Custeio"}

    def avaliar_teste(p_bruto, cal, limiar, fatia):
        p = cal.transform(p_bruto)
        bruto = metricas(yte, wte, p_bruto)
        # a ordem usa a previsão sem calibrar, porque a isotônica cria empates entre perfis
        sinal = sinalizar_por_fatia(p_bruto, wte, te["data_base"], fatia)
        fixo = metricas_limiar(yte, wte, p, limiar)
        r = {**metricas(yte, wte, p), **metricas_sinal(yte, wte, sinal),
             "brier_bruto": bruto["brier"], "inad_prevista_bruta_pct": bruto["inad_media_prevista_pct"],
             "fatia_pct": fatia * 100, "limiar_pct": limiar_da_fatia(p, wte, fatia) * 100,
             "f1_limiar_fixo": fixo["f1"], "precisao_limiar_fixo": fixo["precisao"], "recall_limiar_fixo": fixo["recall"],
             "limiar_fixo_pct": limiar * 100, "calibracao": cal.metodo, **metricas_extras(yte, wte, p)}
        for rot, mask in recortes.items():
            mask = mask.to_numpy()
            r[f"auc_{rot}"] = metricas(yte[mask], wte[mask], p[mask])["auc"]
            r[f"auprc_{rot}"] = metricas_extras(yte[mask], wte[mask], p[mask])["auprc"]
        return r, p

    # referência ingênua: sem calibração, threshold escolhido na validação
    ref = "Referência: inadimplência atual da UF"
    p_ref_va = va["uf_inad_atual_pct"].to_numpy() / 100
    lim_ref = ajustar_limiar(yva, wva, p_ref_va)
    validacao = {ref: {**metricas(yva, wva, p_ref_va), **metricas_limiar(yva, wva, p_ref_va, lim_ref)}}
    resultados = {ref: avaliar_teste(te["uf_inad_atual_pct"].to_numpy() / 100, Calibrador("nenhum"), lim_ref,
                                     fatia_do_limiar(p_ref_va, wva, lim_ref))[0]}

    calib, calib_bruta, curvas, beneficio_teste, distribuicao, arquivos_modelo = {}, {}, {}, {}, {}, {}
    info_base = (PROC / "base_modelagem.parquet").stat()
    assinatura = f"{info_base.st_size}-{info_base.st_mtime_ns}"
    for nome, fn in receitas.items():
        t0 = time.time()
        arq_modelo = caminho_checkpoint(pasta_checkpoint, f"{nome}|{json.dumps(sintonia.get(nome, {}).get('melhor'))}", assinatura)
        arquivos_modelo[nome] = arq_modelo
        if arq_modelo.exists():
            salvo = joblib.load(arq_modelo)  # arquivo gerado por este mesmo treino
            validacao[nome], resultados[nome] = salvo["validacao"], salvo["resultado"]
            calib[nome], calib_bruta[nome], curvas[nome] = salvo["calib"], salvo["calib_bruta"], salvo["curva"]
            beneficio_teste[nome], distribuicao[nome] = salvo["beneficio"], salvo["distribuicao"]
            print(f"  {nome}: reaproveitado do checkpoint", flush=True)
            continue
        p_va = fn(tr_val).prever(va)
        metodo, notas = escolher_calibrador(p_va, yva, wva, va["data_base"])
        cal = Calibrador(metodo).fit(p_va, yva, wva)
        pc_va = cal.transform(p_va)
        limiar = ajustar_limiar(yva, wva, pc_va)
        fatia = fatia_do_limiar(pc_va, wva, limiar)
        # F1 da validação, que define o modelo vencedor; o AUC sem calibrar fica como referência
        validacao[nome] = {**metricas(yva, wva, pc_va), **metricas_limiar(yva, wva, pc_va, limiar),
                           "auc_bruto": metricas(yva, wva, p_va)["auc"], "brier_por_metodo": notas}

        # segunda janela: f1_fatia_janela2 define o vencedor; os demais ficam como diagnóstico
        p_va2 = fn(tr_val2).prever(va2)
        metodo2, _ = escolher_calibrador(p_va2, yva2, wva2, va2["data_base"])
        pc_va2 = Calibrador(metodo2).fit(p_va2, yva2, wva2).transform(p_va2)
        limiar2 = ajustar_limiar(yva2, wva2, pc_va2)
        m2 = {**metricas(yva2, wva2, pc_va2), **metricas_limiar(yva2, wva2, pc_va2, limiar2)}
        validacao[nome].update({
            "f1_janela2": m2["f1"], "auc_janela2": m2["auc"], "brier_janela2": m2["brier"],
            "precisao_janela2": m2["precisao"], "recall_janela2": m2["recall"], "limiar_janela2_pct": limiar2 * 100,
            "inad_real_janela2_pct": m2["inad_media_real_pct"],
            "f1_transferido": metricas_limiar(yva2, wva2, cal.transform(p_va2), limiar)["f1"],
            "f1_medio": (validacao[nome]["f1"] + m2["f1"]) / 2,
            "fatia_pct": fatia * 100,
            "beneficio_medio_janela2": beneficio_medio(yva2, wva2, cal.transform(p_va2)),
            "beneficio_janela2": curva_beneficio(yva2, wva2, cal.transform(p_va2)),
            "brier_transferido_janela2": metricas(yva2, wva2, cal.transform(p_va2))["brier"],
            **{f"{k}_fatia_janela2": v for k, v in metricas_sinal(
                yva2, wva2, sinalizar_por_fatia(p_va2, wva2, va2["data_base"], fatia)).items()},
        })

        p_te = fn(tr).prever(te)
        resultados[nome], pc_te = avaliar_teste(p_te, cal, limiar, fatia)
        calib[nome] = tabela_calibracao(yte, wte, pc_te)
        calib_bruta[nome] = tabela_calibracao(yte, wte, p_te)
        curvas[nome] = curva_limiar(yte, wte, pc_te)
        beneficio_teste[nome] = curva_beneficio(yte, wte, pc_te)
        distribuicao[nome] = distribuicao_por_resultado(yte, wte, pc_te)
        salvar_checkpoint(arq_modelo, {
            "validacao": validacao[nome], "resultado": resultados[nome], "calib": calib[nome], "calib_bruta": calib_bruta[nome],
            "curva": curvas[nome], "beneficio": beneficio_teste[nome], "distribuicao": distribuicao[nome],
            "previsao": (p_te.astype(np.float32), pc_te.astype(np.float32), fatia)})
        print(f"  {nome}: benefício val2 {validacao[nome]['beneficio_medio_janela2']:.5f} | "
              f"F1 val1 {validacao[nome]['f1']:.4f} | val2 com a fatia {validacao[nome]['f1_fatia_janela2']:.4f} "
              f"(fatia {fatia * 100:.1f}%) | teste AUC {resultados[nome]['auc']:.4f} "
              f"Brier {resultados[nome]['brier']:.4f} F1 {resultados[nome]['f1']:.4f} "
              f"[{metodo}, limiar {limiar * 100:.2f}%] ({time.time() - t0:.0f}s)", flush=True)

    colunas = ["auc", "gini", "ks", "brier", "brier_bruto", "log_loss", "precisao", "recall", "f1", "fatia_pct",
               "limiar_pct", "f1_limiar_fixo", "pct_carteira_sinalizada", "inad_media_real_pct", "inad_media_prevista_pct",
               "inad_prevista_bruta_pct", "calibracao", "auc_PF", "auc_PJ", "auc_Cooperativas", "auc_Custeio",
               "auprc", "h_measure", "mcc", "beneficio_medio", "brier_skill", "brier_confiabilidade", "brier_resolucao",
               "calibracao_intercepto", "calibracao_inclinacao", "ece", "auprc_PF", "auprc_PJ"]
    print("\nValidação 1 (perfis jan/2024–jun/2024) e validação 2 (perfis jul/2024–dez/2024)")
    print(pd.DataFrame({k: validacao[k] for k in receitas}).T[
        ["auc", "brier", "f1", "auc_janela2", "brier_janela2", "f1_janela2", "f1_transferido", "f1_medio",
         "limiar_janela2_pct", "inad_real_janela2_pct", "fatia_pct", "f1_fatia_janela2", "precisao_fatia_janela2",
         "recall_fatia_janela2", "beneficio_medio_janela2", "brier_transferido_janela2"]].to_string())
    print(f"\nTeste fora do tempo (perfis a partir de {INI_TESTE[:7]})")
    print(pd.DataFrame(resultados).T[colunas].to_string())

    # o teste não entra na conta da escolha
    vencedores = {"f1_auc": escolher_por_f1_e_auc(validacao, receitas),
                  "f1": escolher_por_fatia_transferida(validacao, receitas),
                  "beneficio": escolher_por_beneficio(validacao, receitas)}
    por_custo = {}
    for k, alarmes in enumerate(ALARMES_POR_ACERTO):
        por_custo[str(alarmes)] = max(receitas, key=lambda n: validacao[n]["beneficio_janela2"][k]["modelo"])
    print(f"\nvencedores na validação: {vencedores}")
    print(f"vencedor do benefício líquido por alarmes aceitos para cada acerto: {por_custo}")

    # intervalos no teste para os vencedores e os dois modelos de referência da comparação
    candidatos = dict.fromkeys([*vencedores.values(), "XGBoost + SCR ampliado", "XGBoost + SCR ampliado (Brier)",
                                "XGBoost + SCR ampliado + Conab", "XGBoost + SCR ampliado + Conab (Brier)"])
    t0 = time.time()
    previsoes = {n: tuple(np.asarray(v, dtype=float) if i < 2 else v for i, v in enumerate(joblib.load(arquivos_modelo[n])["previsao"]))
                 for n in candidatos}
    intervalos = intervalos_bootstrap(yte, wte, te["data_base"], te["uf"], previsoes, vencedores[args.criterio])
    del previsoes
    print(f"intervalos de 95% ({time.time() - t0:.0f}s):")
    print(pd.DataFrame({n: {f"{m}_{k}": round(v[k], 4) for m, v in r.items() for k in ("baixo", "alto", "dif_baixo", "dif_alto")}
                        for n, r in intervalos.items()}).T.to_string())

    # ---------------- produção: mesmo procedimento no fim da base
    data_max = df["data_base"].max()
    fim_interno = data_max - pd.DateOffset(months=12)
    ini_janela = data_max - pd.DateOffset(months=6)
    jan = df[df["data_base"] > ini_janela].reset_index(drop=True)
    yj, wj = jan["alvo_inad"].to_numpy(), jan["peso"].to_numpy()
    atual = pd.read_parquet(PROC / "base_atual.parquet")
    amostra = df.sample(3000, random_state=SEED)

    def gerar_final(nome):
        p_jan = receitas[nome](df[df["data_base"] <= fim_interno].reset_index(drop=True)).prever(jan)
        metodo_final, notas_final = escolher_calibrador(p_jan, yj, wj, jan["data_base"])
        cal_final = Calibrador(metodo_final).fit(p_jan, yj, wj)
        pc_jan = cal_final.transform(p_jan)
        fatia_final = fatia_do_limiar(pc_jan, wj, ajustar_limiar(yj, wj, pc_jan))
        final = receitas[nome](df)
        # o threshold exibido é o que produz a fatia nos perfis do último mês, ainda sem alvo conhecido
        limiar_final = limiar_da_fatia(cal_final.transform(final.prever(atual)), atual["peso"].to_numpy(), fatia_final)
        print(f"produção ({nome}): calibração {metodo_final} {notas_final}; fatia {fatia_final * 100:.1f}%; "
              f"limiar {limiar_final * 100:.2f}% (janela {jan['data_base'].min().date()} a {jan['data_base'].max().date()})")
        final.preparar_explicador(amostra)
        final.calibrador, final.limiar, final.fatia_pct = cal_final, limiar_final, fatia_final * 100
        importancia = final.explicar(amostra).abs().mean().sort_values(ascending=False)
        print(importancia.round(4).to_string())
        return final, importancia, {
            "calibracao": metodo_final, "limiar_pct": limiar_final * 100, "fatia_pct": fatia_final * 100,
            "brier_por_metodo": notas_final, "janela_de": str(jan["data_base"].min().date()),
            "janela_ate": str(jan["data_base"].max().date())}

    finais = {nome: gerar_final(nome) for nome in dict.fromkeys(vencedores.values())}

    opcoes = {c: sorted(df[c].dropna().unique().tolist()) for c in CATEGORICAS}
    for cli in ["PF", "PJ"]:
        sub = df[df["cliente"] == cli]
        opcoes[f"porte_{cli}"] = sorted(sub["porte"].unique().tolist())
        opcoes[f"ocupacao_{cli}"] = sorted(sub["ocupacao_atividade"].unique().tolist())

    hist = (pd.read_parquet(RAIZ / "dados" / "intermediarios" / "scr_rural.parquet")
              .groupby(["data_base", "cliente"])[["carteira_ativa", "carteira_inadimplencia"]].sum()
              .reset_index())
    hist["inad_pct"] = hist["carteira_inadimplencia"] / hist["carteira_ativa"] * 100

    comum = {
        "horizonte_meses": 6, "treino_ate": FIM_TREINO, "teste_desde": INI_TESTE,
        "treino_validacao_ate": FIM_TREINO_VAL, "validacao_de": INI_VAL, "validacao_ate": FIM_VAL,
        "dados_ate": str(data_max.date()),
        "n_treino": int(len(tr)), "n_teste": int(len(te)), "n_total": int(len(df)),
        "validacao2_de": INI_VAL2, "validacao2_ate": FIM_VAL2,
        "vencedores_por_criterio": vencedores, "vencedor_beneficio_por_custo": por_custo, "alarmes_por_acerto": list(ALARMES_POR_ACERTO),
        "comparacao": resultados, "validacao": validacao, "sintonia": sintonia, "intervalos_teste": intervalos,
        "calibracao_teste": calib, "calibracao_teste_bruta": calib_bruta, "curva_limiar_teste": curvas,
        "beneficio_teste": beneficio_teste, "distribuicao_teste": distribuicao,
        "opcoes": opcoes,
        "historico": [{"data": str(r.data_base.date()), "cliente": r.cliente, "inad_pct": round(r.inad_pct, 3),
                       "carteira_bi": round(r.carteira_ativa / 1e9, 2)} for r in hist.itertuples()],
    }
    for criterio, nome in vencedores.items():
        final, importancia, producao = finais[nome]
        pasta = MODELOS / "por_criterio" / criterio
        pasta.mkdir(parents=True, exist_ok=True)
        joblib.dump(final, pasta / "modelo_final.joblib", compress=3)
        meta = {"modelo_escolhido": nome, "tipo": final.tipo, "features": final.features, "criterio_escolha": criterio,
                "producao": producao, "importancia_shap_global": importancia.round(5).to_dict(), **comum}
        (pasta / "metadados.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    trocar(args.criterio, MODELOS)
    shutil.rmtree(pasta_checkpoint, ignore_errors=True)
    print(f"salvo em {MODELOS}; critério ativo: {args.criterio} ({vencedores[args.criterio]})")


if __name__ == "__main__":
    main()
