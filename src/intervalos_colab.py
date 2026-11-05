"""
Intervalos de confiança das métricas da escolha (PR-AUC, precisão, recall e F1) para os resultados do Colab.

Lê as previsões salvas em modelos/colab, refaz os ensembles com o mesmo código e confere se as métricas batem com a tabela do
Colab antes de reamostrar. A reamostragem sorteia grupos de UF e mês, que não capturam toda a dependência entre os meses.
Uso: python src/intervalos_colab.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import otimizacao as o
from treino import SEED, metricas_sinal, sinalizar_por_fatia

RAIZ = Path(__file__).resolve().parents[1]
PASTA = RAIZ / "modelos" / "colab"
MODELOS = ("xgboost", "catboost", "random_forest", "lightgbm")


def carregar_previsoes(janelas: dict) -> dict:
    previsoes = {}
    for m in MODELOS:
        with np.load(PASTA / f"previsoes_{m}.npz") as z:
            previsoes[m] = {k: z[k].astype(float) for k in ("val1", "val2", "teste")}
        for k in previsoes[m]:
            assert len(previsoes[m][k]) == len(janelas[k]), f"{m}/{k}: tamanho não bate com a base"
    return previsoes


def metricas_escolha(y, w, datas, bruto, calibrado, fatia) -> dict:
    sinal = sinalizar_por_fatia(bruto, w, datas, fatia)
    return {"pr_auc": o.pr_auc(y, w, calibrado), **{k: v for k, v in metricas_sinal(y, w, sinal).items()
                                                    if k in ("precisao", "recall", "f1")}}


def main(n: int = 200):
    janelas = o.carregar_janelas(RAIZ / "dados" / "processados" / "base_modelagem.parquet")
    previsoes = carregar_previsoes(janelas)
    comb = o.combinar_ensembles(previsoes, janelas)
    previsoes |= comb["previsoes"]
    colab = pd.read_csv(PASTA / "resultados_optuna.csv", index_col=0)

    te = janelas["teste"]
    y, w, datas = te["alvo_inad"].to_numpy(), te["peso"].to_numpy(), te["data_base"].to_numpy()
    preparadas, conferencia = {}, {}
    for nome, prev in previsoes.items():
        r = o.avaliar(prev, janelas)
        conferencia[nome] = {m: (round(r[m], 4), round(float(colab.loc[nome, m]), 4)) for m in ("pr_auc", "precisao", "recall", "f1")}
        from modelo import Calibrador
        from treino import escolher_calibrador
        y1, w1 = janelas["val1"]["alvo_inad"].to_numpy(), janelas["val1"]["peso"].to_numpy()
        metodo, _ = escolher_calibrador(prev["val1"], y1, w1, janelas["val1"]["data_base"])
        cal = Calibrador(metodo).fit(prev["val1"], y1, w1)
        preparadas[nome] = (prev["teste"], cal.transform(prev["teste"]), r["fatia_pct"] / 100)
    print("conferência com o Colab (recalculado, Colab):")
    for nome, c in conferencia.items():
        print(f"  {nome:20s} {c}")
    diverge = {n: c for n, c in conferencia.items() if any(abs(a - b) > 0.002 for a, b in c.values())}
    if diverge:
        raise SystemExit(f"as métricas não batem com o Colab; a comparação não vale: {diverge}")

    grupo = pd.factorize(pd.Series(datas).astype(str) + "|" + te["uf"].astype(str).to_numpy())[0]
    n_grupos = grupo.max() + 1
    rng = np.random.default_rng(SEED)
    valores = {nome: [] for nome in preparadas}
    t0 = time.time()
    for i in range(n):
        wb = w * rng.multinomial(n_grupos, np.full(n_grupos, 1 / n_grupos))[grupo]
        for nome, (bruto, calibrado, fatia) in preparadas.items():
            valores[nome].append(metricas_escolha(y, wb, datas, bruto, calibrado, fatia))
        if i == 9:
            print(f"10 reamostragens em {time.time() - t0:.0f}s", flush=True)

    saida = {}
    for referencia in ("ensemble_empilhado", "lightgbm"):
        base = pd.DataFrame(valores[referencia])
        for nome in preparadas:
            atual = pd.DataFrame(valores[nome])
            dif = atual - base
            saida.setdefault(referencia, {})[nome] = {
                m: {"valor_baixo": float(atual[m].quantile(0.025)), "valor_alto": float(atual[m].quantile(0.975)),
                    "dif_baixo": float(dif[m].quantile(0.025)), "dif_alto": float(dif[m].quantile(0.975)),
                    "prob_melhor": float((dif[m] > 0).mean())} for m in ("pr_auc", "precisao", "recall", "f1")}
    (PASTA / "intervalos_colab.json").write_text(json.dumps(saida, ensure_ascii=False, indent=1), encoding="utf-8")
    for referencia, tab in saida.items():
        print(f"\ndiferença para {referencia} (intervalo de 95%; prob = fração das reamostragens em que o modelo foi melhor):")
        for nome, r in tab.items():
            print(f"  {nome:20s} " + " | ".join(f"{m} {v['dif_baixo']:+.4f} a {v['dif_alto']:+.4f} (prob {v['prob_melhor']:.2f})"
                                                for m, v in r.items()))


if __name__ == "__main__":
    main()
