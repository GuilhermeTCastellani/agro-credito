"""
Segmentos críticos: quais perfis e quais segmentos UF × finalidade devem passar de um limite de inadimplência em 6 meses.

É uma pergunta diferente da do modelo principal: em vez de ordenar reais de carteira, classifica perfis ou segmentos acima de
5%, 10% ou 20%. Um perfil ou segmento é sinalizado quando a inadimplência prevista (ponderada pela exposição) passa do limite.
A medição no teste compara com duas regras simples: sinalizar tudo e repetir o nível atual de inadimplência.

Uso: python src/segmentos.py   (grava a medição em modelos/metadados.json, na chave "segmentos_teste")
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

LIMITES = (0.05, 0.10, 0.20)
CHAVES_SEGMENTO = ["uf", "submodalidade"]


def agregar(df: pd.DataFrame, chaves=CHAVES_SEGMENTO, mes: str | None = "data_base") -> pd.DataFrame:
    """Média ponderada pela exposição de `previsto`, `atual` e, se houver, `real`, por segmento (e mês)."""
    grupos = ([mes] if mes else []) + list(chaves)
    cols = [c for c in ("previsto", "atual", "real") if c in df.columns]
    pesado = df[grupos].copy()
    for c in cols:
        pesado[c] = df[c] * df["peso"]
    pesado["peso"] = df["peso"]
    pesado["perfis"] = 1
    g = pesado.groupby(grupos, as_index=False)[cols + ["peso", "perfis"]].sum()
    for c in cols:
        g[c] = g[c] / g["peso"]
    return g


def medir(real, sinal, w) -> dict:
    """Precisão, recall e F1 de quem passa do limite, ponderados por `w` (exposição ou 1 por linha)."""
    real, sinal, w = np.asarray(real, bool), np.asarray(sinal, bool), np.asarray(w, float)
    tp, fp, fn = w[real & sinal].sum(), w[~real & sinal].sum(), w[real & ~sinal].sum()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precisao": float(prec), "recall": float(rec), "f1": float(f1), "prevalencia": float(w[real].sum() / w.sum()),
            "sinalizados": float(w[sinal].sum() / w.sum())}


def avaliar(df: pd.DataFrame) -> list[dict]:
    """`df` tem real, previsto e atual (frações), peso e as chaves. Mede perfis e segmentos para cada limite."""
    niveis = {"perfil": df, "segmento": agregar(df)}
    linhas = []
    for nivel, d in niveis.items():
        for limite in LIMITES:
            real = d["real"] >= limite
            for pesa, w in (("reais", d["peso"]), ("contagem", np.ones(len(d)))):
                linhas.append({"nivel": nivel, "limite": limite, "pesa": pesa,
                               "modelo": medir(real, d["previsto"] >= limite, w),
                               "nivel_atual": medir(real, d["atual"] >= limite, w),
                               "sinalizar_tudo": medir(real, np.ones(len(d), bool), w)})
    return linhas


def main():
    import sys
    raiz = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(raiz / "src"))
    import intervalos_colab as ic
    import otimizacao as o
    from modelo import Calibrador
    from treino import escolher_calibrador

    janelas = o.carregar_janelas(raiz / "dados" / "processados" / "base_modelagem.parquet")
    prev = ic.carregar_previsoes(janelas)
    p = o.combinar_ensembles(prev, janelas)["previsoes"]["ensemble_empilhado"]
    v1 = janelas["val1"]
    y1, w1 = v1["alvo_inad"].to_numpy(), v1["peso"].to_numpy()
    metodo, _ = escolher_calibrador(p["val1"], y1, w1, v1["data_base"])
    te = janelas["teste"]
    df = te[["data_base", "peso"] + CHAVES_SEGMENTO].assign(
        real=te["alvo_inad"], previsto=Calibrador(metodo).fit(p["val1"], y1, w1).transform(p["teste"]),
        atual=te["perfil_inad_atual_pct"] / 100)
    linhas = avaliar(df)
    caminho = raiz / "modelos" / "metadados.json"
    meta = json.loads(caminho.read_text(encoding="utf-8"))
    meta["segmentos_teste"] = {"modelo": meta.get("modelo_escolhido"), "regra": "previsto >= limite", "linhas": linhas}
    caminho.write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    for r in linhas:
        if r["pesa"] == "reais" or r["nivel"] == "segmento":
            print(f"{r['nivel']:9s} >= {r['limite']:.0%} ({r['pesa']}): F1 modelo {r['modelo']['f1']:.3f} | nível atual "
                  f"{r['nivel_atual']['f1']:.3f} | sinalizar tudo {r['sinalizar_tudo']['f1']:.3f}")


if __name__ == "__main__":
    main()
