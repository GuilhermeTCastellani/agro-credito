"""
Segmentos críticos: quais perfis e quais segmentos UF × finalidade devem passar de um limite de inadimplência em 6 meses.

É o produto principal: em vez de ordenar reais de carteira, classifica perfis (e segmentos) acima de 5%, 10% ou 20%. Os níveis
de alerta são atenção (5% a 10%) e crítico (10% ou mais). Um perfil ou segmento é sinalizado quando a inadimplência prevista (ponderada pela exposição) passa do limite.
A medição no teste compara com duas regras simples: sinalizar tudo e repetir o nível atual de inadimplência.

Uso: python src/segmentos.py   (grava a medição em modelos/metadados.json, na chave "segmentos_teste")
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

LIMITES = (0.05, 0.10, 0.20)
# Produto principal: nível de alerta pela inadimplência prevista do perfil em 6 meses.
LIMITE_ATENCAO = 0.05   # atenção: de 5% a 10%
LIMITE_CRITICO = 0.10   # crítico: 10% ou mais
LIMITE_PRINCIPAL = LIMITE_CRITICO
NIVEIS_ALERTA = ("crítico", "atenção", "normal")


def nivel_alerta(previsto) -> np.ndarray:
    """'crítico' a partir de 10%, 'atenção' de 5% a 10% e 'normal' abaixo de 5% (inadimplência prevista, em fração)."""
    p = np.asarray(previsto, float)
    return np.select([p >= LIMITE_CRITICO, p >= LIMITE_ATENCAO], ["crítico", "atenção"], "normal")
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


def acompanhamento(df: pd.DataFrame, horizonte: int = 6) -> list[dict]:
    """Inadimplência atual, prevista e real por mês do alvo, no Brasil ("BR") e em cada UF, ponderadas pela exposição.

    `df` tem data_base, uf, peso, janela e real, previsto e atual (frações). O mês do alvo é `horizonte` meses depois da data-base.
    """
    d = df.assign(mes_alvo=(pd.to_datetime(df["data_base"]) + pd.offsets.MonthEnd(horizonte)).dt.strftime("%Y-%m"))
    saida = []
    for uf, g in [("BR", d)] + list(d.groupby("uf", observed=True)):
        a = agregar(g, chaves=["mes_alvo", "janela"], mes=None).sort_values("mes_alvo")
        saida += [{"uf": str(uf), "mes_alvo": r.mes_alvo, "janela": r.janela, "carteira": float(r.peso),
                   "atual_pct": float(r.atual * 100), "previsto_pct": float(r.previsto * 100),
                   "real_pct": float(r.real * 100) if "real" in a.columns else None}
                  for r in a.itertuples()]
    return saida


def perfis_recentes(meses: int = 6) -> pd.DataFrame:
    """Perfis dos últimos `meses` meses do SCR, ainda sem alvo conhecido, montados como em features.montar_atual."""
    import features as f
    scr = f._operacao(f.historico_perfil(f.ler_scr()))
    ultimos = sorted(scr["data_base"].unique())[-meses:]
    base = scr[scr["data_base"].isin(ultimos)].copy()
    base["alvo_inad"] = np.nan
    base["peso"] = base["carteira_ativa"] / 1e6
    return f._contexto(base, scr)


def main():
    import sys
    raiz = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(raiz / "src"))
    import intervalos_colab as ic
    import otimizacao as o
    from modelo import Calibrador
    from treino import escolher_calibrador

    from treino import FIM_VAL, FIM_VAL2, INI_TESTE, INI_VAL, INI_VAL2
    caminho_base = raiz / "dados" / "processados" / "base_modelagem.parquet"
    janelas = o.carregar_janelas(caminho_base)
    # o acompanhamento pesa pela carteira sem o teto do percentil 99, para o nível bater com a taxa oficial
    bruto = pd.read_parquet(caminho_base, columns=["data_base", "peso"])
    d = bruto["data_base"]
    peso_bruto = {"val1": bruto.loc[(d >= INI_VAL) & (d <= FIM_VAL), "peso"].to_numpy(),
                  "val2": bruto.loc[(d >= INI_VAL2) & (d <= FIM_VAL2), "peso"].to_numpy(),
                  "teste": bruto.loc[d >= INI_TESTE, "peso"].to_numpy()}
    del bruto, d
    prev = ic.carregar_previsoes(janelas)
    caminho = raiz / "modelos" / "metadados.json"
    combinacao = json.loads(caminho.read_text(encoding="utf-8")).get("producao", {}).get("combinacao", "empilhado")
    if combinacao == "media_logit":  # média simples das log-chances dos quatro modelos, sem pesos ajustados
        from modelo import logit
        p = {j: 1 / (1 + np.exp(-np.mean([logit(prev[n][j]) for n in prev], axis=0))) for j in ("val1", "val2", "teste")}
        nota_pesos = "a média não tem pesos ajustados, então só a val1 (calibração) não é totalmente fora da amostra"
    else:
        p = o.combinar_ensembles(prev, janelas)["previsoes"]["ensemble_empilhado"]
        nota_pesos = "os pesos do empilhamento, da val2, então essas duas janelas não são totalmente fora da amostra"
    v1 = janelas["val1"]
    y1, w1 = v1["alvo_inad"].to_numpy(), v1["peso"].to_numpy()
    metodo, _ = escolher_calibrador(p["val1"], y1, w1, v1["data_base"])
    cal = Calibrador(metodo).fit(p["val1"], y1, w1)

    def montar(jan, sem_teto=False):
        d = janelas[jan]
        assert len(d) == len(peso_bruto[jan]), f"{jan}: o corte do peso sem teto não bate com a janela"
        return d[["data_base", "peso"] + CHAVES_SEGMENTO].assign(
            real=d["alvo_inad"], previsto=cal.transform(p[jan]), atual=d["perfil_inad_atual_pct"] / 100, janela=jan,
            **({"peso": peso_bruto[jan]} if sem_teto else {}))

    linhas = avaliar(montar("teste"))
    serie = acompanhamento(pd.concat([montar(j, sem_teto=True) for j in ("val1", "val2", "teste")], ignore_index=True))
    del janelas, prev, p
    from modelo import carregar_modelo
    recentes = perfis_recentes()
    proj = recentes[["data_base", "uf", "peso"]].assign(
        previsto=carregar_modelo(raiz).prever(recentes), atual=recentes["perfil_inad_atual_pct"] / 100, janela="projecao")
    serie_proj = acompanhamento(proj)
    meta = json.loads(caminho.read_text(encoding="utf-8"))
    meta["segmentos_teste"] = {"modelo": meta.get("modelo_escolhido"), "regra": "previsto >= limite", "linhas": linhas}
    meta["projecao"] = {"modelo": meta.get("modelo_escolhido"), "datas_base": [str(d.date()) for d in sorted(recentes["data_base"].unique())],
                        "nota": "modelo em produção (todo o histórico, calibração dos meses finais) aplicado aos perfis dos últimos 6 meses, "
                                "ainda sem resultado; pesa pela carteira atual do perfil, sem teto",
                        "linhas": serie_proj}
    meta["acompanhamento"] = {"modelo": meta.get("modelo_escolhido"), "calibracao": metodo,
                              "nota": "modelos treinados antes de cada janela (val1, val2 e teste); a calibração vem da val1; "
                                      + nota_pesos + "; pesa pela carteira no mês do alvo, sem o teto do percentil 99",
                              "linhas": serie}
    caminho.write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    for r in linhas:
        if r["pesa"] == "reais" or r["nivel"] == "segmento":
            print(f"{r['nivel']:9s} >= {r['limite']:.0%} ({r['pesa']}): F1 modelo {r['modelo']['f1']:.3f} | nível atual "
                  f"{r['nivel_atual']['f1']:.3f} | sinalizar tudo {r['sinalizar_tudo']['f1']:.3f}")


if __name__ == "__main__":
    main()
