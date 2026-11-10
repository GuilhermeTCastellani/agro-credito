"""
Compara o ensemble empilhado em produção com a média simples dos quatro modelos, sem trocar o modelo ativo do dashboard.

Motivo: os pesos do empilhamento foram ajustados na validação 2 (2024), com modelos defasados que previam baixo numa alta da
inadimplência. Eles somam 1,12 e têm intercepto positivo, o que amplifica o nível quando aplicados aos modelos de produção,
treinados até o fim. A média não tem pesos para distorcer.

A média é feita nas log-chances (peso 0,25 para cada modelo e intercepto zero): é o mesmo ModeloEnsemble do dashboard, com
a explicação SHAP exata na parte que soma. A média das probabilidades entra só como comparação de nível.

Compara três coisas:
  A. Teste fora da amostra (modelos treinados até jun/2024, calibração da validação 1): PR-AUC, AUC, Brier, alertas por perfil
     em 5% e 10% e o erro de nível mês a mês nas janelas de validação e teste.
  B. Modelo de produção nos 12 meses mais recentes com resultado (dentro da amostra: os componentes treinaram com eles).
  C. Projeção para os 6 meses seguintes e a contagem de alertas na carteira atual.

Os modelos "internos" que geraram a calibração de produção (treinados 12 meses antes) não estão neste PC, então a média
candidata vai sem calibrador. A fatia vem das previsões fora da amostra da janela de teste nos 6 meses mais recentes.

Grava modelos/ensemble/media/ensemble.joblib (só o resumo; os componentes são os de modelos/ensemble/final) e
modelos/ensemble/media/comparacao.json. Uso: python src/comparar_media.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import intervalos_colab as ic
import otimizacao as o
import segmentos as s
from modelo import Calibrador, ModeloEnsemble, carregar_modelo, logit
from treino import (FIM_VAL, FIM_VAL2, INI_TESTE, INI_VAL, INI_VAL2, SEED, ajustar_limiar, escolher_calibrador,
                    fatia_do_limiar, limiar_da_fatia)

RAIZ = Path(__file__).resolve().parents[1]
BASE = RAIZ / "dados" / "processados" / "base_modelagem.parquet"
DESTINO = RAIZ / "modelos" / "ensemble" / "media"
NOMES = o.MODELOS_ENSEMBLE
JANELAS = ("val1", "val2", "teste")


def sigmoide(x):
    return 1 / (1 + np.exp(-x))


def media_logit(previsoes: list) -> np.ndarray:
    return sigmoide(np.mean([logit(p) for p in previsoes], axis=0))


def nivel_por_mes(datas, peso, real, previsto, horizonte: int = 6) -> pd.DataFrame:
    """Inadimplência real e prevista no Brasil por mês do alvo, ponderada pela carteira."""
    d = pd.DataFrame({"mes_alvo": (pd.to_datetime(datas) + pd.offsets.MonthEnd(horizonte)).dt.strftime("%Y-%m"),
                      "w": peso, "real": real, "prev": previsto})
    g = d.groupby("mes_alvo").apply(lambda x: pd.Series({"real_pct": np.average(x["real"], weights=x["w"]) * 100,
                                                         "previsto_pct": np.average(x["prev"], weights=x["w"]) * 100}),
                                    include_groups=False)
    g["erro_pp"] = g["previsto_pct"] - g["real_pct"]
    return g


def resumo_erro(g: pd.DataFrame) -> dict:
    return {"erro_medio_absoluto_pp": float(g["erro_pp"].abs().mean()), "vies_medio_pp": float(g["erro_pp"].mean()),
            "maior_erro_pp": float(g["erro_pp"].iloc[g["erro_pp"].abs().argmax()]), "meses": int(len(g))}


def parte_a(janelas: dict, previsoes: dict) -> dict:
    """Teste fora da amostra, com a calibração e a fatia da validação 1, como no restante do projeto."""
    bruto = pd.read_parquet(BASE, columns=["data_base", "peso"])
    d = bruto["data_base"]
    peso_bruto = {"val1": bruto.loc[(d >= INI_VAL) & (d <= FIM_VAL), "peso"].to_numpy(),
                  "val2": bruto.loc[(d >= INI_VAL2) & (d <= FIM_VAL2), "peso"].to_numpy(),
                  "teste": bruto.loc[d >= INI_TESTE, "peso"].to_numpy()}
    v1 = janelas["val1"]
    y1, w1 = v1["alvo_inad"].to_numpy(), v1["peso"].to_numpy()
    saida = {}
    for nome, p in previsoes.items():
        r = o.avaliar(p, janelas)
        metodo, _ = escolher_calibrador(p["val1"], y1, w1, v1["data_base"])
        cal = Calibrador(metodo).fit(p["val1"], y1, w1)
        te = janelas["teste"]
        df = te[["data_base", "peso"] + s.CHAVES_SEGMENTO].assign(
            real=te["alvo_inad"], previsto=cal.transform(p["teste"]), atual=te["perfil_inad_atual_pct"] / 100)
        alertas = {f"{int(lin['limite'] * 100)}%": lin["modelo"] for lin in s.avaliar(df)
                   if lin["nivel"] == "perfil" and lin["pesa"] == "reais" and lin["limite"] in (0.05, 0.10)}
        nivel = pd.concat([nivel_por_mes(janelas[j]["data_base"], peso_bruto[j], janelas[j]["alvo_inad"],
                                         cal.transform(p[j])) for j in JANELAS])
        saida[nome] = {"calibracao": metodo, "pr_auc": r["pr_auc"], "auc": r["auc"], "brier": r["brier"],
                       "alertas_teste": alertas, "nivel_validacao_e_teste": resumo_erro(nivel),
                       "nivel_por_mes": nivel.round(4).reset_index().to_dict("records")}
    return saida


def main():
    t0 = time.time()
    janelas = o.carregar_janelas(BASE)
    prev = ic.carregar_previsoes(janelas)
    comb = o.combinar_ensembles(prev, janelas)["previsoes"]
    candidatos = {"empilhado": comb["ensemble_empilhado"], "media_prob": comb["ensemble_media"],
                  "media_logit": {j: media_logit([prev[n][j] for n in NOMES]) for j in JANELAS}}
    a = parte_a(janelas, candidatos)
    print(f"A. teste fora da amostra pronto ({time.time() - t0:.0f}s)", flush=True)

    # fatia da média: previsões fora da amostra da janela de teste nos 6 meses mais recentes, sem calibrar
    te = janelas["teste"]
    recente = (te["data_base"] > te["data_base"].max() - pd.DateOffset(months=6)).to_numpy()
    y_j, w_j = te["alvo_inad"].to_numpy()[recente], te["peso"].to_numpy()[recente]
    p_j = candidatos["media_logit"]["teste"][recente]
    fatia = fatia_do_limiar(p_j, w_j, ajustar_limiar(y_j, w_j, p_j))
    del janelas, prev, comb, candidatos

    atual_modelo = carregar_modelo(RAIZ)
    assert getattr(atual_modelo, "tipo", "") == "ensemble", "o modelo ativo não é o ensemble"
    media = ModeloEnsemble(atual_modelo.componentes, {n: 0.25 for n in NOMES}, 0.0)

    def variantes(df) -> dict:
        lg = {n: logit(atual_modelo.componentes[n].prever(df)) for n in NOMES}
        bruto = sigmoide(atual_modelo.intercepto + sum(c * lg[n] for n, c in atual_modelo.coeficientes.items()))
        return {"empilhado": atual_modelo.calibrador.transform(bruto) if atual_modelo.calibrador else bruto,
                "media_logit": sigmoide(np.mean(list(lg.values()), axis=0)),
                "media_prob": np.mean([sigmoide(v) for v in lg.values()], axis=0)}

    # B. produção nos 12 meses mais recentes com resultado (alvos set/2025 a ago/2026)
    base = pd.read_parquet(BASE)
    base = base[base["data_base"] > base["data_base"].max() - pd.DateOffset(months=12)].reset_index(drop=True)
    vb = variantes(base)
    b = {k: nivel_por_mes(base["data_base"], base["peso"], base["alvo_inad"], v) for k, v in vb.items()}
    del base
    print(f"B. produção nos meses recentes pronto ({time.time() - t0:.0f}s)", flush=True)

    # C. projeção e alertas
    recentes = s.perfis_recentes()
    vr = variantes(recentes)
    proj = {k: nivel_por_mes(recentes["data_base"], recentes["peso"], np.zeros(len(recentes)), v)["previsto_pct"]
            for k, v in vr.items()}
    del recentes
    atual = pd.read_parquet(RAIZ / "dados" / "processados" / "base_atual.parquet")
    va = variantes(atual)
    w = atual["peso"].to_numpy()
    alertas = {}
    for k, v in va.items():
        nivel = s.nivel_alerta(v)
        alertas[k] = {n: {"perfis": int((nivel == n).sum()), "carteira_pct": float(w[nivel == n].sum() / w.sum() * 100)}
                      for n in s.NIVEIS_ALERTA}
        alertas[k]["inad_prevista_pct"] = float(np.average(v, weights=w) * 100)
    print(f"C. projeção e alertas prontos ({time.time() - t0:.0f}s)", flush=True)

    media.fatia_pct = fatia * 100
    media.limiar = limiar_da_fatia(va["media_logit"], w, fatia)
    amostra = atual.sample(300, random_state=SEED).reset_index(drop=True)
    media.preparar_explicador(amostra)
    importancia = media.explicar(amostra).abs().mean().sort_values(ascending=False)
    DESTINO.mkdir(parents=True, exist_ok=True)
    media.salvar(DESTINO, componentes=False)
    print(f"SHAP pronto ({time.time() - t0:.0f}s)", flush=True)

    resultado = {
        "gerado_em": pd.Timestamp.now().isoformat(timespec="seconds"),
        "teste_fora_da_amostra": a,
        "producao_meses_recentes": {k: {**resumo_erro(g), "por_mes": g.round(4).reset_index().to_dict("records")}
                                    for k, g in b.items()},
        "projecao": {k: v.round(4).to_dict() for k, v in proj.items()},
        "alertas_carteira_atual": alertas,
        "candidato_media_logit": {"calibracao": "nenhuma", "fatia_pct": fatia * 100, "limiar_pct": media.limiar * 100,
                                  "valor_base": media.valor_base, "erro_aditividade": media.erro_aditividade,
                                  "importancia_shap_global": importancia.round(5).to_dict()},
        "notas": ["B é dentro da amostra: os componentes de produção treinaram com esses meses.",
                  "A média candidata vai sem calibrador, porque os modelos internos da calibração de produção não estão "
                  "neste PC; a fatia vem das previsões fora da amostra da janela de teste nos 6 meses mais recentes.",
                  "Níveis ponderados pela carteira, sem o teto do percentil 99, exceto as métricas de A (PR-AUC, AUC, "
                  "Brier, alertas), que usam o teto do treino."],
    }
    (DESTINO / "comparacao.json").write_text(json.dumps(resultado, ensure_ascii=False, indent=1, default=float),
                                             encoding="utf-8")

    print("\nA. Teste fora da amostra (calibração da validação 1)")
    for k, r in a.items():
        al = r["alertas_teste"]
        print(f"  {k:12s} PR-AUC {r['pr_auc']:.4f} AUC {r['auc']:.4f} Brier {r['brier']:.5f} | F1 ≥10% {al['10%']['f1']:.3f} "
              f"≥5% {al['5%']['f1']:.3f} | nível: erro abs {r['nivel_validacao_e_teste']['erro_medio_absoluto_pp']:.2f} p.p., "
              f"viés {r['nivel_validacao_e_teste']['vies_medio_pp']:+.2f}")
    print("\nB. Produção, alvos set/2025–ago/2026 (dentro da amostra)")
    print(pd.DataFrame({k: g["previsto_pct"] for k, g in b.items()}).assign(real=b["empilhado"]["real_pct"]).round(2).to_string())
    for k, g in b.items():
        e = resumo_erro(g)
        print(f"  {k:12s} erro abs {e['erro_medio_absoluto_pp']:.2f} p.p., viés {e['vies_medio_pp']:+.2f}")
    print("\nC. Projeção (alvos set/2026–fev/2027)")
    print(pd.DataFrame(proj).round(2).to_string())
    print("\nAlertas na carteira de ago/2026 (perfis e % da carteira)")
    for k, al in alertas.items():
        print(f"  {k:12s} inad. prevista {al['inad_prevista_pct']:.2f}% | " +
              " | ".join(f"{n}: {al[n]['perfis']} ({al[n]['carteira_pct']:.1f}%)" for n in s.NIVEIS_ALERTA))
    print(f"\nmédia candidata: fatia {fatia * 100:.1f}%, limiar {media.limiar * 100:.2f}%, valor_base {media.valor_base:.3f}, "
          f"erro da soma (p95) {media.erro_aditividade:.4f}; total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
