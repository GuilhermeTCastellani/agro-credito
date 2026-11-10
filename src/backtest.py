"""
Backtest do procedimento de produção, com retreino trimestral.

Pergunta: como o modelo que roda no dashboard teria se saído no passado, e qual forma de combinar os quatro modelos acerta melhor o
nível? O teste do projeto mede um modelo treinado até jun/2024; a produção usa modelos treinados até o fim da base, combinados por
pesos ajustados em 2024. Aqui cada origem t imita a produção no mês t:

  treino   : perfis com data-base até t − 6 meses (o resultado deles já saiu em t); o teto do peso vem só desse treino
  previsão : perfis de t, t + 1 e t + 2, com o mesmo modelo (retreino a cada 3 meses)
  resultado: conhecido 6 meses depois de cada data-base

Parte 1, no Colab (`rodar`): treina os quatro modelos em cada origem, com os hiperparâmetros já buscados no Optuna
(modelos/hiperparametros_ensemble.json), e salva as previsões em `pasta`; rodar de novo continua de onde parou.
Parte 2, no PC (`analisar`): combina as previsões de seis formas e mede o erro de nível mês a mês, PR-AUC, AUC e os alertas
por perfil. As formas "móveis" só usam, em cada origem, previsões fora da amostra cujo resultado já era conhecido.

Regra de escolha, fixada antes de rodar: vence o menor erro absoluto médio de nível (Brasil, por mês do alvo, nas origens de
avaliação), desde que a PR-AUC e o F1 do nível crítico não caiam mais de 0,01 em relação ao empilhamento fixo (o de produção).
O período avaliado coincide com o teste, que já foi visto: a pergunta aqui é qual procedimento de produção funciona, e a
confirmação limpa continua vindo dos meses novos do Banco Central.

Uso no PC: python src/backtest.py [pasta com os .npz]   (padrão: modelos/backtest)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
BASE = RAIZ / "dados" / "processados" / "base_modelagem.parquet"
HIPERPARAMETROS = RAIZ / "modelos" / "hiperparametros_ensemble.json"
MODELOS = ("xgboost", "catboost", "random_forest", "lightgbm")
# as duas primeiras origens só geram histórico para as formas móveis; a avaliação começa em mar/2025
ORIGENS = ("2024-03-31", "2024-06-30", "2024-09-30", "2024-12-31", "2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31")
INICIO_AVALIACAO = "2025-03-31"
HORIZONTE = 6
MESES_POR_ORIGEM = 3
MESES_AJUSTE = 6  # meses mais recentes com resultado conhecido usados pelas formas móveis de calibração
TOLERANCIA = 0.01


def fim_do_mes(data, meses: int = 0) -> pd.Timestamp:
    return pd.Timestamp(data) + pd.DateOffset(months=meses) + pd.offsets.MonthEnd(0)


def recortes(datas: pd.Series, origem) -> tuple[np.ndarray, np.ndarray]:
    """Máscaras do treino (até origem − 6 meses) e da previsão (origem até origem + 2 meses) de uma origem."""
    datas = pd.to_datetime(datas)
    corte = fim_do_mes(origem, -HORIZONTE)
    fim = fim_do_mes(origem, MESES_POR_ORIGEM - 1)
    return (datas <= corte).to_numpy(), ((datas >= fim_do_mes(origem)) & (datas <= fim)).to_numpy()


def treino_da_origem(df: pd.DataFrame, mascara: np.ndarray) -> pd.DataFrame:
    """Recorte do treino com o teto do peso no percentil 99 calculado só nele, sem olhar o futuro."""
    d = df[mascara].reset_index(drop=True).copy()
    d["peso"] = d["peso"].clip(upper=d["peso"].quantile(0.99))
    return d


# ------------------------------------------------------------------ parte 1: Colab
def rodar(caminho_base=BASE, pasta=RAIZ / "modelos" / "backtest", modelos=MODELOS, origens=ORIGENS, gpu: bool = False,
          hiperparametros=HIPERPARAMETROS) -> pd.DataFrame:
    """Treina e prevê cada origem e modelo; salva `<origem>_<modelo>.npz` com as linhas da base e as previsões."""
    import otimizacao as o
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    params = json.loads(Path(hiperparametros).read_text(encoding="utf-8"))["params"]
    df = pd.read_parquet(caminho_base)
    tempos = []
    for origem in origens:
        m_treino, m_prev = recortes(df["data_base"], origem)
        if not m_prev.any():
            print(f"{origem}: sem perfis para prever, pulando", flush=True)
            continue
        treino, prever, linhas = None, df[m_prev].reset_index(drop=True), np.flatnonzero(m_prev).astype(np.int32)
        for modelo in modelos:
            arquivo = pasta / f"{origem[:7]}_{modelo}.npz"
            if arquivo.exists():
                continue
            if treino is None:
                treino = treino_da_origem(df, m_treino)
            t0 = time.time()
            comp = o.treinar(modelo, o.sugerir(modelo, o._TrialFixo(params[modelo])), treino, gpu)
            p = np.asarray(comp(prever), dtype=np.float32)
            segundos = time.time() - t0
            temporario = pasta / f"{origem[:7]}_{modelo}.tmp.npz"
            np.savez_compressed(temporario, linha=linhas, p=p, segundos=np.float32(segundos),
                                corte_treino=str(fim_do_mes(origem, -HORIZONTE).date()), linhas_treino=np.int64(len(treino)))
            temporario.replace(arquivo)
            tempos.append({"origem": origem[:7], "modelo": modelo, "segundos": round(segundos), "linhas_treino": len(treino)})
            print(f"{origem[:7]} {modelo}: {segundos:.0f}s, treino {len(treino):,} linhas, previsão {len(prever):,}", flush=True)
            del comp
    return pd.DataFrame(tempos)


# ------------------------------------------------------------------ parte 2: análise no PC
def sigmoide(x):
    return 1 / (1 + np.exp(-x))


def carregar_previsoes(pasta, n_linhas: int, modelos=MODELOS, origens=ORIGENS) -> pd.DataFrame:
    """Uma linha por perfil-mês previsto: linha da base, origem e a previsão de cada modelo."""
    partes = []
    for origem in origens:
        cols = {}
        for modelo in modelos:
            with np.load(Path(pasta) / f"{origem[:7]}_{modelo}.npz") as z:
                linha, p = z["linha"], z["p"].astype(float)
            assert linha.max() < n_linhas, f"{origem} {modelo}: linha fora da base; a base mudou depois do backtest?"
            if "linha" in cols:
                assert np.array_equal(cols["linha"], linha), f"{origem}: linhas diferentes entre os modelos"
            cols["linha"], cols[modelo] = linha, p
        partes.append(pd.DataFrame(cols).assign(origem=pd.Timestamp(origem)))
    return pd.concat(partes, ignore_index=True)


def ajuste_de_nivel(logito, w, alvo_medio) -> float:
    """Deslocamento d na log-chance tal que a média ponderada de sigmoide(logito + d) seja `alvo_medio`."""
    from scipy.optimize import brentq
    f = lambda d: np.average(sigmoide(logito + d), weights=w) - alvo_medio  # noqa: E731
    return float(brentq(f, -10, 10))


def combinar(dados: pd.DataFrame, coef_fixo: dict, intercepto_fixo: float, modelos=MODELOS) -> pd.DataFrame:
    """Seis formas de combinar; as móveis, em cada origem, só usam linhas com resultado já conhecido nela."""
    from sklearn.linear_model import LogisticRegression

    from modelo import Calibrador, logit
    from treino import escolher_calibrador, expandir

    lg = np.column_stack([logit(dados[m].to_numpy()) for m in modelos])
    fixo = sigmoide(intercepto_fixo + lg @ np.array([coef_fixo[m] for m in modelos]))
    media_lg = lg.mean(axis=1)
    saida = {"empilhado_fixo": fixo, "media": sigmoide(media_lg)}
    for k in ("empilhado_fixo_calibrado", "empilhado_movel", "media_nivel_movel", "media_calibrada_movel"):
        saida[k] = np.full(len(dados), np.nan)

    datas = dados["data_base"]
    for origem in dados["origem"].unique():
        alvo = (dados["origem"] == origem).to_numpy()
        conhecido = (datas <= fim_do_mes(origem, -HORIZONTE)).to_numpy() & (dados["origem"] < origem).to_numpy()
        recente = conhecido & (datas > fim_do_mes(origem, -HORIZONTE - MESES_AJUSTE)).to_numpy()
        if recente.sum() == 0:
            continue
        y, w, w_bruto = dados["alvo_inad"].to_numpy(), dados["peso_teto"].to_numpy(), dados["peso"].to_numpy()
        a, pw = expandir(y[conhecido], w[conhecido])
        ok = pw > 0
        x = lg[conhecido]
        pilha = LogisticRegression(C=1.0, max_iter=2000).fit(np.vstack([x, x])[ok], a[ok], sample_weight=pw[ok])
        saida["empilhado_movel"][alvo] = pilha.predict_proba(lg[alvo])[:, 1]
        real_recente = np.average(y[recente], weights=w_bruto[recente])
        d = ajuste_de_nivel(media_lg[recente], w_bruto[recente], real_recente)
        saida["media_nivel_movel"][alvo] = sigmoide(media_lg[alvo] + d)
        for nome, p in (("empilhado_fixo_calibrado", fixo), ("media_calibrada_movel", saida["media"])):
            metodo, _ = escolher_calibrador(p[recente], y[recente], w[recente], datas[recente])
            cal = Calibrador(metodo).fit(p[recente], y[recente], w[recente])
            saida[nome][alvo] = cal.transform(p[alvo])
    return pd.DataFrame(saida, index=dados.index)


def nivel_por_mes(dados: pd.DataFrame, previsto: np.ndarray) -> pd.DataFrame:
    """Real e previsto no Brasil por mês do alvo, ponderados pela carteira sem teto (bate com a taxa oficial)."""
    d = pd.DataFrame({"mes_alvo": (dados["data_base"] + pd.offsets.MonthEnd(HORIZONTE)).dt.strftime("%Y-%m").to_numpy(),
                      "w": dados["peso"].to_numpy(), "real": dados["alvo_inad"].to_numpy(), "prev": previsto})
    g = d.groupby("mes_alvo").apply(lambda x: pd.Series({"real_pct": np.average(x["real"], weights=x["w"]) * 100,
                                                         "previsto_pct": np.average(x["prev"], weights=x["w"]) * 100}),
                                    include_groups=False)
    g["erro_pp"] = g["previsto_pct"] - g["real_pct"]
    return g


def avaliar(dados: pd.DataFrame, previsoes: pd.DataFrame) -> dict:
    """Métricas de cada forma nas origens de avaliação."""
    import otimizacao as o
    import segmentos as s
    from treino import metricas

    aval = (dados["origem"] >= pd.Timestamp(INICIO_AVALIACAO)).to_numpy()
    d = dados[aval]
    y, w = d["alvo_inad"].to_numpy(), d["peso_teto"].to_numpy()
    resultado = {}
    for nome in previsoes.columns:
        p = previsoes[nome].to_numpy()[aval]
        if np.isnan(p).any():
            continue
        g = nivel_por_mes(d, p)
        df = d[["data_base"] + s.CHAVES_SEGMENTO].assign(peso=w, real=y, previsto=p, atual=d["perfil_inad_atual_pct"] / 100)
        alertas = {f"{int(lin['limite'] * 100)}%": lin["modelo"] for lin in s.avaliar(df)
                   if lin["nivel"] == "perfil" and lin["pesa"] == "reais" and lin["limite"] in (0.05, 0.10)}
        resultado[nome] = {
            "erro_nivel_abs_pp": float(g["erro_pp"].abs().mean()), "vies_nivel_pp": float(g["erro_pp"].mean()),
            "maior_erro_pp": float(g["erro_pp"].iloc[g["erro_pp"].abs().argmax()]),
            "erro_p10_pp": float(g["erro_pp"].quantile(0.1)), "erro_p90_pp": float(g["erro_pp"].quantile(0.9)),
            "pr_auc": o.pr_auc(y, w, p), "auc": metricas(y, w, p)["auc"], "alertas": alertas,
            "por_mes": g.round(4).reset_index().to_dict("records")}
    return resultado


def escolher(resultado: dict, referencia: str = "empilhado_fixo") -> str:
    ref = resultado[referencia]
    elegiveis = [n for n, r in resultado.items()
                 if r["pr_auc"] >= ref["pr_auc"] - TOLERANCIA and r["alertas"]["10%"]["f1"] >= ref["alertas"]["10%"]["f1"] - TOLERANCIA]
    return min(elegiveis, key=lambda n: resultado[n]["erro_nivel_abs_pp"])


def analisar(pasta=RAIZ / "modelos" / "backtest", caminho_base=BASE) -> dict:
    import joblib
    t0 = time.time()
    pasta = Path(pasta)
    base = pd.read_parquet(caminho_base)
    base["peso_teto"] = base["peso"].clip(upper=base["peso"].quantile(0.99))  # mesmo teto das métricas do projeto
    prev = carregar_previsoes(pasta, len(base))
    dados = base.iloc[prev["linha"].to_numpy()].reset_index(drop=True)
    for m in MODELOS:
        dados[m] = prev[m].to_numpy()
    dados["origem"] = prev["origem"].to_numpy()
    del base, prev
    fixo = joblib.load(RAIZ / "modelos" / "ensemble" / "final" / "ensemble.joblib")  # arquivo gerado por este projeto
    combinadas = combinar(dados, fixo["coeficientes"], fixo["intercepto"])
    resultado = avaliar(dados, combinadas)
    vencedor = escolher(resultado)
    tempos = {}
    for arquivo in sorted(pasta.glob("*.npz")):
        with np.load(arquivo) as z:
            tempos[arquivo.stem] = float(z["segundos"]) if "segundos" in z else None
    saida = {"gerado_em": pd.Timestamp.now().isoformat(timespec="seconds"), "origens": list(ORIGENS),
             "inicio_avaliacao": INICIO_AVALIACAO, "regra": __doc__.split("Regra de escolha, fixada antes de rodar: ")[1].split("\n\n")[0],
             "vencedor": vencedor, "resultados": resultado, "tempos_colab_s": tempos}
    (pasta / "resultados.json").write_text(json.dumps(saida, ensure_ascii=False, indent=1, default=float), encoding="utf-8")

    print(f"análise em {time.time() - t0:.0f}s; avaliação nas origens a partir de {INICIO_AVALIACAO[:7]}")
    print(f"{'forma':26s} {'erro abs':>8s} {'viés':>7s} {'maior':>7s} {'PR-AUC':>7s} {'AUC':>6s} {'F1 10%':>7s} {'F1 5%':>6s}")
    for n, r in sorted(resultado.items(), key=lambda kv: kv[1]["erro_nivel_abs_pp"]):
        print(f"{n:26s} {r['erro_nivel_abs_pp']:8.2f} {r['vies_nivel_pp']:+7.2f} {r['maior_erro_pp']:+7.2f} {r['pr_auc']:7.4f} "
              f"{r['auc']:6.3f} {r['alertas']['10%']['f1']:7.3f} {r['alertas']['5%']['f1']:6.3f}")
    print("vencedor pela regra:", vencedor)
    return saida


if __name__ == "__main__":
    sys.path.insert(0, str(RAIZ / "src"))
    analisar(Path(sys.argv[1]) if len(sys.argv) > 1 else RAIZ / "modelos" / "backtest")
