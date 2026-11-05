"""
Busca com Optuna, ensembles e comparação final, segundo a orientação do professor. Feito para rodar no Colab.

  busca      : maximiza a PR-AUC na validação 2 (perfis jul–dez/2024, treino até dez/2023), com 50 tentativas por modelo
  ajustes    : calibração e fatia da carteira vêm da validação 1 (perfis jan–jun/2024, treino até jun/2023)
  escolha    : melhor posição média em PR-AUC, precisão, recall e F1 no teste; em empate, a maior PR-AUC

Escolher no teste segue o pedido do professor, mas deixa o número otimista: confirme com os próximos meses do Banco Central.
Os intervalos de busca e as adaptações estão em docs/hiperparametros-literatura.md.
O progresso fica em `pasta`, que no Colab deve estar no Drive; rodar de novo continua de onde parou.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from modelo import (CATEGORICAS, CUSTOS, NUMERICAS_TODAS, PERFIL, PERFIL_AMPLIADO, Calibrador, Componente, ModeloEnsemble,
                    logit, preprocessador)
from treino import (FIM_TREINO, FIM_TREINO_VAL, FIM_TREINO_VAL2, FIM_VAL, FIM_VAL2, INI_TESTE, INI_VAL, INI_VAL2, SEED,
                    ajustar_limiar, escolher_calibrador, expandir, fatia_do_limiar, limiar_da_fatia, metricas,
                    metricas_extras, metricas_sinal, sinalizar_por_fatia)

MODELOS_ENSEMBLE = ("xgboost", "catboost", "random_forest", "lightgbm")

NUMERICAS = NUMERICAS_TODAS + PERFIL + PERFIL_AMPLIADO + CUSTOS
FEATURES = CATEGORICAS + NUMERICAS
ESTIMADORES = 1000
PARADA = 100
AMOSTRA_RF = 200_000
AMOSTRA_REALMLP = 300_000
MODELOS_BUSCA = ("xgboost", "lightgbm", "catboost", "random_forest", "realmlp")
METRICAS_ESCOLHA = ("pr_auc", "precisao", "recall", "f1")


# ------------------------------------------------------------------ espaços de busca
def sugerir(modelo: str, trial) -> dict:
    """Intervalos da literatura, com as adaptações registradas em docs/hiperparametros-literatura.md."""
    if modelo == "xgboost":
        return {"max_depth": trial.suggest_int("max_depth", 1, 11),
                "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 100, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1e-2, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 4.0, log=True),
                "gamma": trial.suggest_float("gamma", 1e-8, 7.0, log=True)}
    if modelo == "lightgbm":
        p = {"num_leaves": trial.suggest_int("num_leaves", 2, int(np.exp(7)), log=True),
             "learning_rate": trial.suggest_float("learning_rate", float(np.exp(-7)), 1.0, log=True),
             "subsample": trial.suggest_float("subsample", 0.5, 1.0), "subsample_freq": 1,
             "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
             "min_child_samples": trial.suggest_int("min_child_samples", 1, int(np.exp(6)), log=True),
             "min_child_weight": trial.suggest_float("min_child_weight", float(np.exp(-16)), float(np.exp(5)), log=True)}
        for nome, alias in (("reg_alpha", "usa_l1"), ("reg_lambda", "usa_l2")):
            usa = trial.suggest_categorical(alias, [True, False])
            p[nome] = trial.suggest_float(nome, float(np.exp(-16)), float(np.exp(2)), log=True) if usa else 0.0
        return p
    if modelo == "catboost":
        # one_hot_max_size do artigo não se aplica: as categorias já chegam codificadas pelo pré-processamento
        return {"learning_rate": trial.suggest_float("learning_rate", float(np.exp(-5)), 1.0, log=True),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
                "random_strength": trial.suggest_int("random_strength", 1, 20),
                "leaf_estimation_iterations": trial.suggest_int("leaf_estimation_iterations", 1, 20)}
    if modelo == "random_forest":
        return {"max_depth": trial.suggest_categorical("max_depth", [None, 10, 15, 20, 25]),
                "n_estimators": trial.suggest_int("n_estimators", 50, 300, log=True),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]),
                "min_samples_split": trial.suggest_categorical("min_samples_split", [2, 3]),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 50, log=True),
                "bootstrap": trial.suggest_categorical("bootstrap", [True, False])}
    if modelo == "realmlp":
        camadas = {"256x3": [256, 256, 256], "64x5": [64] * 5, "512": [512]}
        return {"num_emb_type": trial.suggest_categorical("num_emb_type", ["none", "pbld", "pl", "plr"]),
                "add_front_scale": trial.suggest_categorical("add_front_scale", [True, False]),
                "lr": trial.suggest_float("lr", 2e-2, 0.3, log=True),
                "p_drop": trial.suggest_categorical("p_drop", [0.0, 0.15, 0.3]),
                "act": trial.suggest_categorical("act", ["relu", "selu", "mish"]),
                "hidden_sizes": camadas[trial.suggest_categorical("camadas", list(camadas))],
                "wd": trial.suggest_categorical("wd", [0.0, 0.02]),
                "plr_sigma": trial.suggest_float("plr_sigma", 0.05, 0.5, log=True), "use_ls": False}
    raise ValueError(f"modelo desconhecido: {modelo}")


# ------------------------------------------------------------------ métricas e escolha
def pr_auc(y_frac, w, p) -> float:
    y, peso = expandir(y_frac, w)
    ok = peso > 0
    return float(average_precision_score(y[ok], np.r_[p, p][ok], sample_weight=peso[ok]))


def escolher_no_teste(resultados: dict, metricas_escolha=METRICAS_ESCOLHA) -> str:
    """Melhor posição média nas métricas pedidas pelo professor; em empate, a maior PR-AUC."""
    nomes = list(resultados)
    posicoes = {n: 0.0 for n in nomes}
    for metrica in metricas_escolha:
        for i, n in enumerate(sorted(nomes, key=lambda x: -resultados[x][metrica])):
            posicoes[n] += i
    return min(nomes, key=lambda n: (posicoes[n] / len(metricas_escolha), -resultados[n]["pr_auc"]))


# ------------------------------------------------------------------ treino de cada família
def amostra_por_exposicao(d: pd.DataFrame, n: int, seed: int = SEED) -> tuple[pd.DataFrame, np.ndarray]:
    """Sorteia linhas proporcionalmente à exposição e o rótulo pela fração inadimplente.

    Serve para o RealMLP, cuja documentação não mostra sample_weight nem alvo fracionário no fit.
    Em média, preserva a exposição e a inadimplência da carteira.
    """
    rng = np.random.default_rng(seed)
    peso = d["peso"].to_numpy(dtype=float)
    idx = rng.choice(len(d), size=n, replace=True, p=peso / peso.sum())
    return d.iloc[idx].reset_index(drop=True), rng.binomial(1, d["alvo_inad"].to_numpy()[idx]).astype(int)


def _matriz(pre, d):
    return pre.transform(d[FEATURES]).astype(np.float32)


def _separar(d: pd.DataFrame):
    """Os últimos 6 meses do próprio treino decidem a parada antecipada."""
    corte = d["data_base"].max() - pd.DateOffset(months=6)
    marca = d["data_base"] <= corte
    return d[marca], d[~marca]


def treinar(modelo: str, params: dict, d: pd.DataFrame, gpu: bool = False):
    """Treina e devolve um objeto chamável que prevê a probabilidade de um DataFrame (Componente, nas árvores)."""
    pre = preprocessador(NUMERICAS, escalar=modelo == "realmlp").fit(d[FEATURES])
    a, b = _separar(d)
    if modelo == "xgboost":
        import xgboost as xgb
        comum = dict(objective="reg:logistic", tree_method="hist", device="cuda" if gpu else "cpu", random_state=SEED, n_jobs=-1, **params)
        m = xgb.XGBRegressor(n_estimators=ESTIMADORES, early_stopping_rounds=PARADA, eval_metric="logloss", **comum)
        m.fit(_matriz(pre, a), a["alvo_inad"], sample_weight=a["peso"], eval_set=[(_matriz(pre, b), b["alvo_inad"])],
              sample_weight_eval_set=[b["peso"]], verbose=False)
        final = xgb.XGBRegressor(n_estimators=int(m.best_iteration) + 1, **comum).fit(_matriz(pre, d), d["alvo_inad"], sample_weight=d["peso"])
        return Componente("xgboost", pre, final.get_booster(), NUMERICAS)
    if modelo == "lightgbm":
        import lightgbm as lgb
        comum = dict(objective="cross_entropy", random_state=SEED, n_jobs=-1, verbose=-1, **params)
        m = lgb.LGBMRegressor(n_estimators=ESTIMADORES, **comum)
        m.fit(_matriz(pre, a), a["alvo_inad"], sample_weight=a["peso"], eval_set=[(_matriz(pre, b), b["alvo_inad"])],
              eval_sample_weight=[b["peso"]], eval_metric="cross_entropy", callbacks=[lgb.early_stopping(PARADA, verbose=False)])
        final = lgb.LGBMRegressor(n_estimators=max(1, int(m.best_iteration_)), **comum).fit(_matriz(pre, d), d["alvo_inad"], sample_weight=d["peso"])
        return Componente("lightgbm", pre, final.booster_, NUMERICAS)
    if modelo == "catboost":
        from catboost import CatBoostClassifier, Pool
        comum = dict(loss_function="CrossEntropy", boosting_type="Plain", bootstrap_type="Bayesian", depth=6, random_seed=SEED, verbose=0,
                     allow_writing_files=False, task_type="GPU" if gpu else "CPU", **params)
        m = CatBoostClassifier(iterations=ESTIMADORES, od_type="Iter", od_wait=PARADA, **comum)
        m.fit(Pool(_matriz(pre, a), a["alvo_inad"], weight=a["peso"]), eval_set=Pool(_matriz(pre, b), b["alvo_inad"], weight=b["peso"]))
        final = CatBoostClassifier(iterations=max(1, int(m.get_best_iteration()) + 1), **comum)
        final.fit(Pool(_matriz(pre, d), d["alvo_inad"], weight=d["peso"]))
        return Componente("catboost", pre, final, NUMERICAS)
    if modelo == "random_forest":
        s = d.sample(min(AMOSTRA_RF, len(d)), random_state=SEED)
        final = RandomForestRegressor(n_jobs=-1, random_state=SEED, **params).fit(_matriz(pre, s), s["alvo_inad"], sample_weight=s["peso"])
        return Componente("random_forest", pre, final, NUMERICAS)
    if modelo == "realmlp":
        from pytabkit import RealMLP_TD_Classifier
        s, y = amostra_por_exposicao(d, min(AMOSTRA_REALMLP, 2 * len(d)))
        final = RealMLP_TD_Classifier(device="cuda:0" if gpu else "cpu", random_state=SEED, val_metric_name="cross_entropy", **params)
        final.fit(_matriz(pre, s), y)
        return lambda x: final.predict_proba(_matriz(pre, x))[:, 1]
    raise ValueError(f"modelo desconhecido: {modelo}")


# ------------------------------------------------------------------ dados e janelas
def carregar_janelas(caminho) -> dict:
    df = pd.read_parquet(caminho)
    df["peso"] = df["peso"].clip(upper=df["peso"].quantile(0.99))
    d = df["data_base"]
    return {"treino_val1": df[d <= FIM_TREINO_VAL].reset_index(drop=True), "val1": df[(d >= INI_VAL) & (d <= FIM_VAL)].reset_index(drop=True),
            "treino_val2": df[d <= FIM_TREINO_VAL2].reset_index(drop=True), "val2": df[(d >= INI_VAL2) & (d <= FIM_VAL2)].reset_index(drop=True),
            "treino_teste": df[d <= FIM_TREINO].reset_index(drop=True), "teste": df[d >= INI_TESTE].reset_index(drop=True)}


def _partes(d):
    return d["alvo_inad"].to_numpy(), d["peso"].to_numpy()


# ------------------------------------------------------------------ busca
def buscar(modelo: str, janelas: dict, pasta: Path, n_tentativas: int = 50, gpu: bool = False) -> dict:
    """Continua de onde parou: o banco do Optuna é copiado para `pasta` a cada tentativa."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    pasta.mkdir(parents=True, exist_ok=True)
    salvo, local = pasta / f"optuna_{modelo}.db", Path(f"optuna_{modelo}.db")
    if salvo.exists() and not local.exists():
        shutil.copy2(salvo, local)
    estudo = optuna.create_study(study_name=modelo, direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED),
                                 storage=f"sqlite:///{local}", load_if_exists=True)
    y, w = _partes(janelas["val2"])

    def objetivo(trial):
        t0 = time.time()
        p = treinar(modelo, sugerir(modelo, trial), janelas["treino_val2"], gpu)(janelas["val2"])
        valor = pr_auc(y, w, p)
        print(f"{modelo} [{trial.number + 1}] PR-AUC {valor:.4f} ({time.time() - t0:.0f}s)", flush=True)
        return valor

    def guardar(_estudo, _trial):
        temporario = salvo.with_suffix(".tmp")
        shutil.copy2(local, temporario)
        temporario.replace(salvo)

    feitas = sum(t.state.is_finished() and t.value is not None for t in estudo.trials)
    if feitas < n_tentativas:
        # uma tentativa com erro (como falta de memória na GPU) não derruba as demais
        estudo.optimize(objetivo, n_trials=n_tentativas - feitas, callbacks=[guardar], catch=(RuntimeError, MemoryError), gc_after_trial=True)
    return {"params": estudo.best_trial.params, "pr_auc_val2": estudo.best_value, "tentativas": len(estudo.trials),
            "tentativas_com_erro": sum(t.state.name == "FAIL" for t in estudo.trials)}


class _TrialFixo:
    """Reconstrói os parâmetros de `sugerir` a partir do que o Optuna guardou, inclusive os derivados."""

    def __init__(self, params: dict):
        self.params = params

    def _pegar(self, nome, *_args, **_kwargs):
        return self.params.get(nome, 0.0)

    suggest_int = suggest_float = suggest_categorical = _pegar


def gerar_previsoes(modelo: str, params_optuna: dict, janelas: dict, pasta: Path, gpu: bool = False) -> dict:
    """Treina nas três janelas e guarda as previsões; não refaz o que já está salvo."""
    arquivo = pasta / f"previsoes_{modelo}.npz"
    if arquivo.exists():
        with np.load(arquivo) as z:
            return {k: z[k] for k in ("val1", "val2", "teste")}
    params = sugerir(modelo, _TrialFixo(params_optuna))
    prev = {"val1": treinar(modelo, params, janelas["treino_val1"], gpu)(janelas["val1"]),
            "val2": treinar(modelo, params, janelas["treino_val2"], gpu)(janelas["val2"]),
            "teste": treinar(modelo, params, janelas["treino_teste"], gpu)(janelas["teste"])}
    temporario = pasta / f"previsoes_{modelo}.tmp.npz"
    np.savez_compressed(temporario, **prev)
    temporario.replace(arquivo)
    return prev


# ------------------------------------------------------------------ ensembles
def _posicoes(p) -> np.ndarray:
    return (pd.Series(p).rank(method="average").to_numpy() - 0.5) / len(p)


def selecao_gulosa(prev: dict, y, w, passos: int = 20) -> dict:
    """Caruana et al. (2004): adiciona modelos com repetição, pela PR-AUC na validação 2. Devolve os pesos."""
    nomes = list(prev)
    soma, contagem, melhor = np.zeros_like(next(iter(prev.values())), dtype=float), {n: 0 for n in nomes}, -np.inf
    historico = []
    for passo in range(1, passos + 1):
        notas = {n: pr_auc(y, w, (soma + prev[n]) / passo) for n in nomes}
        escolhido = max(notas, key=notas.get)
        soma += prev[escolhido]
        contagem[escolhido] += 1
        historico.append(dict(contagem))
        if notas[escolhido] > melhor:
            melhor, melhores = notas[escolhido], dict(contagem)
    total = sum(melhores.values())
    return {n: c / total for n, c in melhores.items() if c}


def combinar_ensembles(previsoes: dict, janelas: dict) -> dict:
    """Média simples, média de posições, seleção gulosa e empilhamento. Só a validação 2 define pesos."""
    y2, w2 = _partes(janelas["val2"])
    nomes = list(previsoes)
    pv2 = {n: previsoes[n]["val2"] for n in nomes}
    pesos = selecao_gulosa(pv2, y2, w2)

    def matriz(jan):
        return np.column_stack([logit(previsoes[n][jan]) for n in nomes])

    alvo, peso = expandir(y2, w2)
    ok = peso > 0
    x2 = matriz("val2")
    pilha = LogisticRegression(C=1.0, max_iter=2000).fit(np.vstack([x2, x2])[ok], alvo[ok], sample_weight=peso[ok])
    saida = {}
    for jan in ("val1", "val2", "teste"):
        saida.setdefault("ensemble_media", {})[jan] = np.mean([previsoes[n][jan] for n in nomes], axis=0)
        saida.setdefault("ensemble_posicoes", {})[jan] = np.mean([_posicoes(previsoes[n][jan]) for n in nomes], axis=0)
        saida.setdefault("ensemble_guloso", {})[jan] = np.sum([q * previsoes[n][jan] for n, q in pesos.items()], axis=0)
        saida.setdefault("ensemble_empilhado", {})[jan] = pilha.predict_proba(matriz(jan))[:, 1]
    return {"previsoes": saida, "pesos_guloso": pesos, "coeficientes_empilhado": dict(zip(nomes, pilha.coef_[0].round(4).tolist()))}


# ------------------------------------------------------------------ comparação
def avaliar(prev: dict, janelas: dict) -> dict:
    """Calibração e fatia vêm só da validação 1; o teste apenas mede."""
    y1, w1 = _partes(janelas["val1"])
    y, w = _partes(janelas["teste"])
    metodo, _ = escolher_calibrador(prev["val1"], y1, w1, janelas["val1"]["data_base"])
    cal = Calibrador(metodo).fit(prev["val1"], y1, w1)
    pc1 = cal.transform(prev["val1"])
    fatia = fatia_do_limiar(pc1, w1, ajustar_limiar(y1, w1, pc1))
    p = cal.transform(prev["teste"])
    sinal = sinalizar_por_fatia(prev["teste"], w, janelas["teste"]["data_base"], fatia)
    y2, w2 = _partes(janelas["val2"])
    return {**metricas(y, w, p), **metricas_sinal(y, w, sinal), **metricas_extras(y, w, p), "pr_auc": metricas_extras(y, w, p)["auprc"],
            "pr_auc_val2": pr_auc(y2, w2, prev["val2"]), "calibracao": metodo, "fatia_pct": fatia * 100,
            "limiar_pct": limiar_da_fatia(p, w, fatia) * 100}


def comparar(previsoes: dict, janelas: dict, pasta: Path, extras: dict | None = None) -> dict:
    """Avalia modelos e ensembles, aplica a regra do professor e salva resultados em JSON e CSV."""
    resultados = {n: avaliar(p, janelas) for n, p in previsoes.items()}
    vencedor = escolher_no_teste(resultados)
    saida = {"vencedor": vencedor, "regra": "melhor posição média em PR-AUC, precisão, recall e F1 no teste; em empate, a maior PR-AUC",
             "resultados": resultados, **(extras or {})}
    (pasta / "resultados_optuna.json").write_text(json.dumps(saida, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    pd.DataFrame(resultados).T.to_csv(pasta / "resultados_optuna.csv")
    return saida


def rodar_tudo(caminho_base, pasta, modelos=MODELOS_BUSCA, n_tentativas: int = 50, gpu: bool = False) -> dict:
    pasta = Path(pasta)
    pasta.mkdir(parents=True, exist_ok=True)
    janelas = carregar_janelas(caminho_base)
    buscas = {m: buscar(m, janelas, pasta, n_tentativas, gpu) for m in modelos}
    previsoes = {m: gerar_previsoes(m, buscas[m]["params"], janelas, pasta, gpu) for m in modelos}
    extras = {"buscas": buscas}
    if len(previsoes) > 1:
        comb = combinar_ensembles(previsoes, janelas)
        previsoes = {**previsoes, **comb["previsoes"]}
        extras |= {"pesos_guloso": comb["pesos_guloso"], "coeficientes_empilhado": comb["coeficientes_empilhado"]}
    return comparar(previsoes, janelas, pasta, extras)


# ------------------------------------------------------------------ produção do ensemble empilhado
def coeficientes_empilhamento(previsoes: dict, janelas: dict, nomes=MODELOS_ENSEMBLE) -> tuple[dict, float]:
    """Os mesmos pesos do ensemble avaliado no teste: regressão logística ajustada na validação 2."""
    y2, w2 = _partes(janelas["val2"])
    x2 = np.column_stack([logit(previsoes[n]["val2"]) for n in nomes])
    alvo, peso = expandir(y2, w2)
    ok = peso > 0
    pilha = LogisticRegression(C=1.0, max_iter=2000).fit(np.vstack([x2, x2])[ok], alvo[ok], sample_weight=peso[ok])
    return dict(zip(nomes, pilha.coef_[0].astype(float).tolist())), float(pilha.intercept_[0])


def treinar_producao(caminho_base, caminho_atual, pasta, gpu: bool = False, nomes=MODELOS_ENSEMBLE) -> dict:
    """Treina o ensemble final e salva em `pasta`/producao, no mesmo procedimento do treino local:

    calibrador e fatia vêm dos 6 meses mais recentes com alvo conhecido, previstos por modelos treinados 12 meses antes;
    o threshold é o que gera essa fatia nos perfis do último mês. Cada componente já pronto é salvo e não é refeito.
    """
    pasta = Path(pasta)
    destino = pasta / "producao"
    janelas = carregar_janelas(caminho_base)
    buscas = json.loads((pasta / "resultados_optuna.json").read_text(encoding="utf-8"))["buscas"]
    previsoes = {n: gerar_previsoes(n, buscas[n]["params"], janelas, pasta, gpu) for n in nomes}
    coef, intercepto = coeficientes_empilhamento(previsoes, janelas, nomes)
    del janelas

    df = pd.read_parquet(caminho_base)
    df["peso"] = df["peso"].clip(upper=df["peso"].quantile(0.99))
    data_max = df["data_base"].max()
    interno = df[df["data_base"] <= data_max - pd.DateOffset(months=12)].reset_index(drop=True)
    janela = df[df["data_base"] > data_max - pd.DateOffset(months=6)].reset_index(drop=True)

    def componente(nome, dados, rotulo):
        caminho = destino / rotulo / nome
        if (caminho / "pre.joblib").exists():
            return Componente.carregar(caminho)
        t0 = time.time()
        comp = treinar(nome, sugerir(nome, _TrialFixo(buscas[nome]["params"])), dados, gpu)
        comp.salvar(caminho)
        tamanho = sum(a.stat().st_size for a in caminho.iterdir()) / 1e6
        print(f"{rotulo}/{nome}: {time.time() - t0:.0f}s, {tamanho:.0f} MB", flush=True)
        return comp

    internos = ModeloEnsemble({n: componente(n, interno, "interno") for n in nomes}, coef, intercepto)
    y, w = _partes(janela)
    p_jan = internos.prever_bruto(janela)
    metodo, notas = escolher_calibrador(p_jan, y, w, janela["data_base"])
    cal = Calibrador(metodo).fit(p_jan, y, w)
    pc = cal.transform(p_jan)
    fatia = fatia_do_limiar(pc, w, ajustar_limiar(y, w, pc))
    del internos, interno

    final = ModeloEnsemble({n: componente(n, df, "final") for n in nomes}, coef, intercepto)
    atual = pd.read_parquet(caminho_atual)
    final.calibrador, final.fatia_pct = cal, fatia * 100
    final.limiar = limiar_da_fatia(final.prever(atual), atual["peso"].to_numpy(), fatia)
    # o valor de partida da explicação é calculado no PC, onde o SHAP roda; aqui só salva os modelos
    joblib_info = {"calibracao": metodo, "brier_por_metodo": notas, "fatia_pct": fatia * 100, "limiar_pct": final.limiar * 100,
                   "janela_de": str(janela["data_base"].min().date()), "janela_ate": str(janela["data_base"].max().date()),
                   "coeficientes": coef, "intercepto": intercepto}
    final.salvar(destino / "final", componentes=False)  # cada componente já foi salvo ao terminar o treino dele
    (destino / "producao.json").write_text(json.dumps(joblib_info, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"produção: calibração {metodo}; fatia {fatia * 100:.1f}%; limiar {final.limiar * 100:.2f}%", flush=True)
    return joblib_info
