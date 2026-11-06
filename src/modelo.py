"""
Peças do modelo compartilhadas entre treino e dashboard.
Ficam num módulo próprio para que o joblib consiga carregar o modelo salvo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

CATEGORICAS = ["uf", "regiao", "segmento", "cliente", "ocupacao_atividade", "porte",
               "submodalidade", "origem", "indexador"]
OPERACAO = ["prazo_remanescente_anos", "valor_medio_operacao_mil"]
MACRO = ["selic", "ipca_12m", "dolar_var12m"]
SAFRA = ["safra_anomalia_pct", "safra_perda_area_pct", "safra_var_producao_pct"]
REGIONAL = ["uf_inad_atual_pct", "uf_atraso_15_90_pct", "uf_inad_var_6m_pp"]
NUMERICAS_TODAS = OPERACAO + MACRO + SAFRA + REGIONAL
PERFIL = ["perfil_inad_atual_pct", "perfil_atraso_15_90_pct", "perfil_inad_var_6m_pp"]
PERFIL_AMPLIADO = ["perfil_inad_var_1m_pp", "perfil_inad_var_3m_pp", "perfil_inad_var_12m_pp",
                   "perfil_problematico_pct", "perfil_vence_90d_pct", "perfil_cresc_6m_pct", "perfil_cresc_12m_pct",
                   "pares_inad_atual_pct", "uf_finalidade_inad_pct", "mes_alvo"]
PRECOS = ["preco_cesta_var_6m_pct", "preco_cesta_var_12m_pct", "fert_var_6m_pct", "fert_var_12m_pct",
          "boi_var_12m_pct", "relacao_troca_var_12m_pct"]
CUSTOS = ["conab_margem_pct", "conab_custo_var_anual_pct"]


def log_valor(X):
    X = np.asarray(X, dtype=float).copy()
    X[:, 1] = np.log1p(np.clip(X[:, 1], 0, None))  # 2ª numérica = valor médio da operação
    return X


def preprocessador(numericas: list[str], escalar: bool) -> ColumnTransformer:
    passos = [("imp", SimpleImputer(strategy="median")),
              ("log", FunctionTransformer(log_valor, feature_names_out="one-to-one"))]
    if escalar:
        passos.append(("esc", StandardScaler()))
    return ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=200,
                               sparse_output=False, dtype=np.float32), CATEGORICAS),
         ("num", Pipeline(passos), numericas)],
        verbose_feature_names_out=False,
    )


def mapa_colunas(pre: ColumnTransformer, numericas: list[str]) -> list[str]:
    """Variável original de cada coluna transformada (para somar SHAP dos dummies)."""
    enc: OneHotEncoder = pre.named_transformers_["cat"]
    origem = []
    for i, (nome, cats) in enumerate(zip(CATEGORICAS, enc.categories_)):
        n = len(cats)
        if enc.infrequent_categories_ is not None and enc.infrequent_categories_[i] is not None:
            n = n - len(enc.infrequent_categories_[i]) + 1
        origem += [nome] * n
    origem += list(numericas)
    assert len(origem) == len(pre.get_feature_names_out())
    return origem


def offset_uf(df: pd.DataFrame) -> np.ndarray:
    """Log-chance da inadimplência rural atual da UF: patamar de partida do modelo relativo."""
    p = np.clip(df["uf_inad_atual_pct"].to_numpy(dtype=float) / 100, 1e-3, 0.5)
    return np.log(p / (1 - p))


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class Calibrador:
    """Ajusta a probabilidade prevista ao risco observado. y é a fração inadimplente; w, a exposição."""

    METODOS = ("nenhum", "platt", "isotonica")

    def __init__(self, metodo: str):
        if metodo not in self.METODOS:
            raise ValueError(f"método de calibração desconhecido: {metodo}")
        self.metodo, self.modelo = metodo, None

    def fit(self, p, y, w) -> "Calibrador":
        p, y, w = (np.asarray(v, dtype=float) for v in (p, y, w))
        if self.metodo == "platt":
            x = logit(p).reshape(-1, 1)
            alvo = np.r_[np.ones_like(y), np.zeros_like(y)]
            peso = np.r_[w * y, w * (1 - y)]
            ok = peso > 0
            self.modelo = LogisticRegression(C=1e6, max_iter=1000).fit(
                np.vstack([x, x])[ok], alvo[ok], sample_weight=peso[ok])
        elif self.metodo == "isotonica":
            ok = w > 0
            self.modelo = IsotonicRegression(y_min=1e-5, y_max=1 - 1e-5, out_of_bounds="clip").fit(
                p[ok], y[ok], sample_weight=w[ok])
        return self

    def transform(self, p) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        if self.metodo == "platt":
            return self.modelo.predict_proba(logit(p).reshape(-1, 1))[:, 1]
        if self.metodo == "isotonica":
            return self.modelo.predict(p)
        return p


class ModeloRisco:
    """Interface única para os modelos: prever() (calibrado, se houver calibrador) e explicar()."""

    def __init__(self, tipo: str, pre, mod, numericas: list[str]):
        self.tipo, self.pre, self.mod = tipo, pre, mod
        self.numericas = list(numericas)
        self.features = CATEGORICAS + self.numericas
        self.origem = mapa_colunas(pre, self.numericas)
        self.explainer = None
        self.offset_medio = 0.0
        self.calibrador: Calibrador | None = None
        self.limiar: float | None = None  # probabilidade a partir da qual o perfil é sinalizado
        self.fatia_pct: float | None = None  # fatia da carteira atual que esse limiar sinaliza

    # ----------------------------------------------------------- previsão
    def prever(self, df: pd.DataFrame) -> np.ndarray:
        p = self.prever_bruto(df)
        return p if self.calibrador is None else self.calibrador.transform(p)

    def prever_bruto(self, df: pd.DataFrame) -> np.ndarray:
        X = self.pre.transform(df[self.features])
        if self.tipo == "logistica":
            return self.mod.predict_proba(X)[:, 1]
        if self.tipo == "xgb_offset":
            mg = self.mod.predict(X, output_margin=True, base_margin=offset_uf(df))
            return 1 / (1 + np.exp(-mg))
        return np.clip(self.mod.predict(X), 0, 1)

    @property
    def escala_log_chance(self) -> bool:
        return self.tipo in ("logistica", "xgb", "xgb_offset")

    # ----------------------------------------------------------- SHAP
    def preparar_explicador(self, amostra: pd.DataFrame):
        import shap
        X = self.pre.transform(amostra[self.features])
        if self.tipo == "logistica":
            self.explainer = shap.LinearExplainer(self.mod, X)
        else:
            self.explainer = shap.TreeExplainer(self.mod)
        if self.tipo == "xgb_offset":
            self.offset_medio = float(offset_uf(amostra).mean())
        # valor de partida da explicação: valor_base + soma dos SHAP = previsão (log-chance ou prob.).
        # Calculado empiricamente porque, com offset, o base_score do XGBoost não entra na previsão
        # mas entra no expected_value do TreeExplainer.
        p = self.prever_bruto(amostra)
        alvo = logit(p) if self.escala_log_chance else p
        resid = alvo - self._contribuicoes(amostra).sum(axis=1).to_numpy()
        self.valor_base = float(np.median(resid))
        self.erro_aditividade = float(np.max(np.abs(resid - self.valor_base)))

    def explicar(self, df: pd.DataFrame) -> pd.DataFrame:
        """Contribuição de cada variável original (log-chance, ou probabilidade no Random Forest).

        A coluna "calibracao" guarda o efeito do calibrador, para que valor_base + soma = previsão final.
        """
        out = self._contribuicoes(df)
        if self.calibrador is not None and self.calibrador.metodo != "nenhum":
            bruto = self.prever_bruto(df)
            cal = self.calibrador.transform(bruto)
            out["calibracao"] = logit(cal) - logit(bruto) if self.escala_log_chance else cal - bruto
        return out

    def _contribuicoes(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self.pre.transform(df[self.features])
        sv = np.atleast_2d(self.explainer.shap_values(X))
        out = pd.DataFrame(sv, columns=self.origem).T.groupby(level=0, sort=False).sum().T
        if self.tipo == "xgb_offset":
            # o patamar regional entra como offset fora das árvores; sua contribuição é o desvio da média
            out["uf_inad_atual_pct"] = out.get("uf_inad_atual_pct", 0) + (offset_uf(df) - self.offset_medio)
        return out


# ------------------------------------------------------------------ ensemble treinado no Colab
class Componente:
    """Um modelo do ensemble com o seu pré-processamento.

    Salvo no formato próprio de cada biblioteca, e não em pickle do modelo, para abrir no Windows mesmo treinado no
    Linux do Colab. O pré-processamento e o Random Forest são objetos do scikit-learn, salvos com joblib.
    """

    ARQUIVOS = {"xgboost": "modelo.ubj", "lightgbm": "modelo.txt", "catboost": "modelo.cbm", "random_forest": "modelo.joblib"}

    def __init__(self, tipo: str, pre, modelo, numericas: list[str]):
        self.tipo, self.pre, self.modelo = tipo, pre, modelo
        self.numericas = list(numericas)
        self.features = CATEGORICAS + self.numericas
        self._explicador = None

    def _matriz(self, df):
        return self.pre.transform(df[self.features]).astype(np.float32)

    def prever(self, df) -> np.ndarray:
        X = self._matriz(df)
        if self.tipo == "xgboost":
            p = self.modelo.inplace_predict(X)
        elif self.tipo == "lightgbm":
            p = self.modelo.predict(X)
        elif self.tipo == "catboost":
            p = self.modelo.predict_proba(X)[:, 1]
        else:
            p = self.modelo.predict(X)
        return np.clip(np.asarray(p, dtype=float), 0, 1)

    __call__ = prever

    def contribuicoes(self, df) -> pd.DataFrame:
        """SHAP em log-chance por variável original.

        No Random Forest, usa o SHAP aproximado do shap (método de Saabas), porque o exato leva cerca de 1 s por linha nas
        árvores profundas, e converte a escala de probabilidade pela derivada do logit no ponto previsto. As duas coisas
        tornam a parte dele aproximada."""
        import shap
        if self._explicador is None:
            self._explicador = shap.TreeExplainer(self.modelo)
        X = self._matriz(df)
        sv = np.asarray(self._explicador.shap_values(X, approximate=True) if self.tipo == "random_forest"
                        else self._explicador.shap_values(X))
        if sv.ndim == 3:  # classificadores que devolvem uma matriz por classe
            sv = sv[..., -1] if sv.shape[-1] == 2 else sv[-1]
        sv = np.atleast_2d(sv)
        if self.tipo == "random_forest":
            p = np.clip(self.prever(df), 1e-4, 1 - 1e-4)
            sv = sv / (p * (1 - p))[:, None]
        origem = mapa_colunas(self.pre, self.numericas)
        return pd.DataFrame(sv, columns=origem).T.groupby(level=0, sort=False).sum().T

    def salvar(self, pasta) -> None:
        import joblib
        from pathlib import Path
        pasta = Path(pasta)
        pasta.mkdir(parents=True, exist_ok=True)
        arquivo = pasta / self.ARQUIVOS[self.tipo]
        if self.tipo in ("xgboost", "lightgbm", "catboost"):
            self.modelo.save_model(str(arquivo))
        else:
            joblib.dump(self.modelo, arquivo, compress=3)
        joblib.dump({"tipo": self.tipo, "pre": self.pre, "numericas": self.numericas}, pasta / "pre.joblib", compress=3)

    @classmethod
    def carregar(cls, pasta) -> "Componente":
        import joblib
        from pathlib import Path
        pasta = Path(pasta)
        info = joblib.load(pasta / "pre.joblib")  # arquivo gerado por este projeto
        tipo, arquivo = info["tipo"], pasta / cls.ARQUIVOS[info["tipo"]]
        if tipo == "xgboost":
            import xgboost as xgb
            modelo = xgb.Booster()
            modelo.load_model(str(arquivo))
            modelo.set_param({"device": "cpu"})
        elif tipo == "lightgbm":
            import lightgbm as lgb
            modelo = lgb.Booster(model_file=str(arquivo))
        elif tipo == "catboost":
            from catboost import CatBoostClassifier
            modelo = CatBoostClassifier()
            modelo.load_model(str(arquivo))
        else:
            modelo = joblib.load(arquivo)
        return cls(tipo, info["pre"], modelo, info["numericas"])


class ModeloEnsemble:
    """Empilhamento: regressão logística sobre as log-chances dos componentes, seguida do calibrador.

    Tem a mesma interface de ModeloRisco usada pelo dashboard: prever(), prever_bruto(), explicar(), features, limiar,
    fatia_pct, valor_base e escala_log_chance.
    """

    tipo = "ensemble"
    escala_log_chance = True

    def __init__(self, componentes: dict, coeficientes: dict, intercepto: float):
        self.componentes = componentes
        self.coeficientes = dict(coeficientes)
        self.intercepto = float(intercepto)
        self.features = list(next(iter(componentes.values())).features)
        self.calibrador: Calibrador | None = None
        self.limiar: float | None = None
        self.fatia_pct: float | None = None
        self.valor_base = 0.0
        self.erro_aditividade = None

    def prever_bruto(self, df) -> np.ndarray:
        margem = self.intercepto + sum(c * logit(self.componentes[n].prever(df)) for n, c in self.coeficientes.items())
        return 1 / (1 + np.exp(-margem))

    def prever(self, df) -> np.ndarray:
        p = self.prever_bruto(df)
        return p if self.calibrador is None else self.calibrador.transform(p)

    def _contribuicoes(self, df) -> pd.DataFrame:
        partes = [c * self.componentes[n].contribuicoes(df) for n, c in self.coeficientes.items()]
        return pd.concat(partes, axis=1).T.groupby(level=0, sort=False).sum().T

    def explicar(self, df) -> pd.DataFrame:
        out = self._contribuicoes(df)
        if self.calibrador is not None and self.calibrador.metodo != "nenhum":
            bruto = self.prever_bruto(df)
            out["calibracao"] = logit(self.calibrador.transform(bruto)) - logit(bruto)
        return out

    def preparar_explicador(self, amostra) -> None:
        """Valor de partida da explicação: valor_base + soma das contribuições = log-chance antes da calibração."""
        resid = logit(self.prever_bruto(amostra)) - self._contribuicoes(amostra).sum(axis=1).to_numpy()
        self.valor_base = float(np.median(resid))
        self.erro_aditividade = float(np.quantile(np.abs(resid - self.valor_base), 0.95))

    def salvar(self, pasta, componentes: bool = True) -> None:
        """Com componentes=False, grava só o resumo do ensemble, quando os componentes já estão salvos em `pasta`."""
        import joblib
        from pathlib import Path
        pasta = Path(pasta)
        if componentes:
            for nome, comp in self.componentes.items():
                comp.salvar(pasta / nome)
        joblib.dump({"coeficientes": self.coeficientes, "intercepto": self.intercepto, "calibrador": self.calibrador,
                     "limiar": self.limiar, "fatia_pct": self.fatia_pct, "valor_base": self.valor_base,
                     "erro_aditividade": self.erro_aditividade}, pasta / "ensemble.joblib", compress=3)

    @classmethod
    def carregar(cls, pasta) -> "ModeloEnsemble":
        import joblib
        from pathlib import Path
        pasta = Path(pasta)
        info = joblib.load(pasta / "ensemble.joblib")  # arquivo gerado por este projeto
        m = cls({n: Componente.carregar(pasta / n) for n in info["coeficientes"]}, info["coeficientes"], info["intercepto"])
        for chave in ("calibrador", "limiar", "fatia_pct", "valor_base", "erro_aditividade"):
            setattr(m, chave, info[chave])
        return m


def carregar_modelo(raiz):
    """Modelo ativo do dashboard: o ensemble, se os metadados apontarem para ele, ou o modelo salvo em joblib."""
    import json
    import joblib
    from pathlib import Path
    raiz = Path(raiz)
    meta = json.loads((raiz / "modelos" / "metadados.json").read_text(encoding="utf-8"))
    if meta.get("formato") == "ensemble":
        pasta = raiz / "modelos" / "ensemble" / "final"
        if not (pasta / "ensemble.joblib").exists():
            raise FileNotFoundError(f"os modelos do ensemble não estão em {pasta}. Eles não vão para o repositório (~130 MB): rode "
                                    "src/montar_ensemble.py com o producao_final.zip do Colab, ou volte ao modelo anterior com "
                                    "src/trocar_criterio.py f1_auc")
        return ModeloEnsemble.carregar(pasta)
    return joblib.load(raiz / "modelos" / "modelo_final.joblib")  # arquivo gerado por este projeto
