"""
Dashboard de risco de crédito rural — PoC
Rodar:  streamlit run app.py
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

RAIZ = Path(__file__).resolve().parent
sys.path.insert(0, str(RAIZ / "src"))  # necessário para carregar o pipeline salvo

st.set_page_config(page_title="Risco de Crédito Agro", page_icon="🌾", layout="wide")

# ------------------------------------------------------------------ rótulos
ROTULOS = {
    "uf": "UF do tomador",
    "regiao": "Região",
    "segmento": "Tipo de instituição",
    "cliente": "Tipo de tomador",
    "ocupacao_atividade": "Ocupação / atividade",
    "porte": "Porte / faixa de renda",
    "submodalidade": "Finalidade do crédito",
    "origem": "Origem dos recursos",
    "indexador": "Indexador",
    "prazo_remanescente_anos": "Prazo remanescente",
    "valor_medio_operacao_mil": "Valor da operação",
    "selic": "Selic",
    "ipca_12m": "IPCA 12 meses",
    "dolar_var12m": "Variação do dólar 12m",
    "safra_anomalia_pct": "Produtividade da safra de grãos na UF vs média 5 anos",
    "safra_perda_area_pct": "Área de grãos perdida na UF",
    "safra_var_producao_pct": "Variação da produção de grãos na UF",
    "uf_inad_atual_pct": "Inadimplência rural atual na UF",
    "uf_atraso_15_90_pct": "Atraso de 15 a 90 dias na carteira rural da UF",
    "uf_inad_var_6m_pp": "Variação da inadimplência rural da UF em 6 meses",
    "perfil_inad_atual_pct": "Inadimplência atual deste perfil",
    "perfil_atraso_15_90_pct": "Atraso de 15 a 90 dias deste perfil",
    "perfil_inad_var_6m_pp": "Variação da inadimplência deste perfil em 6 meses",
    "perfil_inad_var_1m_pp": "Variação da inadimplência deste perfil em 1 mês",
    "perfil_inad_var_3m_pp": "Variação da inadimplência deste perfil em 3 meses",
    "perfil_inad_var_12m_pp": "Variação da inadimplência deste perfil em 12 meses",
    "perfil_problematico_pct": "Ativo problemático deste perfil",
    "perfil_vence_90d_pct": "Parcelas deste perfil vencendo em até 90 dias",
    "perfil_cresc_6m_pct": "Crescimento da carteira deste perfil em 6 meses",
    "perfil_cresc_12m_pct": "Crescimento da carteira deste perfil em 12 meses",
    "pares_inad_atual_pct": "Inadimplência de perfis parecidos no país",
    "uf_finalidade_inad_pct": "Inadimplência da mesma finalidade na UF",
    "mes_alvo": "Mês do vencimento da previsão",
    "preco_cesta_var_6m_pct": "Preço da cesta agrícola da UF em 6 meses",
    "preco_cesta_var_12m_pct": "Preço da cesta agrícola da UF em 12 meses",
    "fert_var_6m_pct": "Preço dos fertilizantes em 6 meses",
    "fert_var_12m_pct": "Preço dos fertilizantes em 12 meses",
    "boi_var_12m_pct": "Preço do boi em 12 meses",
    "relacao_troca_var_12m_pct": "Relação cesta agrícola/fertilizantes em 12 meses",
    "conab_margem_pct": "Margem estimada de soja e milho na UF",
    "conab_custo_var_anual_pct": "Variação anual do custo de soja e milho na UF",
    "calibracao": "Ajuste de calibração",
}
def br(v, casas=1, sinal=False):
    """Número no formato brasileiro (vírgula decimal, ponto de milhar)."""
    t = f"{v:+,.{casas}f}" if sinal else f"{v:,.{casas}f}"
    return t.replace(",", "_").replace(".", ",").replace("_", ".")


UNIDADE = {
    "prazo_remanescente_anos": lambda v: f"{br(v)} anos",
    "valor_medio_operacao_mil": lambda v: f"R$ {br(v, 0)} mil",
    "selic": lambda v: f"{br(v, 2)}% a.a.",
    "ipca_12m": lambda v: f"{br(v)}%",
    "dolar_var12m": lambda v: f"{br(v, sinal=True)}%",
    "safra_anomalia_pct": lambda v: f"{br(v, sinal=True)}%",
    "safra_perda_area_pct": lambda v: f"{br(v)}%",
    "safra_var_producao_pct": lambda v: f"{br(v, sinal=True)}%",
    "uf_inad_atual_pct": lambda v: f"{br(v, 2)}%",
    "uf_atraso_15_90_pct": lambda v: f"{br(v, 2)}%",
    "uf_inad_var_6m_pp": lambda v: f"{br(v, 2, sinal=True)} p.p.",
    "perfil_inad_atual_pct": lambda v: f"{br(v, 2)}%",
    "perfil_atraso_15_90_pct": lambda v: f"{br(v, 2)}%",
    "perfil_inad_var_6m_pp": lambda v: f"{br(v, 2, sinal=True)} p.p.",
    "perfil_inad_var_1m_pp": lambda v: f"{br(v, 2, sinal=True)} p.p.",
    "perfil_inad_var_3m_pp": lambda v: f"{br(v, 2, sinal=True)} p.p.",
    "perfil_inad_var_12m_pp": lambda v: f"{br(v, 2, sinal=True)} p.p.",
    "perfil_problematico_pct": lambda v: f"{br(v, 2)}%",
    "perfil_vence_90d_pct": lambda v: f"{br(v, 1)}%",
    "perfil_cresc_6m_pct": lambda v: f"{br(v, 1, sinal=True)}%",
    "perfil_cresc_12m_pct": lambda v: f"{br(v, 1, sinal=True)}%",
    "pares_inad_atual_pct": lambda v: f"{br(v, 2)}%",
    "uf_finalidade_inad_pct": lambda v: f"{br(v, 2)}%",
    "mes_alvo": lambda v: f"{int(v):02d}",
    **{c: (lambda v: f"{br(v, 1, sinal=True)}%") for c in [
        "preco_cesta_var_6m_pct", "preco_cesta_var_12m_pct", "fert_var_6m_pct", "fert_var_12m_pct",
        "boi_var_12m_pct", "relacao_troca_var_12m_pct", "conab_margem_pct", "conab_custo_var_anual_pct"]},
}
REGIAO = {
    **{u: "Norte" for u in ["RO", "AC", "AM", "RR", "PA", "AP", "TO"]},
    **{u: "Nordeste" for u in ["MA", "PI", "CE", "RN", "PB", "PE", "AL", "SE", "BA"]},
    **{u: "Sudeste" for u in ["MG", "ES", "RJ", "SP"]},
    **{u: "Sul" for u in ["PR", "SC", "RS"]},
    **{u: "Centro-Oeste" for u in ["MS", "MT", "GO", "DF"]},
}
ORDEM_PORTE_PF = ["Sem rendimento", "Até 1 salário mínimo", "Mais de 1 a 2 salários mínimos",
                  "Mais de 2 a 3 salários mínimos", "Mais de 3 a 5 salários mínimos",
                  "Mais de 5 a 10 salários mínimos", "Mais de 10 a 20 salários mínimos",
                  "Acima de 20 salários mínimos", "Indisponível"]
ORDEM_PORTE_PJ = ["Micro", "Pequeno", "Médio", "Grande", "Indisponível"]
FAIXAS = [(0.01, "A", "Muito baixo", "#2e7d32"), (0.03, "B", "Baixo", "#7cb342"),
          (0.06, "C", "Moderado", "#f9a825"), (0.12, "D", "Alto", "#ef6c00"),
          (1.01, "E", "Muito alto", "#c62828")]


def fmt_int(n):
    return f"{int(n):,}".replace(",", ".")


def fmt_pct(p, casas=2):
    return f"{p * 100:.{casas}f}%".replace(".", ",")


def faixa(pd_):
    for lim, letra, nome, cor in FAIXAS:
        if pd_ < lim:
            return letra, nome, cor
    return FAIXAS[-1][1:]


def score(pd_):
    """Escala de pontos tipo scorecard: 600 pontos = chance 20:1 de adimplência; +40 pontos dobra a chance."""
    pd_ = float(np.clip(pd_, 1e-5, 1 - 1e-5))
    odds = (1 - pd_) / pd_
    return int(np.clip(600 + 40 / np.log(2) * np.log(odds / 20), 0, 1000))


# ------------------------------------------------------------------ carga
@st.cache_resource
def carregar():
    from modelo import carregar_modelo
    modelo = carregar_modelo(RAIZ)
    meta = json.loads((RAIZ / "modelos" / "metadados.json").read_text(encoding="utf-8"))
    ctx = json.loads((RAIZ / "dados" / "processados" / "contexto_atual.json").read_text(encoding="utf-8"))
    perfis = pd.read_parquet(RAIZ / "dados" / "processados" / "perfil_atual.parquet")
    return modelo, meta, ctx, perfis


MODELO, meta, ctx, PERFIS = carregar()
FEATURES = MODELO.features
OPC = meta["opcoes"]
USA_MACRO = any(c in FEATURES for c in ["selic", "ipca_12m", "dolar_var12m"])
COLS_PERFIL = ["perfil_inad_atual_pct", "perfil_atraso_15_90_pct", "perfil_inad_var_6m_pp"]
COLS_EXTRAS = ["perfil_inad_var_1m_pp", "perfil_inad_var_3m_pp", "perfil_inad_var_12m_pp", "perfil_problematico_pct",
               "perfil_vence_90d_pct", "perfil_cresc_6m_pct", "perfil_cresc_12m_pct", "pares_inad_atual_pct",
               "uf_finalidade_inad_pct", "mes_alvo"]
COLS_PRECOS = ["preco_cesta_var_6m_pct", "preco_cesta_var_12m_pct", "fert_var_6m_pct", "fert_var_12m_pct",
               "boi_var_12m_pct", "relacao_troca_var_12m_pct", "conab_margem_pct", "conab_custo_var_anual_pct"]
NOME_CRITERIO = {"f1_auc": "F1 e AUC-ROC juntos", "f1": "F1", "beneficio": "benefício líquido médio"}
REGRA_PROFESSOR = "professor_pr_auc_precisao_recall_f1"
if meta.get("criterio_escolha") == REGRA_PROFESSOR:
    TEXTO_ESCOLHA = (
        "- **Ajustes:** XGBoost, CatBoost, Random Forest e LightGBM tiveram os hiperparâmetros buscados com o Optuna (50 tentativas, "
        "PR-AUC) e foram combinados por empilhamento, uma regressão logística sobre as previsões deles. O F1 define a fatia sinalizada.\n"
        f"- **Escolha:** no teste, pela melhor posição média em PR-AUC, precisão, recall e F1, por orientação do professor "
        f"({meta['modelo_escolhido']}). Isso deixa os números do teste otimistas.\n"
        "- **Explicação:** os fatores somam as explicações SHAP dos quatro modelos com os pesos do empilhamento. A parte do Random "
        "Forest é aproximada, então os valores servem como orientação.")
else:
    TEXTO_ESCOLHA = (
        "- **Ajustes:** o aprendizado usa a perda logística, os hiperparâmetros são buscados pelo Brier e o F1 define a fatia sinalizada.\n"
        f"- **Escolha:** o modelo de maior {NOME_CRITERIO.get(meta.get('criterio_escolha'), 'F1')} na segunda janela de validação, "
        f"usando os ajustes da primeira ({meta['modelo_escolhido']}). O teste não entra nessa conta.")
MES_ALVO = (pd.Timestamp(ctx["data_base"]) + pd.DateOffset(months=meta["horizonte_meses"])).month
CHAVES_PERFIL = ["uf", "segmento", "cliente", "ocupacao_atividade", "porte", "submodalidade", "origem", "indexador"]


def prever(df: pd.DataFrame) -> np.ndarray:
    return MODELO.prever(df)


def explicar(df: pd.DataFrame) -> pd.DataFrame:
    return MODELO.explicar(df)


def preencher_perfil(df: pd.DataFrame) -> pd.DataFrame:
    """Completa o histórico do perfil com o SCR mais recente; sem histórico, usa o patamar da UF.

    Colunas já informadas são mantidas. Precisa das colunas uf_inad_atual_pct e uf_atraso_15_90_pct.
    """
    achado = df[CHAVES_PERFIL].merge(PERFIS, on=CHAVES_PERFIL, how="left")
    saida = df.copy()
    saida["perfil_encontrado"] = achado["perfil_inad_atual_pct"].notna().to_numpy()
    reserva = {"perfil_inad_atual_pct": df["uf_inad_atual_pct"], "perfil_atraso_15_90_pct": df["uf_atraso_15_90_pct"]}
    for c in COLS_PERFIL + COLS_EXTRAS:
        if c in saida.columns:
            continue
        valores = pd.Series(achado[c].to_numpy(), index=df.index)
        saida[c] = valores.fillna(reserva[c]) if c in reserva else valores
    saida["mes_alvo"] = saida["mes_alvo"].fillna(MES_ALVO)
    return saida


def descrever(var, valor):
    if var == "calibracao":
        return ROTULOS[var]
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return f"{ROTULOS[var]}: sem dado"
    if var in UNIDADE:
        return f"{ROTULOS[var]}: {UNIDADE[var](valor)}"
    return f"{ROTULOS[var]}: {valor}"


def efeito_txt(contrib, curto=False):
    if MODELO.escala_log_chance:  # contribuições em log-chance
        mult = f"×{br(np.exp(contrib), 2)}"
        return mult if curto else f"chance de inadimplir {mult}"
    return f"{br(contrib * 100, 2, sinal=True)} p.p."


# ------------------------------------------------------------------ cabeçalho
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.6rem;}
      .kpi {border-radius: 10px; padding: 14px 18px; border: 1px solid rgba(128,128,128,.25);
            min-height: 148px; margin-bottom: 12px;}
      .kpi h1 {margin: 0; font-size: 2.4rem;}
      .kpi p {margin: 0; opacity: .75;}
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("🌾 Risco de Crédito Rural")
st.caption(
    f"Probabilidade de a operação estar inadimplente (atraso > 90 dias) daqui a {meta['horizonte_meses']} meses · "
    f"modelo **{meta['modelo_escolhido']}** treinado com dados oficiais do Banco Central (SCR.data) e IBGE · "
    f"dados até {ctx['data_base'][:7].replace('-', '/')}"
)

aba_avaliar, aba_lote, aba_segmentos, aba_modelo, aba_dados = st.tabs(
    ["Avaliar produtor", "Avaliação em lote", "Segmentos críticos", "Desempenho do modelo", "Dados e método"])

# ================================================================== AVALIAR
with aba_avaliar:
    col_form, col_res = st.columns([0.42, 0.58], gap="large")

    with col_form:
        st.subheader("Perfil do produtor")
        c1, c2 = st.columns(2)
        tipo = c1.radio("Tipo de tomador", ["PF", "PJ"], horizontal=True,
                        format_func=lambda x: "Pessoa física" if x == "PF" else "Pessoa jurídica")
        uf = c2.selectbox("UF", sorted(ctx["uf"]), index=sorted(ctx["uf"]).index("RS"))

        if tipo == "PF":
            ocup_ops = OPC["ocupacao_PF"]
            porte_ops = [p for p in ORDEM_PORTE_PF if p in OPC["porte_PF"]]
            ocup = st.selectbox("Natureza da ocupação", ocup_ops,
                                index=ocup_ops.index("Autônomo") if "Autônomo" in ocup_ops else 0,
                                help="Classificação do SCR. Produtores rurais PF costumam aparecer como Autônomo, Empresário ou Outros.")
            porte = st.selectbox("Renda mensal", porte_ops, index=porte_ops.index("Mais de 5 a 10 salários mínimos"))
        else:
            ocup_ops = OPC["ocupacao_PJ"]
            agro = "Agricultura, pecuária, produção florestal, pesca e aquicultura"
            ocup = st.selectbox("Atividade (seção CNAE)", ocup_ops,
                                index=ocup_ops.index(agro) if agro in ocup_ops else 0)
            porte_ops = [p for p in ORDEM_PORTE_PJ if p in OPC["porte_PJ"]]
            porte = st.selectbox("Porte da empresa", porte_ops, index=porte_ops.index("Pequeno"))

        st.subheader("Operação")
        c1, c2 = st.columns(2)
        finalidade = c1.selectbox("Finalidade", OPC["submodalidade"],
                                  index=OPC["submodalidade"].index("Custeio"))
        segmento = c2.selectbox("Instituição credora", OPC["segmento"],
                                index=OPC["segmento"].index("Cooperativa"))
        c1, c2 = st.columns(2)
        origem = c1.selectbox("Origem dos recursos", OPC["origem"],
                              index=OPC["origem"].index("Com destinação específica"),
                              help="Com destinação específica = crédito direcionado/controlado (ex.: Pronaf, Pronamp, LCA obrigatória, fundos constitucionais).")
        indexador = c2.selectbox("Indexador", OPC["indexador"],
                                 index=OPC["indexador"].index("Prefixado"),
                                 help="TCR/TRFC = taxas do crédito rural e dos fundos constitucionais.")
        c1, c2 = st.columns(2)
        valor = c1.number_input("Valor da operação (R$ mil)", 1.0, 100_000.0, 120.0, step=10.0)
        prazo = c2.number_input("Prazo remanescente (anos)", 0.1, 20.0, 1.0, step=0.5)

        cu = ctx["uf"][uf]
        with st.expander(f"Cenário econômico e de safra — preenchido com dados oficiais de {uf}", expanded=False):
            st.caption("Valores mais recentes disponíveis. Altere para simular estresse (ex.: quebra de safra, alta de juros).")
            if USA_MACRO:
                c1, c2, c3 = st.columns(3)
                selic = c1.number_input("Selic (% a.a.)", 0.0, 30.0, float(ctx["macro"]["selic"]), 0.25)
                ipca = c2.number_input("IPCA 12m (%)", -5.0, 30.0, round(float(ctx["macro"]["ipca_12m"]), 2), 0.1)
                dolar = c3.number_input("Dólar, var. 12m (%)", -50.0, 100.0, round(float(ctx["macro"]["dolar_var12m"]), 1), 1.0)
            else:
                selic, ipca, dolar = ctx["macro"]["selic"], ctx["macro"]["ipca_12m"], ctx["macro"]["dolar_var12m"]
            c1, c2, c3 = st.columns(3)
            saf_anom = c1.number_input("Produtividade grãos vs média 5a (%)", -90.0, 200.0,
                                       round(cu.get("safra_anomalia_pct") or 0.0, 1), 1.0)
            saf_perda = c2.number_input("Área de grãos perdida (%)", 0.0, 100.0,
                                        round(cu.get("safra_perda_area_pct") or 0.0, 2), 0.5)
            saf_var = c3.number_input("Var. produção grãos (%)", -100.0, 300.0,
                                      round(cu.get("safra_var_producao_pct") or 0.0, 1), 1.0)
            c1, c2, c3 = st.columns(3)
            uf_inad = c1.number_input("Inadimplência rural UF (%)", 0.0, 60.0, round(cu["uf_inad_atual_pct"], 2), 0.1)
            uf_atr = c2.number_input("Atraso 15–90d UF (%)", 0.0, 60.0, round(cu["uf_atraso_15_90_pct"], 2), 0.1)
            uf_var = c3.number_input("Var. inad. UF 6m (p.p.)", -30.0, 30.0, round(cu.get("uf_inad_var_6m_pp") or 0.0, 2), 0.1)
            ref_perfil = preencher_perfil(pd.DataFrame([{
                "uf": uf, "segmento": segmento, "cliente": tipo, "ocupacao_atividade": ocup, "porte": porte,
                "submodalidade": finalidade, "origem": origem, "indexador": indexador,
                "uf_inad_atual_pct": uf_inad, "uf_atraso_15_90_pct": uf_atr}])).iloc[0]
            var_ref = ref_perfil["perfil_inad_var_6m_pp"]
            c1, c2, c3 = st.columns(3)
            perfil_inad = c1.number_input("Inadimplência do perfil (%)", 0.0, 100.0,
                                          round(float(ref_perfil["perfil_inad_atual_pct"]), 2), 0.1)
            perfil_atr = c2.number_input("Atraso 15–90d do perfil (%)", 0.0, 100.0,
                                         round(float(ref_perfil["perfil_atraso_15_90_pct"]), 2), 0.1)
            perfil_var = c3.number_input("Var. inad. do perfil 6m (p.p.)", -100.0, 100.0,
                                         0.0 if pd.isna(var_ref) else round(float(var_ref), 2), 0.1)
            st.caption("Histórico deste perfil no SCR.data mais recente." if ref_perfil["perfil_encontrado"] else
                       "Este perfil não aparece no SCR.data mais recente: usamos o patamar da UF no lugar do histórico.")
            st.caption(f"Safra de referência: {ctx['ano_safra']} (IBGE PAM/LSPA). Carteira: SCR.data {ctx['data_base'][:7]}."
                       + ("" if USA_MACRO else " Selic, IPCA e câmbio foram testados e removidos do modelo final por "
                          "piorarem a previsão fora do tempo; o ciclo econômico entra pela inadimplência atual da UF."))

    entrada = pd.DataFrame([{
        "uf": uf, "regiao": REGIAO[uf], "segmento": segmento, "cliente": tipo,
        "ocupacao_atividade": ocup, "porte": porte, "submodalidade": finalidade,
        "origem": origem, "indexador": indexador,
        "prazo_remanescente_anos": prazo, "valor_medio_operacao_mil": valor,
        "selic": selic, "ipca_12m": ipca, "dolar_var12m": dolar,
        "safra_anomalia_pct": saf_anom, "safra_perda_area_pct": saf_perda, "safra_var_producao_pct": saf_var,
        "uf_inad_atual_pct": uf_inad, "uf_atraso_15_90_pct": uf_atr, "uf_inad_var_6m_pp": uf_var,
        "perfil_inad_atual_pct": perfil_inad, "perfil_atraso_15_90_pct": perfil_atr, "perfil_inad_var_6m_pp": perfil_var,
    }])

    for c in COLS_EXTRAS:
        entrada[c] = ref_perfil[c]
    for c in COLS_PRECOS:
        entrada[c] = cu.get(c)

    with col_res:
        p = float(prever(entrada)[0])
        letra, nome_faixa, cor = faixa(p)
        sc = score(p)
        ref_uf = uf_inad / 100

        k1, k2, k3 = st.columns(3)
        k1.markdown(f"<div class='kpi'><p>Probabilidade de inadimplência</p><h1 style='color:{cor}'>{fmt_pct(p)}</h1></div>",
                    unsafe_allow_html=True)
        k2.markdown(f"<div class='kpi'><p>Score (0–1000)</p><h1>{sc}</h1></div>", unsafe_allow_html=True)
        k3.markdown(f"<div class='kpi'><p>Faixa de risco</p><h1 style='color:{cor}'>{letra}</h1><p>{nome_faixa}</p></div>",
                    unsafe_allow_html=True)

        rel = p / ref_uf if ref_uf > 0 else np.nan
        if np.isfinite(rel):
            rel_txt = f"{rel:.1f}".replace(".", ",")
            st.markdown(f"Para comparação, a carteira rural de **{uf}** tem hoje **{fmt_pct(ref_uf)}** inadimplente. "
                        f"Este perfil está em **{rel_txt}×** esse nível.")
        if p >= MODELO.limiar:
            st.warning(f"**Sinalizado para análise:** este perfil está entre os {br(MODELO.fatia_pct)}% da carteira "
                       f"de maior risco hoje (a partir de {fmt_pct(MODELO.limiar)} de probabilidade).", icon="⚠️")
        else:
            st.success(f"**Não sinalizado:** este perfil está fora dos {br(MODELO.fatia_pct)}% da carteira "
                       f"de maior risco hoje (a partir de {fmt_pct(MODELO.limiar)} de probabilidade).", icon="✅")

        st.subheader("O que mais pesou nesta avaliação")
        contrib = explicar(entrada).iloc[0]
        if MODELO.escala_log_chance:
            p_base = 1 / (1 + np.exp(-MODELO.valor_base))
            st.markdown(f"A explicação parte da inadimplência média histórica da carteira rural "
                        f"(**{fmt_pct(p_base)}**). Cada fator multiplica a chance de inadimplir a partir dela, "
                        f"chegando aos **{fmt_pct(p)}** deste perfil.")
        contrib = contrib.reindex(contrib.abs().sort_values(ascending=True).index)
        top = contrib.tail(10)
        rotulos_y = [descrever(v, entrada.iloc[0].get(v)) for v in top.index]
        fig = go.Figure(go.Bar(
            x=top.values, y=rotulos_y, orientation="h",
            marker_color=["#c62828" if v > 0 else "#2e7d32" for v in top.values],
            text=[efeito_txt(v, curto=True) for v in top.values], textposition="outside", cliponaxis=False,
            hovertemplate="%{y}<br>%{text}<extra></extra>",
        ))
        unidade_eixo = "contribuição SHAP (log-chance)" if MODELO.escala_log_chance else "contribuição SHAP (prob.)"
        lim = float(np.abs(top.values).max()) * 1.3 + 1e-9
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=30), xaxis_title=unidade_eixo,
                          xaxis=dict(range=[min(top.min(), 0) * 1.3 - lim * 0.05, max(top.max(), 0) * 1.3 + lim * 0.05]),
                          yaxis=dict(tickfont=dict(size=12)), showlegend=False,
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        fig.add_vline(x=0, line_width=1, line_color="gray")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Vermelho aumenta o risco, verde reduz; ×2 dobra a chance de inadimplir, ×0,5 a reduz à metade. "
                   "Valores SHAP do modelo final, somando todos os níveis de cada variável categórica.")

        sobe = contrib[contrib > 0].sort_values(ascending=False).head(3)
        desce = contrib[contrib < 0].sort_values().head(3)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Principais fatores de risco**")
            for v, c in sobe.items():
                st.markdown(f"- {descrever(v, entrada.iloc[0].get(v))} — {efeito_txt(c)}")
        with c2:
            st.markdown("**Principais atenuantes**")
            for v, c in desce.items():
                st.markdown(f"- {descrever(v, entrada.iloc[0].get(v))} — {efeito_txt(c)}")

        st.info("Estimativa de risco do **perfil** do produtor no cenário informado, aprendida com a carteira rural "
                "brasileira agregada. Use como apoio à análise, não como decisão automática de crédito.", icon="ℹ️")

# ================================================================== LOTE
with aba_lote:
    st.subheader("Avaliar vários produtores de uma vez")
    st.write("Envie um CSV com uma linha por produtor. Colunas de cenário ausentes são preenchidas "
             "automaticamente com os dados oficiais mais recentes da UF.")
    perfil_cols = ["uf", "cliente", "ocupacao_atividade", "porte", "submodalidade", "segmento",
                   "origem", "indexador", "valor_medio_operacao_mil", "prazo_remanescente_anos"]
    modelo_csv = pd.DataFrame([
        {"id_produtor": "P001", "uf": "RS", "cliente": "PF", "ocupacao_atividade": "Autônomo",
         "porte": "Mais de 5 a 10 salários mínimos", "submodalidade": "Custeio", "segmento": "Cooperativa",
         "origem": "Com destinação específica", "indexador": "Prefixado",
         "valor_medio_operacao_mil": 150, "prazo_remanescente_anos": 1.0},
        {"id_produtor": "P002", "uf": "MT", "cliente": "PJ",
         "ocupacao_atividade": "Agricultura, pecuária, produção florestal, pesca e aquicultura",
         "porte": "Médio", "submodalidade": "Investimento", "segmento": "Banco",
         "origem": "Sem destinação específica", "indexador": "Pós-fixado",
         "valor_medio_operacao_mil": 2500, "prazo_remanescente_anos": 4.0},
    ])
    st.download_button("Baixar modelo de CSV", modelo_csv.to_csv(index=False).encode("utf-8"),
                       "modelo_produtores.csv", "text/csv")
    with st.expander("Valores aceitos em cada coluna"):
        for c in ["cliente", "segmento", "submodalidade", "origem", "indexador", "porte", "ocupacao_atividade"]:
            st.markdown(f"**{c}**: " + " · ".join(OPC[c]))

    arq = st.file_uploader("CSV de produtores", type="csv")
    if arq is not None:
        lote = pd.read_csv(arq)
        faltam = [c for c in perfil_cols if c not in lote.columns]
        if faltam:
            st.error("Colunas obrigatórias ausentes: " + ", ".join(faltam))
        else:
            lote["regiao"] = lote["uf"].map(REGIAO)
            for c, v in ctx["macro"].items():
                if c not in lote.columns:
                    lote[c] = v
            for c in ["safra_anomalia_pct", "safra_perda_area_pct", "safra_var_producao_pct",
                      "uf_inad_atual_pct", "uf_atraso_15_90_pct", "uf_inad_var_6m_pp"] + COLS_PRECOS:
                if c not in lote.columns:
                    lote[c] = lote["uf"].map(lambda u, c=c: (ctx["uf"].get(u) or {}).get(c))
            lote = preencher_perfil(lote)
            invalidos = {c: sorted(set(lote[c]) - set(OPC[c])) for c in
                         ["uf", "cliente", "segmento", "submodalidade", "origem", "indexador", "porte", "ocupacao_atividade"]
                         if c in OPC}
            invalidos = {k: v for k, v in invalidos.items() if v}
            if invalidos:
                st.warning("Valores não reconhecidos (serão tratados como categoria rara): " +
                           "; ".join(f"{k}: {v}" for k, v in invalidos.items()))
            probs = prever(lote)
            sv = explicar(lote)
            lote["prob_inadimplencia_pct"] = (probs * 100).round(2)
            lote["score"] = [score(x) for x in probs]
            lote["faixa"] = [faixa(x)[0] for x in probs]
            lote["sinalizado"] = ["sim" if x >= MODELO.limiar else "não" for x in probs]
            # fatores do próprio perfil/operação (o cenário regional é igual para todos da mesma UF)
            CENARIO = {"uf_inad_atual_pct", "uf_atraso_15_90_pct", "uf_inad_var_6m_pp", "selic", "ipca_12m",
                       "dolar_var12m", "safra_anomalia_pct", "safra_perda_area_pct", "safra_var_producao_pct",
                       *COLS_PRECOS}
            svp = sv[[c for c in sv.columns if c not in CENARIO | {"calibracao"}]]
            lote["principal_fator_risco"] = [descrever(svp.iloc[i].idxmax(), lote.iloc[i][svp.iloc[i].idxmax()])
                                             for i in range(len(lote))]
            lote["principal_atenuante"] = [descrever(svp.iloc[i].idxmin(), lote.iloc[i][svp.iloc[i].idxmin()])
                                           for i in range(len(lote))]
            lote["efeito_cenario_uf"] = [efeito_txt(sv.iloc[i][[c for c in sv.columns if c in CENARIO]].sum())
                                         for i in range(len(lote))]
            mostrar = [c for c in ["id_produtor"] if c in lote.columns] + perfil_cols[:2] + \
                      ["prob_inadimplencia_pct", "score", "faixa", "sinalizado", "principal_fator_risco", "principal_atenuante", "efeito_cenario_uf"]
            st.dataframe(lote[mostrar], use_container_width=True, hide_index=True)
            st.download_button("Baixar resultado", lote.to_csv(index=False).encode("utf-8"),
                               "resultado_risco.csv", "text/csv")

# ================================================================== SEGMENTOS
from segmentos import LIMITES, agregar  # noqa: E402


@st.cache_data(show_spinner="Calculando o risco dos perfis da carteira atual...")
def risco_carteira() -> pd.DataFrame:
    atual = pd.read_parquet(RAIZ / "dados" / "processados" / "base_atual.parquet")
    return atual.assign(previsto=MODELO.prever(atual), atual=atual["perfil_inad_atual_pct"] / 100)


with aba_segmentos:
    st.subheader("Segmentos críticos nos próximos 6 meses")
    st.write("Quais segmentos (UF × finalidade) e quais perfis devem passar de um limite de inadimplência daqui a 6 meses, "
             f"com os perfis da carteira de **{ctx['data_base'][:7].replace('-', '/')}**. Um segmento é crítico quando a "
             "inadimplência prevista, ponderada pela carteira, passa do limite.")
    limite = st.radio("Limite de inadimplência", LIMITES, index=1, horizontal=True, format_func=lambda x: fmt_pct(x, 0))
    carteira = risco_carteira()
    seg = agregar(carteira, mes=None).sort_values("previsto", ascending=False)
    seg_crit = seg[seg["previsto"] >= limite]
    perf_crit = carteira[carteira["previsto"] >= limite]
    k1, k2, k3 = st.columns(3)
    k1.metric("Segmentos críticos", f"{len(seg_crit)} de {len(seg)}")
    k2.metric("Carteira nesses segmentos", fmt_pct(seg_crit["peso"].sum() / seg["peso"].sum(), 1))
    k3.metric("Perfis críticos", f"{fmt_int(len(perf_crit))} de {fmt_int(len(carteira))}")
    previsto_carteira = np.average(carteira["previsto"], weights=carteira["peso"])
    atual_carteira = np.average(carteira["atual"], weights=carteira["peso"])
    if previsto_carteira > 1.2 * atual_carteira:
        st.info(f"O modelo prevê {fmt_pct(previsto_carteira, 1)} de inadimplência para a carteira daqui a 6 meses, contra "
                f"{fmt_pct(atual_carteira, 1)} hoje. Ele está assumindo que a piora recente continua; se o ciclo estabilizar, "
                "parte desses segmentos não chegará ao limite. Use a lista mais como ordem de prioridade do que como contagem exata.",
                icon="ℹ️")

    tabela = pd.DataFrame({
        "UF": seg["uf"], "Finalidade": seg["submodalidade"], "Carteira (R$ mi)": seg["peso"], "Perfis": seg["perfis"],
        "Inadimplência hoje (%)": seg["atual"] * 100, "Prevista em 6 meses (%)": seg["previsto"] * 100,
        "Crítico": np.where(seg["previsto"] >= limite, "sim", "não"),
    })
    st.markdown("**Segmentos, do maior para o menor risco previsto**")
    st.dataframe(tabela.style.format({"Carteira (R$ mi)": lambda v: br(v, 1), "Inadimplência hoje (%)": lambda v: br(v, 2),
                                      "Prevista em 6 meses (%)": lambda v: br(v, 2)}),
                 use_container_width=True, hide_index=True, height=380)

    st.markdown("**Perfis críticos com mais dinheiro em risco**")
    st.caption("Ordenados pelo valor previsto de inadimplência (taxa prevista × carteira). Ordenar só pela taxa levaria ao topo "
               "perfis muito pequenos que já estão quase todos inadimplentes.")
    cols_perfil = ["uf", "segmento", "cliente", "ocupacao_atividade", "porte", "submodalidade", "origem", "indexador"]
    top = perf_crit.assign(em_risco=perf_crit["previsto"] * perf_crit["peso"]).sort_values("em_risco", ascending=False).head(30)
    st.dataframe(top[cols_perfil].assign(**{
        "Carteira (R$ mi)": top["peso"].map(lambda v: br(v, 1)),
        "Inadimplência hoje (%)": top["atual"].map(lambda v: br(v * 100, 2)),
        "Prevista em 6 meses (%)": top["previsto"].map(lambda v: br(v * 100, 2)),
        "Valor em risco (R$ mi)": top["em_risco"].map(lambda v: br(v, 1))}),
        use_container_width=True, hide_index=True)
    st.download_button("Baixar todos os segmentos", tabela.to_csv(index=False).encode("utf-8"), "segmentos_criticos.csv", "text/csv")

    medicao = meta.get("segmentos_teste")
    if medicao:
        st.markdown(f"**Como essa previsão se saiu no teste (limite de {fmt_pct(limite, 0)})**")
        linhas = [r for r in medicao["linhas"] if abs(r["limite"] - limite) < 1e-9]
        st.dataframe(pd.DataFrame([{
            "Nível": "Perfil" if r["nivel"] == "perfil" else "Segmento UF × finalidade",
            "Pesa por": "reais de carteira" if r["pesa"] == "reais" else "contagem",
            "F1 do modelo": r["modelo"]["f1"], "Precisão": r["modelo"]["precisao"], "Recall": r["modelo"]["recall"],
            "F1 repetindo o nível atual": r["nivel_atual"]["f1"], "F1 sinalizando tudo": r["sinalizar_tudo"]["f1"],
        } for r in linhas]).style.format(lambda v: br(v, 3) if isinstance(v, float) else v), use_container_width=True, hide_index=True)
        perde = [r for r in linhas if r["modelo"]["f1"] < r["nivel_atual"]["f1"]]
        if perde:
            st.warning("Neste limite, em pelo menos uma visão a regra simples de repetir o nível atual tem F1 maior que o modelo. "
                       "Use essa visão com cautela.", icon="⚠️")
    st.caption("É outra pergunta, diferente da probabilidade do modelo principal: o F1 daqui não é comparável ao da aba de desempenho. "
               "Os limites foram escolhidos depois de ver o teste, e o modelo foi escolhido no teste, então essas medidas são otimistas. "
               f"O modelo prevê hoje {fmt_pct(np.average(carteira['previsto'], weights=carteira['peso']), 1)} de inadimplência para a "
               f"carteira, contra {fmt_pct(np.average(carteira['atual'], weights=carteira['peso']), 1)} agora; se esse nível estiver alto, "
               "a lista de críticos fica maior do que deveria.")

# ================================================================== MODELO
with aba_modelo:
    st.subheader("Comparação dos modelos — teste fora do tempo")
    if meta.get("criterio_escolha") == REGRA_PROFESSOR:
        st.write(f"Treino com perfis até **{meta['treino_ate'][:7]}**, teste com perfis de **{meta['teste_desde'][:7]}** "
                 f"em diante (alvos observados 6 meses depois). Os hiperparâmetros foram buscados com o Optuna pela PR-AUC "
                 f"na janela de **{meta['validacao2_de'][:7]}** a **{meta['validacao2_ate'][:7]}**; a calibração e a fatia sinalizada "
                 f"vêm da janela de **{meta['validacao_de'][:7]}** a **{meta['validacao_ate'][:7]}**. Por orientação do professor, "
                 "o modelo foi escolhido **no próprio teste**, pela melhor posição média em PR-AUC, precisão, recall e F1. "
                 "Por isso, os números do modelo escolhido são otimistas até serem confirmados com meses novos.")
    elif "validacao2_de" in meta:
        st.write(f"Treino com perfis até **{meta['treino_ate'][:7]}**, teste com perfis de **{meta['teste_desde'][:7]}** "
                 f"em diante (alvos observados 6 meses depois). A calibração e a fatia de carteira sinalizada vêm da validação "
                 f"com perfis de **{meta['validacao_de'][:7]}** a **{meta['validacao_ate'][:7]}**. O modelo é o de maior "
                 f"**{NOME_CRITERIO.get(meta.get('criterio_escolha'), 'F1')}** ao aplicar esses ajustes na janela de "
                 f"**{meta['validacao2_de'][:7]}** a **{meta['validacao2_ate'][:7]}**.")
    else:
        st.write(f"Treino com perfis até **{meta['treino_ate'][:7]}**, teste com perfis de **{meta['teste_desde'][:7]}** "
                 f"em diante (alvos observados 6 meses depois). O modelo, a calibração e o threshold foram escolhidos na "
                 f"validação (perfis de **{meta['validacao_de'][:7]}** a **{meta['validacao_ate'][:7]}**), sem olhar o teste.")
    comp = pd.DataFrame(meta["comparacao"]).T
    tab = pd.DataFrame({
        "AUC": comp["auc"], "Gini": comp["gini"], "KS": comp["ks"], "Brier": comp["brier"],
        "Brier sem calibrar": comp["brier_bruto"], "Precisão": comp["precisao"], "Recall": comp["recall"],
        "F1": comp["f1"], "F1 com threshold fixo": comp.get("f1_limiar_fixo"), "Fatia-alvo (%)": comp.get("fatia_pct"),
        "Carteira sinalizada (%)": comp["pct_carteira_sinalizada"], "Threshold equivalente (%)": comp["limiar_pct"],
        "Inad. real (%)": comp["inad_media_real_pct"], "Inad. prevista (%)": comp["inad_media_prevista_pct"],
        "AUC PF": comp.get("auc_PF"), "AUC PJ": comp.get("auc_PJ"),
        "AUC cooperativas": comp.get("auc_Cooperativas"), "AUC custeio": comp.get("auc_Custeio"),
        "AUPRC": comp.get("auprc"), "H-measure": comp.get("h_measure"), "MCC": comp.get("mcc"),
        "Benefício líquido": comp.get("beneficio_medio"), "Brier Skill": comp.get("brier_skill"),
        "Intercepto de calibração": comp.get("calibracao_intercepto"), "Inclinação de calibração": comp.get("calibracao_inclinacao"),
        "ECE": comp.get("ece"),
    })
    tab = tab.apply(pd.to_numeric, errors="coerce")
    st.dataframe(tab.style.format(lambda v: "—" if pd.isna(v) else br(v, 4 if abs(v) < 0.1 else 3))
                 .highlight_max(subset=["AUC", "Gini", "KS", "Precisão", "Recall", "F1"], color="#c8e6c9")
                 .highlight_min(subset=["Brier"], color="#c8e6c9"),
                 use_container_width=True)
    st.caption("Precisão, recall e F1 medem reais de carteira, não número de produtores. Em cada mês, são sinalizados os "
               "perfis de maior risco até completar a fatia-alvo, ajustada na validação. Brier menor é melhor. "
               "Benefício líquido: média de 3, 5, 10 e 20 alarmes falsos aceitos por real inadimplente encontrado. "
               "Intercepto de calibração: alvo 0, negativo indica superestimação. Inclinação: alvo 1.")
    venc = meta.get("vencedores_por_criterio")
    if venc:
        st.caption(f"Vencedor por critério: F1 e AUC-ROC: **{venc['f1_auc']}**; só F1: **{venc['f1']}**; "
                   f"benefício líquido: **{venc['beneficio']}**. Para alternar: `python src/trocar_criterio.py f1_auc` "
                   "(ou `f1`, `beneficio`) e reinicie o dashboard.")
    prod = meta["producao"]
    st.caption(f"Modelo em produção no dashboard: **{meta['modelo_escolhido']}**, retreinado com todo o histórico "
               f"({fmt_int(meta['n_total'])} perfis-mês). Calibração: **{prod['calibracao']}**; fatia sinalizada: "
               f"**{br(prod.get('fatia_pct', 0))}%**, ajustadas nos perfis de {prod['janela_de'][:7]} a {prod['janela_ate'][:7]}. "
               f"Threshold na carteira atual: **{br(prod['limiar_pct'], 2)}%**.")

    with st.expander("Resultados na validação (usados para escolher o modelo)"):
        val = pd.DataFrame(meta["validacao"]).T
        st.dataframe(pd.DataFrame({
            "AUC sem calibrar": val.get("auc_bruto"), "AUC": val["auc"], "Brier": val["brier"], "Precisão": val["precisao"],
            "Recall": val["recall"], "F1": val["f1"], "Fatia (%)": val.get("fatia_pct"),
            "F1 na janela 2 com a fatia": val.get("f1_fatia_janela2"), "F1 máximo na janela 2": val.get("f1_janela2"),
            "Inad. real (%)": val["inad_media_real_pct"],
            "Inad. prevista (%)": val["inad_media_prevista_pct"],
        }).apply(pd.to_numeric, errors="coerce").style.format(lambda v: "—" if pd.isna(v) else br(v, 3)),
            use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Calibração no teste (por decil de risco previsto)**")
        cal = pd.DataFrame(meta["calibracao_teste"][meta["modelo_escolhido"]])
        fig = go.Figure()
        fig.add_bar(x=cal["faixa"] + 1, y=cal["real_pct"], name="Real", marker_color="#8d6e63")
        cal_bruta = pd.DataFrame(meta["calibracao_teste_bruta"][meta["modelo_escolhido"]])
        fig.add_scatter(x=cal_bruta["faixa"] + 1, y=cal_bruta["prev_pct"], name="Previsto sem calibrar", mode="lines+markers",
                        line=dict(color="#9e9e9e", width=2, dash="dot"))
        fig.add_scatter(x=cal["faixa"] + 1, y=cal["prev_pct"], name="Previsto calibrado", mode="lines+markers",
                        line=dict(color="#2e7d32", width=3))
        fig.update_layout(height=340, xaxis_title="decil de risco previsto", yaxis_title="inadimplência (%)",
                          margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.1),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown("**Importância global (média do |SHAP|)**")
        imp = pd.Series(meta["importancia_shap_global"]).drop("calibracao", errors="ignore").sort_values()
        fig = go.Figure(go.Bar(x=imp.values, y=[ROTULOS[i] for i in imp.index], orientation="h",
                               marker_color="#558b2f"))
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Precisão, recall e F1 por threshold (teste)**")
    curva = pd.DataFrame(meta["curva_limiar_teste"][meta["modelo_escolhido"]]).sort_values("limiar_pct")
    fig = go.Figure()
    for col, nome, cor in [("precisao", "Precisão", "#8d6e63"), ("recall", "Recall", "#558b2f"), ("f1", "F1", "#2e7d32")]:
        fig.add_scatter(x=curva["limiar_pct"], y=curva[col], name=nome, mode="lines", line=dict(color=cor, width=3))
    linha = float(meta["comparacao"][meta["modelo_escolhido"]]["limiar_pct"])
    fig.add_vline(x=linha, line_dash="dash", line_color="gray",
                  annotation_text=f"equivalente à fatia da validação: {br(linha, 2)}%")
    fig.update_layout(height=320, xaxis_title="threshold (% de probabilidade)", yaxis_title="valor da métrica",
                      margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.12),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    if "beneficio_teste" in meta:
        c1, c2 = st.columns(2)
        escolhido = meta["modelo_escolhido"]
        with c1:
            st.markdown("**Benefício líquido por alarmes falsos aceitos (teste)**")
            ben = pd.DataFrame(meta["beneficio_teste"][escolhido])
            fig = go.Figure()
            fig.add_scatter(x=ben["alarmes_por_acerto"], y=ben["modelo"], name="Modelo", mode="lines+markers", line=dict(color="#2e7d32", width=3))
            fig.add_scatter(x=ben["alarmes_por_acerto"], y=ben["todos"], name="Sinalizar todos", mode="lines+markers", line=dict(color="#8d6e63", dash="dot"))
            fig.add_hline(y=0, line_color="gray", annotation_text="Sinalizar ninguém")
            fig.update_layout(height=320, xaxis_title="alarmes falsos aceitos por real inadimplente", yaxis_title="benefício líquido por real",
                              margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.12),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.markdown("**Probabilidade prevista por resultado real (teste)**")
            dist = meta["distribuicao_teste"][escolhido]
            fig = go.Figure()
            fig.add_scatter(x=dist["centro_pct"], y=dist["adimplentes"], name="Adimplentes", mode="lines", line=dict(color="#558b2f", width=3))
            fig.add_scatter(x=dist["centro_pct"], y=dist["inadimplentes"], name="Inadimplentes", mode="lines", line=dict(color="#c62828", width=3))
            fig.update_layout(height=320, xaxis_title="probabilidade prevista (%)", yaxis_title="fração da exposição",
                              margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.12),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
        with st.expander("Intervalos de 95% no teste (reamostragem de UF e mês)"):
            st.dataframe(pd.DataFrame({
                nome: {f"{rot} ({lim})": v[chave][lim] for rot, chave in
                       [("AUC", "auc"), ("Brier", "brier"), ("F1", "f1"), ("Benefício", "beneficio_medio")]
                       for lim in ("baixo", "alto", "dif_baixo", "dif_alto")}
                for nome, v in meta["intervalos_teste"].items()}).T.style.format(lambda x: br(x, 4)), use_container_width=True)
            st.caption("As colunas dif mostram a diferença para o modelo do critério ativo. Se o intervalo inclui zero, a diferença pode ser ruído. "
                       "A reamostragem não captura toda a dependência entre os meses.")

    st.markdown("**Inadimplência da carteira rural brasileira (SCR.data)**")
    h = pd.DataFrame(meta["historico"])
    fig = go.Figure()
    for cli, cor in [("PF", "#2e7d32"), ("PJ", "#8d6e63")]:
        s = h[h.cliente == cli]
        fig.add_scatter(x=s["data"], y=s["inad_pct"], name=cli, mode="lines", line=dict(color=cor, width=3))
    fig.update_layout(height=300, yaxis_title="% da carteira com atraso > 90 dias", margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation="h", y=1.12), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

# ================================================================== DADOS
with aba_dados:
    st.markdown(f"""
### Fontes (todas oficiais e públicas)
| Fonte | O que entra no modelo |
|---|---|
| [SCR.data v2 — Banco Central](https://dadosabertos.bcb.gov.br/dataset/scr_data) | Carteira de crédito rural e inadimplência (>90 dias) por UF, tipo de instituição, PF/PJ, ocupação/CNAE, porte, finalidade, origem e indexador. Alvo do modelo e contexto regional. |
| [IBGE PAM, tabela 5457](https://sidra.ibge.gov.br/tabela/5457) | Área plantada, colhida e produção de soja, milho, arroz, trigo, feijão e algodão por UF (2014–2025). |
| [IBGE LSPA, tabela 6588](https://sidra.ibge.gov.br/tabela/6588) | Estimativa da safra do ano corrente (2026) por UF. Coletada, mas fora do modelo atual. |
| [BCB SGS](https://www3.bcb.gov.br/sgspub/) séries 432, 433 e 1 | Meta Selic, IPCA mensal e dólar comercial. |
| [Banco Mundial, Pink Sheet](https://www.worldbank.org/en/research/commodity-markets) | Preços internacionais mensais de produtos agrícolas, boi e fertilizantes, convertidos para reais. |
| [Conab, custos de produção](https://www.gov.br/conab/pt-br/atuacao/informacoes-agropecuarias/custos-de-producao/planilhas-de-custos-de-producao) | Custo total de soja e milho por município e safra. |

### Como o modelo funciona
- **Unidade de aprendizado:** cada perfil de crédito rural do SCR.data em um mês ({fmt_int(meta['n_total'])} observações reais, {meta['dados_ate'][:7]} como último mês com alvo conhecido).
- **Alvo:** fração da carteira desse perfil com atraso acima de 90 dias **{meta['horizonte_meses']} meses depois**. Cada perfil entra ponderado pela exposição em reais.
- **Modelos comparados:** Regressão Logística, Random Forest, XGBoost e XGBoost relativo à UF, além de versões com o
  histórico do próprio perfil e com mais peso nos meses recentes. O modelo relativo à UF parte do patamar atual de
  inadimplência rural da UF (offset em log-chance) e aprende quanto cada perfil desvia dele.
{TEXTO_ESCOLHA}
- **Calibração e sinalização:** a probabilidade é calibrada ({meta['producao']['calibracao']}) e cada mês sinaliza a fatia de maior risco da carteira,
  ambas ajustadas em previsões fora da amostra dos meses mais recentes com alvo conhecido.
- **Saída:** probabilidade de inadimplência de uma operação com aquele perfil, no cenário informado.
- **Explicabilidade:** valores SHAP do modelo final, agregados por variável original.

### Limites que o analista precisa conhecer
- O Brasil não publica inadimplência individual por produtor (sigilo bancário). O modelo estima o risco do **perfil**, não o histórico pessoal do produtor. Para nível individual, o mesmo pipeline pode ser treinado com a base interna de uma cooperativa ou banco.
- A inadimplência rural mudou de patamar entre 2022 (≈0,4%) e 2026 (≈7%). O modelo ordena bem o risco fora do tempo, mas o **nível** previsto pode ficar defasado em mudanças bruscas de ciclo; vale recalibrar periodicamente com o SCR.data mais recente.
- Classificações do SCR (ocupação, porte) são as declaradas pelas instituições financeiras.
""")
