"""
Monta a base de modelagem a partir dos dados oficiais coletados.

Unidade de observação: um perfil de crédito rural em um mês (célula do SCR.data):
    UF x segmento da instituição x PF/PJ x ocupação/CNAE x porte x submodalidade
       x origem dos recursos x indexador

Alvo: fração da carteira desse perfil com parcela vencida há mais de 90 dias,
      observada HORIZONTE meses depois (previsão, não descrição).

Variáveis explicativas:
  - perfil do tomador e da operação (o que o analista informa no dashboard)
  - contexto no mês t, preenchido automaticamente:
      Selic, IPCA 12m, variação do dólar 12m (BCB/SGS)
      desempenho da última safra de grãos na UF já publicada no mês t (IBGE PAM)
      inadimplência e atraso curto (15-90d) da carteira rural da UF (SCR.data)
      histórico do próprio perfil: inadimplência, atraso, ativo problemático, vencimentos próximos,
      variação em 1, 3, 6 e 12 meses, crescimento da carteira e inadimplência de perfis parecidos
      preços (Banco Mundial, em reais): cesta agrícola da UF, fertilizantes, boi e relação de troca
      custos (Conab, soja e milho): margem estimada e variação anual do custo total na UF
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from modelo import CUSTOS, LSPA, PERFIL, PERFIL_AMPLIADO, PRECOS

RAIZ = Path(__file__).resolve().parents[1]
INTERM = RAIZ / "dados" / "intermediarios"
PROC = RAIZ / "dados" / "processados"

HORIZONTE = 6  # meses
MES_PUBLICACAO_PAM = 10  # premissa: o PAM de um ano fica disponível em outubro do ano seguinte
MESES_PUBLICACAO_CONAB = 3  # premissa conservadora: a planilha não informa quando cada custo foi publicado
MAX_IDADE_CUSTO_MESES = 24  # custos de locais inativados pela Conab deixam de contar depois disso

CHAVES = ["uf", "segmento", "cliente", "cnae_ocupacao", "porte",
          "submodalidade", "origem", "indexador"]

COD_UF = {
    "11": "RO", "12": "AC", "13": "AM", "14": "RR", "15": "PA", "16": "AP", "17": "TO",
    "21": "MA", "22": "PI", "23": "CE", "24": "RN", "25": "PB", "26": "PE", "27": "AL",
    "28": "SE", "29": "BA", "31": "MG", "32": "ES", "33": "RJ", "35": "SP", "41": "PR",
    "42": "SC", "43": "RS", "50": "MS", "51": "MT", "52": "GO", "53": "DF",
}
REGIAO = {
    **{u: "Norte" for u in ["RO", "AC", "AM", "RR", "PA", "AP", "TO"]},
    **{u: "Nordeste" for u in ["MA", "PI", "CE", "RN", "PB", "PE", "AL", "SE", "BA"]},
    **{u: "Sudeste" for u in ["MG", "ES", "RJ", "SP"]},
    **{u: "Sul" for u in ["PR", "SC", "RS"]},
    **{u: "Centro-Oeste" for u in ["MS", "MT", "GO", "DF"]},
}

# ponto médio (em anos) de cada faixa de prazo a vencer do SCR.data
FAIXAS_PRAZO = {
    "a_vencer_ate_90_dias": 45 / 365,
    "a_vencer_de_91_ate_360_dias": 225 / 365,
    "a_vencer_de_361_ate_1080_dias": 720 / 365,
    "a_vencer_de_1081_ate_1800_dias": 1440 / 365,
    "a_vencer_de_1801_ate_5400_dias": 3600 / 365,
    "a_vencer_acima_de_5400_dias": 6000 / 365,
}
GRAOS = ["soja", "milho", "arroz", "trigo", "feijao", "algodao"]

# Converte o preço internacional em dólares por tonelada colhida, só para pesar os produtos da cesta da UF.
# Aproximações: arroz em casca rende ~65% de beneficiado; algodão em caroço, ~38% de pluma; cana, ~13% de açúcar.
FATOR_RECEITA = {"soja": 1, "milho": 1, "trigo": 1, "arroz": 0.65, "algodao": 380, "cafe": 1000, "cana": 130, "laranja": 1000}
FERTILIZANTES = ["ureia", "dap", "potassio"]


def indicadores_safra() -> pd.DataFrame:
    """Por UF e ano-safra: produtividade de grãos vs média dos 5 anos anteriores e perda de área."""
    s = pd.read_parquet(INTERM / "safra_uf.parquet")
    s = s[s["produto"].isin(GRAOS)]
    t = s.pivot_table(index=["cod_uf", "ano"], columns="var", values="V", aggfunc="sum").reset_index()
    t["uf"] = t["cod_uf"].map(COD_UF)
    t = t.sort_values(["uf", "ano"])
    t["produtividade"] = t["producao"] / t["area_colhida"]
    t["prod_por_area_plantada"] = t["producao"] / t["area_plantada"]
    media5 = (t.groupby("uf")["prod_por_area_plantada"]
                .transform(lambda x: x.shift(1).rolling(5, min_periods=3).mean()))
    t["safra_anomalia_pct"] = (t["prod_por_area_plantada"] / media5 - 1) * 100
    t["safra_perda_area_pct"] = (1 - t["area_colhida"] / t["area_plantada"]) * 100
    t["safra_var_producao_pct"] = t.groupby("uf")["producao"].pct_change() * 100
    # UFs sem produção relevante de grãos (ex.: AP, RR em alguns anos) ficam com 0 = "sem choque"
    cols = ["safra_anomalia_pct", "safra_perda_area_pct", "safra_var_producao_pct"]
    t[cols] = t[cols].replace([np.inf, -np.inf], np.nan)
    return t[["uf", "ano"] + cols]


PRODUCAO_MINIMA_LSPA = 50_000  # toneladas; abaixo disso a variação de grãos da UF é ruído (AP, RR, AM em alguns anos)


def variaveis_lspa(lev: pd.DataFrame | None = None) -> pd.DataFrame:
    """Safra corrente estimada pela LSPA, por UF e mês, só com levantamentos já publicados.

    Premissa: o levantamento do mês m sai no mês m + 1, então o mês t usa o levantamento de t − 1.
      lspa_var_producao_pct: produção de grãos estimada para o ano do levantamento contra a estimativa final (dezembro) do ano anterior
      lspa_revisao_3m_pct  : quanto a estimativa do mesmo ano mudou desde 3 levantamentos antes (ou desde janeiro, se for mais recente)
    """
    l = (pd.read_parquet(INTERM / "lspa_levantamentos.parquet") if lev is None else lev).copy()
    l["uf"] = l["cod_uf"].astype(str).map(COD_UF)
    l["ano"] = l["levantamento"].dt.year
    l = l.dropna(subset=["uf"]).sort_values(["uf", "levantamento"])
    final = l[l["levantamento"].dt.month == 12][["uf", "ano", "producao_t"]].rename(columns={"producao_t": "final_anterior"})
    final["ano"] += 1
    l = l.merge(final, on=["uf", "ano"], how="left")
    base_ok = l["final_anterior"] >= PRODUCAO_MINIMA_LSPA
    l["lspa_var_producao_pct"] = np.where(base_ok, (l["producao_t"] / l["final_anterior"] - 1) * 100, np.nan)
    # revisão: compara com o levantamento de max(m − 3, janeiro) do mesmo ano
    ref = l["levantamento"] - pd.DateOffset(months=3)
    ref = ref.where(ref.dt.year == l["ano"], pd.to_datetime(l["ano"].astype(str) + "-01-31"))
    ref = ref + pd.offsets.MonthEnd(0)
    anterior = l[["uf", "levantamento", "producao_t"]].rename(columns={"levantamento": "ref", "producao_t": "producao_ref"})
    l = l.assign(ref=ref).merge(anterior, on=["uf", "ref"], how="left")
    ok = l["producao_ref"] >= PRODUCAO_MINIMA_LSPA
    l["lspa_revisao_3m_pct"] = np.where(ok, (l["producao_t"] / l["producao_ref"] - 1) * 100, np.nan)
    l["data_base"] = l["levantamento"] + pd.offsets.MonthEnd(1)  # publicado no mês seguinte
    return l[["data_base", "uf"] + LSPA]


def contexto_uf(scr: pd.DataFrame) -> pd.DataFrame:
    """Inadimplência e atraso curto da carteira rural de cada UF no mês."""
    g = scr.groupby(["data_base", "uf"])[
        ["carteira_ativa", "carteira_inadimplencia", "vencido_de_15_ate_90_dias"]].sum()
    out = pd.DataFrame({
        "uf_inad_atual_pct": g["carteira_inadimplencia"] / g["carteira_ativa"] * 100,
        "uf_atraso_15_90_pct": g["vencido_de_15_ate_90_dias"] / g["carteira_ativa"] * 100,
    }).reset_index().sort_values(["uf", "data_base"])
    out["uf_inad_var_6m_pp"] = out.groupby("uf")["uf_inad_atual_pct"].diff(6)
    return out


def ano_safra_publicada(datas: pd.Series) -> np.ndarray:
    """Último ano com PAM já publicado em cada mês, para não usar safra que ainda não era conhecida."""
    return np.where(datas.dt.month >= MES_PUBLICACAO_PAM, datas.dt.year - 1, datas.dt.year - 2)


def historico_perfil(scr: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta as variáveis do próprio perfil, usando só dados do mês t ou anteriores."""
    out = scr.copy()
    ativa = out["carteira_ativa"]
    out["perfil_inad_atual_pct"] = (out["carteira_inadimplencia"] / ativa).clip(0, 1) * 100
    out["perfil_atraso_15_90_pct"] = (out["vencido_de_15_ate_90_dias"] / ativa).clip(0, 1) * 100
    out["perfil_problematico_pct"] = (out["ativo_problematico"] / ativa).clip(0, 1) * 100
    out["perfil_vence_90d_pct"] = (out["a_vencer_ate_90_dias"] / ativa).clip(0, 1) * 100
    out["mes_alvo"] = (out["data_base"] + pd.DateOffset(months=HORIZONTE)).dt.month

    g = (out.groupby(["data_base"] + CHAVES, dropna=False)[["carteira_inadimplencia", "carteira_ativa"]]
            .sum().reset_index())
    g["inad_passada"] = (g["carteira_inadimplencia"] / g["carteira_ativa"]).clip(0, 1) * 100
    g = g.rename(columns={"carteira_ativa": "ativa_passada"})
    chaves = ["data_base"] + CHAVES
    for meses in (1, 3, 6, 12):
        lag = g[chaves + ["inad_passada", "ativa_passada"]].copy()
        lag["data_base"] = lag["data_base"] + pd.DateOffset(months=meses) + pd.offsets.MonthEnd(0)
        out = out.merge(lag, on=chaves, how="left")
        out[f"perfil_inad_var_{meses}m_pp"] = out["perfil_inad_atual_pct"] - out.pop("inad_passada")
        passada = out.pop("ativa_passada")
        if meses in (6, 12):
            out[f"perfil_cresc_{meses}m_pct"] = ((out["carteira_ativa"] / passada - 1) * 100).clip(-100, 1000)

    def taxa_do_grupo(colunas):
        grupo = out.groupby(colunas, dropna=False)
        return (grupo["carteira_inadimplencia"].transform("sum") / grupo["carteira_ativa"].transform("sum")).clip(0, 1) * 100

    out["pares_inad_atual_pct"] = taxa_do_grupo(["data_base", "segmento", "cliente", "porte", "submodalidade", "origem", "indexador"])
    out["uf_finalidade_inad_pct"] = taxa_do_grupo(["data_base", "uf", "submodalidade"])
    return out


def variaveis_precos(safra: pd.DataFrame, precos: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """Variação de preços em reais por UF e mês; a cesta usa os pesos do último PAM já publicado."""
    produtos = list(FATOR_RECEITA)
    p = precos.merge(macro[["mes", "dolar"]], on="mes", how="inner").sort_values("mes").reset_index(drop=True)
    series = produtos + FERTILIZANTES + ["boi"]
    reais = p[series].multiply(p["dolar"], axis=0)
    var = {m: (reais / reais.shift(m) - 1) * 100 for m in (6, 12)}

    # pesos: receita aproximada de cada produto na UF, com o preço médio em dólar do próprio ano do PAM
    media_ano = precos.assign(ano=precos["mes"].dt.year).groupby("ano")[produtos].mean()
    s = safra[(safra["fonte"] == "PAM") & (safra["var"] == "producao") & safra["produto"].isin(produtos)
              & (safra["V"] > 0)].copy()
    s["uf"] = s["cod_uf"].astype(str).map(COD_UF)
    s["receita"] = [v * FATOR_RECEITA[prod] * media_ano[prod].get(ano, np.nan)
                    for v, prod, ano in zip(s["V"], s["produto"], s["ano"])]
    s = s.dropna(subset=["receita"])
    s["peso"] = s["receita"] / s.groupby(["uf", "ano"])["receita"].transform("sum")

    longo = pd.DataFrame({"mes": p["mes"]}).assign(ano=lambda d: ano_safra_publicada(d["mes"]))
    cesta = longo.merge(s[["uf", "ano", "produto", "peso"]], on="ano", how="inner")
    for m in (6, 12):
        tab = pd.concat([p[["mes"]], var[m][produtos]], axis=1).melt("mes", var_name="produto", value_name=f"v{m}")
        cesta = cesta.merge(tab, on=["mes", "produto"], how="left")

    def media_ponderada(col):
        ok = cesta[col].notna()
        num = (cesta[col] * cesta["peso"]).where(ok).groupby([cesta["mes"], cesta["uf"]]).sum(min_count=1)
        den = cesta["peso"].where(ok).groupby([cesta["mes"], cesta["uf"]]).sum(min_count=1)
        return num / den

    out = pd.DataFrame({"preco_cesta_var_6m_pct": media_ponderada("v6"),
                        "preco_cesta_var_12m_pct": media_ponderada("v12")}).reset_index()
    nacionais = pd.DataFrame({"mes": p["mes"],
                              "fert_var_6m_pct": var[6][FERTILIZANTES].mean(axis=1),
                              "fert_var_12m_pct": var[12][FERTILIZANTES].mean(axis=1),
                              "boi_var_12m_pct": var[12]["boi"]})
    out = out.merge(nacionais, on="mes", how="left")
    out["relacao_troca_var_12m_pct"] = ((1 + out["preco_cesta_var_12m_pct"] / 100)
                                        / (1 + out["fert_var_12m_pct"] / 100) - 1) * 100
    return out.rename(columns={"mes": "data_base"})[["data_base", "uf"] + PRECOS]


def precos_por_uf() -> pd.DataFrame:
    return variaveis_precos(pd.read_parquet(INTERM / "safra_uf.parquet"),
                            pd.read_parquet(INTERM / "precos_mundo.parquet"),
                            pd.read_parquet(INTERM / "macro.parquet"))


def variaveis_custos(custos: pd.DataFrame, safra: pd.DataFrame, precos: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """Margem estimada e variação do custo por UF e mês, usando só custos já disponíveis no mês.

    A margem compara o preço internacional em reais com o custo total por saca de 60 kg. Soja e milho são pesados pela
    produção do último PAM publicado. UFs sem custo da Conab ficam sem linha e são tratadas como ausentes.
    """
    c = custos.sort_values("data_ref").copy()
    c["custo_saca"] = c["custo_total_ha"] / (c["produtividade_kg_ha"] / 60)
    c["chave"] = c["produto"] + "|" + c["local"] + "|" + c["uf"]
    anterior = c.groupby("chave")["custo_total_ha"].shift(1)
    data_anterior = c.groupby("chave")["data_ref"].shift(1)
    consecutivo = (c["data_ref"] - data_anterior).dt.days.between(300, 430)
    c["custo_var"] = ((c["custo_total_ha"] / anterior - 1) * 100).where(consecutivo)
    c["disponivel"] = c["data_ref"] + pd.DateOffset(months=MESES_PUBLICACAO_CONAB) + pd.offsets.MonthEnd(0)

    p = precos.merge(macro[["mes", "dolar"]], on="mes", how="inner").sort_values("mes").reset_index(drop=True)
    meses = p[["mes"]].astype({"mes": "datetime64[ns]"})
    partes = []
    for _, grupo in c.groupby("chave"):
        recente = pd.merge_asof(meses, grupo.sort_values("disponivel").astype({"disponivel": "datetime64[ns]"}),
                                left_on="mes", right_on="disponivel",
                                tolerance=pd.Timedelta(days=MAX_IDADE_CUSTO_MESES * 31))
        partes.append(recente.dropna(subset=["custo_saca"]))
    if not partes:
        return pd.DataFrame(columns=["data_base", "uf"] + CUSTOS)
    por_uf = (pd.concat(partes).groupby(["mes", "uf", "produto"], as_index=False)
              .agg(custo_saca=("custo_saca", "mean"), custo_var=("custo_var", "mean")))

    preco = pd.concat([p[["mes"]], p[["soja", "milho"]].multiply(p["dolar"] * 0.06, axis=0)], axis=1)
    preco = preco.melt("mes", var_name="produto", value_name="preco_saca")
    por_uf = por_uf.merge(preco, on=["mes", "produto"], how="left")
    por_uf["margem"] = (por_uf["preco_saca"] / por_uf["custo_saca"] - 1) * 100

    s = safra[(safra["fonte"] == "PAM") & (safra["var"] == "producao") & safra["produto"].isin(["soja", "milho"])
              & (safra["V"] > 0)].copy()
    s["uf"] = s["cod_uf"].astype(str).map(COD_UF)
    por_uf["ano"] = ano_safra_publicada(por_uf["mes"])
    por_uf = por_uf.merge(s[["uf", "ano", "produto", "V"]], on=["uf", "ano", "produto"], how="inner")

    def media_ponderada(col):
        ok = por_uf[col].notna()
        chaves = [por_uf["mes"], por_uf["uf"]]
        return (por_uf[col] * por_uf["V"]).where(ok).groupby(chaves).sum(min_count=1) / por_uf["V"].where(ok).groupby(chaves).sum(min_count=1)

    out = pd.DataFrame({"conab_margem_pct": media_ponderada("margem"),
                        "conab_custo_var_anual_pct": media_ponderada("custo_var")}).reset_index()
    return out.rename(columns={"mes": "data_base"})[["data_base", "uf"] + CUSTOS]


def custos_por_uf() -> pd.DataFrame:
    return variaveis_custos(pd.read_parquet(INTERM / "custos_conab.parquet"),
                            pd.read_parquet(INTERM / "safra_uf.parquet"),
                            pd.read_parquet(INTERM / "precos_mundo.parquet"),
                            pd.read_parquet(INTERM / "macro.parquet"))


def ler_scr() -> pd.DataFrame:
    scr = pd.read_parquet(INTERM / "scr_rural.parquet")
    scr["submodalidade"] = scr["submodalidade"].str.strip()
    return scr[scr["carteira_ativa"] > 0].copy()


def perfil_atual() -> pd.DataFrame:
    """Histórico mais recente de cada perfil, usado pelo dashboard para preencher as variáveis do perfil."""
    h = historico_perfil(ler_scr())
    h = h[h["data_base"] == h["data_base"].max()].rename(columns={"cnae_ocupacao": "ocupacao_atividade"})
    chaves = ["ocupacao_atividade" if c == "cnae_ocupacao" else c for c in CHAVES]
    return h[chaves + PERFIL + PERFIL_AMPLIADO].drop_duplicates(chaves)


COLUNAS_BASE = (["data_base", "uf", "regiao", "segmento", "cliente", "ocupacao_atividade", "porte",
                 "submodalidade", "origem", "indexador",
                 "prazo_remanescente_anos", "valor_medio_operacao_mil",
                 "selic", "ipca_12m", "dolar_var12m",
                 "safra_anomalia_pct", "safra_perda_area_pct", "safra_var_producao_pct",
                 "uf_inad_atual_pct", "uf_atraso_15_90_pct", "uf_inad_var_6m_pp"] + PERFIL + PERFIL_AMPLIADO + PRECOS + CUSTOS + LSPA +
                ["alvo_inad", "peso"])


def _operacao(scr: pd.DataFrame) -> pd.DataFrame:
    a_vencer = scr[list(FAIXAS_PRAZO)].sum(axis=1)
    prazo = sum(scr[c] * m for c, m in FAIXAS_PRAZO.items())
    scr["prazo_remanescente_anos"] = np.where(a_vencer > 0, prazo / a_vencer.replace(0, np.nan), np.nan)
    scr["valor_medio_operacao_mil"] = scr["carteira_ativa"] / scr["numero_de_operacoes"] / 1e3
    return scr


def _contexto(base: pd.DataFrame, scr: pd.DataFrame) -> pd.DataFrame:
    base = base.merge(contexto_uf(scr), on=["data_base", "uf"], how="left")

    macro = pd.read_parquet(INTERM / "macro.parquet").rename(columns={"mes": "data_base"})
    base = base.merge(macro[["data_base", "selic", "ipca_12m", "dolar_var12m"]], on="data_base", how="left")

    base["ano_safra"] = ano_safra_publicada(base["data_base"])
    base = base.merge(indicadores_safra().rename(columns={"ano": "ano_safra"}),
                      on=["uf", "ano_safra"], how="left")
    base = base.merge(precos_por_uf(), on=["data_base", "uf"], how="left")
    base = base.merge(custos_por_uf(), on=["data_base", "uf"], how="left")
    base = base.merge(variaveis_lspa(), on=["data_base", "uf"], how="left")

    base["regiao"] = base["uf"].map(REGIAO)
    base = base.rename(columns={"cnae_ocupacao": "ocupacao_atividade"})
    return base[COLUNAS_BASE]


def montar() -> pd.DataFrame:
    scr = _operacao(historico_perfil(ler_scr()))

    # --- alvo: inadimplência do mesmo perfil HORIZONTE meses depois ---
    fut = scr[["data_base"] + CHAVES + ["carteira_ativa", "carteira_inadimplencia"]].copy()
    fut["data_base"] = fut["data_base"] - pd.DateOffset(months=HORIZONTE)
    fut["data_base"] = fut["data_base"] + pd.offsets.MonthEnd(0)
    fut = fut.rename(columns={"carteira_ativa": "carteira_futura",
                              "carteira_inadimplencia": "inad_futura"})
    base = scr.merge(fut, on=["data_base"] + CHAVES, how="inner")
    base["alvo_inad"] = (base["inad_futura"] / base["carteira_futura"]).clip(0, 1)
    base["peso"] = base["carteira_futura"] / 1e6  # exposição em R$ milhões
    return _contexto(base, scr)


def montar_atual() -> pd.DataFrame:
    """Perfis do último mês, ainda sem alvo conhecido; definem o threshold da fatia em produção."""
    scr = _operacao(historico_perfil(ler_scr()))
    base = scr[scr["data_base"] == scr["data_base"].max()].copy()
    base["alvo_inad"] = np.nan
    base["peso"] = base["carteira_ativa"] / 1e6
    return _contexto(base, scr)


def contexto_atual() -> dict:
    """Valores de contexto mais recentes por UF, usados pelo dashboard para preencher o cenário."""
    scr = pd.read_parquet(INTERM / "scr_rural.parquet")
    ctx = contexto_uf(scr)
    ultimo = ctx["data_base"].max()
    ctx = ctx[ctx["data_base"] == ultimo].set_index("uf")
    macro = pd.read_parquet(INTERM / "macro.parquet").dropna(subset=["selic"])
    m = macro[macro["mes"] <= ultimo].iloc[-1]
    ano_safra = int(ano_safra_publicada(pd.Series([ultimo]))[0])
    saf = indicadores_safra()
    saf = saf[saf["ano"] == ano_safra].set_index("uf")
    prc = precos_por_uf()
    prc = prc[prc["data_base"] == ultimo].drop(columns="data_base").set_index("uf")
    cst = custos_por_uf()
    cst = cst[cst["data_base"] == ultimo].drop(columns="data_base").set_index("uf")
    lsp = variaveis_lspa()
    lsp = lsp[lsp["data_base"] == ultimo].drop(columns="data_base").set_index("uf")
    return {
        "data_base": str(ultimo.date()),
        "ano_safra": int(ano_safra),
        "macro": {"selic": float(m["selic"]), "ipca_12m": float(m["ipca_12m"]),
                  "dolar_var12m": float(m["dolar_var12m"])},
        "uf": {uf: {**{k: (None if pd.isna(v) else float(v)) for k, v in ctx.loc[uf].drop("data_base").items()},
                    **{k: (None if pd.isna(v) else float(v))
                       for k, v in (saf.loc[uf] if uf in saf.index else pd.Series(dtype=float)).items()},
                    **{k: (None if pd.isna(v) else float(v))
                       for k, v in (prc.loc[uf] if uf in prc.index else pd.Series(dtype=float)).items()},
                    **{c: None for c in CUSTOS},
                    **{k: (None if pd.isna(v) else float(v))
                       for k, v in (cst.loc[uf] if uf in cst.index else pd.Series(dtype=float)).items()},
                    **{c: None for c in LSPA},
                    **{k: (None if pd.isna(v) else float(v))
                       for k, v in (lsp.loc[uf] if uf in lsp.index else pd.Series(dtype=float)).items()}}
               for uf in ctx.index},
    }


if __name__ == "__main__":
    import json
    PROC.mkdir(parents=True, exist_ok=True)
    df = montar()
    df.to_parquet(PROC / "base_modelagem.parquet", index=False, compression="zstd")  # abaixo do limite de 100 MB por arquivo do GitHub
    perfil_atual().to_parquet(PROC / "perfil_atual.parquet", index=False)
    montar_atual().to_parquet(PROC / "base_atual.parquet", index=False)
    with open(PROC / "contexto_atual.json", "w", encoding="utf-8") as f:
        json.dump(contexto_atual(), f, ensure_ascii=False, indent=1)
    print(df.shape)
    print(df["data_base"].min().date(), "->", df["data_base"].max().date())
    print("inadimplência média ponderada (alvo):", round(np.average(df["alvo_inad"], weights=df["peso"]) * 100, 2), "%")
    print(df.isna().mean().round(3).to_string())
