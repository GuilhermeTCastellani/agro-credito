"""
Coleta de dados públicos oficiais.

Fontes:
  - SCR.data v2 (Banco Central)  -> carteira de crédito e inadimplência (>90 dias)
  - IBGE PAM tabela 5457          -> produção agrícola municipal, agregada por UF (2014-2025)
  - IBGE LSPA tabela 6588         -> levantamento sistemático da produção agrícola (ano corrente)
  - BCB SGS 432 / 433             -> meta Selic e IPCA mensal
  - Banco Mundial (Pink Sheet)    -> preços internacionais mensais de produtos agrícolas e fertilizantes
  - Conab (custos de produção)    -> custo por município e safra de soja e milho

Uso:
    python src/coleta.py                 # baixa tudo que faltar
    python src/coleta.py --anos 2024 2025
    python src/coleta.py --so-precos     # atualiza só os preços internacionais
    python src/coleta.py --so-custos     # atualiza só os custos da Conab
"""
from __future__ import annotations

import argparse
import io
import json
import re
import unicodedata
import zipfile
from pathlib import Path

import pandas as pd
import requests

RAIZ = Path(__file__).resolve().parents[1]
BRUTOS = RAIZ / "dados" / "brutos"
INTERM = RAIZ / "dados" / "intermediarios"

URL_SCR = "https://www.bcb.gov.br/pda/desig/scrdata_{ano}.zip"
URL_SGS = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{serie}/dados?formato=json&dataInicial={ini}&dataFinal={fim}"
SIDRA = "https://apisidra.ibge.gov.br/values"

# rótulo usado no SCR.data v2 (inclui dois espaços)
MODALIDADE_RURAL = "Financiamentos rurais  (ex-financiamentos rurais e agroindustriais)"
CHAVE_RURAL = "Financiamentos rurais"

COLS_NUM = [
    "numero_de_operacoes", "a_vencer_ate_90_dias", "a_vencer_de_91_ate_360_dias",
    "a_vencer_de_361_ate_1080_dias", "a_vencer_de_1081_ate_1800_dias",
    "a_vencer_de_1801_ate_5400_dias", "a_vencer_acima_de_5400_dias",
    "carteira_a_vencer", "vencido_de_15_ate_90_dias", "vencido_acima_de_90_dias",
    "carteira_vencida", "carteira_ativa", "carteira_inadimplencia", "ativo_problematico",
]

# Produtos PAM (c782) e LSPA (c48) usados no indicador de safra de grãos
PAM_PRODUTOS = {
    "40124": "soja", "40122": "milho", "40102": "arroz", "40127": "trigo",
    "40112": "feijao", "40099": "algodao", "40106": "cana", "40139": "cafe",
    "40119": "mandioca", "40151": "laranja",
}
LSPA_PRODUTOS = {
    "39443": "soja", "39441": "milho", "39442": "milho", "39432": "arroz",
    "39445": "trigo", "39436": "feijao", "39437": "feijao", "39438": "feijao",
    "39429": "algodao",
}
GRAOS = ["soja", "milho", "arroz", "trigo", "feijao", "algodao"]

_CONAB = ("https://www.gov.br/conab/pt-br/atuacao/informacoes-agropecuarias/custos-de-producao/"
          "arquivos-custo-de-producao/agricolas")
# (produto, arquivo local, endereço); os nomes têm o ano final e podem mudar quando a Conab atualizar a série
CUSTOS_CONAB = [
    ("soja", "custos_soja.xls", f"{_CONAB}/serie-historica-custos-soja-1997-a-2026.xls"),
    ("milho", "custos_milho_1a.xls", f"{_CONAB}/milho/milho_1a_safra_serie_historica_1997-2026.xls"),
    ("milho", "custos_milho_2a.xls", f"{_CONAB}/milho/milho_2a_safra_serie_historica_2005-2026.xls"),
]
MESES_PT = {m: i + 1 for i, m in enumerate(["janeiro", "fevereiro", "marco", "abril", "maio", "junho", "julho",
                                            "agosto", "setembro", "outubro", "novembro", "dezembro"])}

PAGINA_CMO = "https://www.worldbank.org/en/research/commodity-markets"
SERIES_CMO = {
    "Soybeans": "soja", "Maize": "milho", "Rice, Thai 5%": "arroz", "Wheat, US HRW": "trigo",
    "Cotton, A Index": "algodao", "Coffee, Arabica": "cafe", "Sugar, world": "cana", "Orange": "laranja",
    "Beef": "boi", "Urea": "ureia", "DAP": "dap", "Potassium chloride": "potassio",
}


def baixar(url: str, destino: Path, timeout: int = 1800) -> Path:
    if destino.exists() and destino.stat().st_size > 0:
        return destino
    destino.parent.mkdir(parents=True, exist_ok=True)
    print(f"  baixando {url}")
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        tmp = destino.with_suffix(destino.suffix + ".part")
        with open(tmp, "wb") as f:
            for bloco in r.iter_content(1 << 20):
                f.write(bloco)
        tmp.rename(destino)
    return destino


# ---------------------------------------------------------------- SCR.data
def extrair_scr_rural(anos: list[int]) -> pd.DataFrame:
    """Lê os CSVs mensais do SCR.data e mantém só a modalidade rural."""
    partes = []
    for ano in anos:
        zp = baixar(URL_SCR.format(ano=ano), BRUTOS / f"scrdata_{ano}.zip")
        with zipfile.ZipFile(zp) as z:
            for nome in sorted(z.namelist()):
                if not nome.endswith(".csv"):
                    continue
                with z.open(nome) as fh:
                    texto = io.TextIOWrapper(fh, encoding="utf-8-sig")
                    cab = texto.readline()
                    linhas = [cab] + [ln for ln in texto if CHAVE_RURAL in ln]
                df = pd.read_csv(io.StringIO("".join(linhas)), sep=";", dtype=str)
                df = df[df["modalidade"] == MODALIDADE_RURAL]
                for c in COLS_NUM:
                    df[c] = pd.to_numeric(df[c].str.replace(".", "", regex=False)
                                          .str.replace(",", ".", regex=False), errors="coerce")
                df["data_base"] = pd.to_datetime(df["data_base"])
                partes.append(df)
                print(f"  {nome}: {len(df):,} linhas rurais")
    out = pd.concat(partes, ignore_index=True)
    # numero_de_operacoes = -1 indica valor não divulgado (sigilo); vira ausente
    out.loc[out["numero_de_operacoes"] < 0, "numero_de_operacoes"] = pd.NA
    return out


# ---------------------------------------------------------------- IBGE
def _sidra(caminho: str) -> pd.DataFrame:
    r = requests.get(f"{SIDRA}/{caminho}", timeout=300)
    r.raise_for_status()
    dados = r.json()
    df = pd.DataFrame(dados[1:])
    df["V"] = pd.to_numeric(df["V"], errors="coerce")
    return df


def safra_por_uf() -> pd.DataFrame:
    """Área plantada, colhida e produção de grãos por UF e ano (PAM + LSPA ano corrente)."""
    prod = ",".join(PAM_PRODUTOS)
    pam = _sidra(f"t/5457/n3/all/v/8331,216,214/p/2014-2099/c782/{prod}")
    pam["produto"] = pam["D4C"].map(PAM_PRODUTOS)
    pam["ano"] = pam["D3C"].astype(int)
    pam["var"] = pam["D2C"].map({"8331": "area_plantada", "216": "area_colhida", "214": "producao"})
    pam = pam.rename(columns={"D1C": "cod_uf"})[["cod_uf", "ano", "produto", "var", "V"]]
    pam["fonte"] = "PAM"

    prod_l = ",".join(LSPA_PRODUTOS)
    lspa = _sidra(f"t/6588/n3/all/v/109,216,35/p/last%201/c48/{prod_l}")
    lspa["produto"] = lspa["D4C"].map(LSPA_PRODUTOS)
    lspa["ano"] = lspa["D3C"].str[:4].astype(int)
    lspa["var"] = lspa["D2C"].map({"109": "area_plantada", "216": "area_colhida", "35": "producao"})
    lspa = lspa.rename(columns={"D1C": "cod_uf"})[["cod_uf", "ano", "produto", "var", "V"]]
    lspa["fonte"] = "LSPA"
    # usa LSPA só para anos ainda sem PAM
    lspa = lspa[~lspa["ano"].isin(pam["ano"].unique())]

    base = pd.concat([pam, lspa])
    base = base.groupby(["cod_uf", "ano", "produto", "var", "fonte"], as_index=False)["V"].sum(min_count=1)
    return base


# ---------------------------------------------------------------- SGS
def serie_sgs(serie: int, ini="01/01/2014", fim="31/12/2026") -> pd.DataFrame:
    r = requests.get(URL_SGS.format(serie=serie, ini=ini, fim=fim), timeout=120)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    df["data"] = pd.to_datetime(df["data"], dayfirst=True)
    df["valor"] = pd.to_numeric(df["valor"])
    return df


def macro_mensal() -> pd.DataFrame:
    # séries diárias do SGS aceitam no máximo 10 anos por consulta
    selic = serie_sgs(432, ini="01/01/2018")  # meta Selic, % a.a. (diária)
    selic = selic.set_index("data")["valor"].resample("ME").last().rename("selic")
    ipca = serie_sgs(433, ini="01/01/2017").set_index("data")["valor"].rename("ipca_mes")  # % no mês
    ipca.index = ipca.index + pd.offsets.MonthEnd(0)
    dolar = serie_sgs(1, ini="01/01/2018")  # dólar comercial venda, R$/US$ (diária)
    dolar = dolar.set_index("data")["valor"].resample("ME").mean().rename("dolar")
    df = pd.concat([selic, ipca, dolar], axis=1, sort=True)
    df["dolar_var12m"] = df["dolar"].pct_change(12) * 100
    df["ipca_12m"] = ((1 + df["ipca_mes"] / 100).rolling(12).apply(lambda x: x.prod()) - 1) * 100
    df.index.name = "mes"
    return df.reset_index()


# ---------------------------------------------------------------- Banco Mundial
def link_cmo(html: str) -> str:
    """O endereço da planilha muda a cada atualização, então ele é lido da página oficial."""
    links = re.findall(r"https://thedocs\.worldbank\.org/[^\"'\s]*CMO-Historical-Data-Monthly\.xlsx", html)
    if not links:
        raise RuntimeError(f"planilha mensal do Banco Mundial não encontrada em {PAGINA_CMO}")
    return links[0]


def ler_cmo(caminho: Path) -> pd.DataFrame:
    """Lê a aba de preços mensais; devolve o mês e uma coluna em dólar para cada série usada."""
    bruto = pd.read_excel(caminho, sheet_name="Monthly Prices", header=None)
    nomes = [str(v).replace("**", "").strip() for v in bruto.iloc[4].tolist()]
    faltam = set(SERIES_CMO) - set(nomes)
    if faltam:
        raise RuntimeError(f"séries ausentes na planilha do Banco Mundial: {sorted(faltam)}")
    meses = bruto[0].astype(str).str.fullmatch(r"\d{4}M\d{2}")
    linhas = bruto[meses]
    out = pd.DataFrame({"mes": pd.to_datetime(linhas[0].str.replace("M", "-"), format="%Y-%m") + pd.offsets.MonthEnd(0)})
    for coluna, nome in enumerate(nomes):
        if nome in SERIES_CMO:
            out[SERIES_CMO[nome]] = pd.to_numeric(linhas[coluna], errors="coerce")
    return out.reset_index(drop=True)


def precos_mundiais() -> pd.DataFrame:
    pagina = requests.get(PAGINA_CMO, timeout=120, headers={"User-Agent": "agro-credito-poc"})
    pagina.raise_for_status()
    url = link_cmo(pagina.text)
    versao = url.split("/")[-3]
    return ler_cmo(baixar(url, BRUTOS / f"cmo_{versao}.xlsx"))


# ---------------------------------------------------------------- Conab
def _sem_acento(valor) -> str:
    return unicodedata.normalize("NFKD", str(valor)).encode("ascii", "ignore").decode().lower().strip()


def ler_aba_conab(aba: pd.DataFrame) -> dict:
    """Extrai data dos preços, produtividade e custo total por hectare; o formato da aba mudou ao longo dos anos."""
    r = {"data_ref": None, "produtividade_kg_ha": None, "custo_total_ha": None}
    for linha in aba.itertuples(index=False):
        valores = list(linha)
        rotulo = _sem_acento(valores[0])
        tudo = " ".join(_sem_acento(v) for v in valores if not pd.isna(v))
        numeros = [v for v in valores[1:] if isinstance(v, (int, float)) and not pd.isna(v)]
        if r["data_ref"] is None and "a precos de" in tudo:
            for v in valores:
                if hasattr(v, "year") and not pd.isna(v):
                    r["data_ref"] = pd.Timestamp(v) + pd.offsets.MonthEnd(0)
        if r["data_ref"] is None and rotulo.startswith("mes/ano"):
            m = re.search(r"([a-z]+)\s*/\s*(\d{4})", rotulo)
            if m and m.group(1) in MESES_PT:
                r["data_ref"] = pd.Timestamp(int(m.group(2)), MESES_PT[m.group(1)], 1) + pd.offsets.MonthEnd(0)
        if r["produtividade_kg_ha"] is None and "produtividade" in rotulo:
            m = re.search(r"([\d.,]+)\s*kg", tudo)
            if numeros:
                r["produtividade_kg_ha"] = float(numeros[0])
            elif m:
                r["produtividade_kg_ha"] = float(m.group(1).replace(".", "").replace(",", "."))
        if r["custo_total_ha"] is None and "custo total" in rotulo and numeros:
            r["custo_total_ha"] = float(numeros[0])
    return r


def ler_custos_conab(caminho: Path, produto: str) -> pd.DataFrame:
    """Uma linha por aba lida. Abas sem custo, data ou produtividade ficam de fora."""
    planilha = pd.ExcelFile(caminho)
    linhas = []
    for nome in planilha.sheet_names[1:]:
        m = re.match(r"(.+)-([A-Z]{2})-?\s*\d{4}", nome)
        if m:
            linhas.append({"produto": produto, "local": m.group(1), "uf": m.group(2), "aba": nome,
                           **ler_aba_conab(planilha.parse(nome, header=None))})
    out = pd.DataFrame(linhas)
    print(f"  {caminho.name}: {len(out)} abas, {out['custo_total_ha'].notna().sum()} com custo")
    return out.dropna(subset=["data_ref", "produtividade_kg_ha", "custo_total_ha"])


def custos_conab() -> pd.DataFrame:
    partes = [ler_custos_conab(baixar(url, BRUTOS / arquivo), produto) for produto, arquivo, url in CUSTOS_CONAB]
    return pd.concat(partes, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--so-precos", action="store_true", help="atualiza só os preços internacionais")
    ap.add_argument("--so-custos", action="store_true", help="atualiza só os custos da Conab")
    ap.add_argument("--anos", type=int, nargs="+", default=[2022, 2023, 2024, 2025, 2026])
    a = ap.parse_args()
    INTERM.mkdir(parents=True, exist_ok=True)

    if a.so_precos or not a.so_custos:
        print("Banco Mundial (preços)...")
        precos_mundiais().to_parquet(INTERM / "precos_mundo.parquet", index=False)
    if a.so_custos or not a.so_precos:
        print("Conab (custos)...")
        custos_conab().to_parquet(INTERM / "custos_conab.parquet", index=False)
    if a.so_precos or a.so_custos:
        print("ok")
        return

    alvo_scr = INTERM / "scr_rural.parquet"
    if alvo_scr.exists():
        print("SCR.data já extraído (apague dados/intermediarios/scr_rural.parquet para refazer)")
    else:
        print("SCR.data (modalidade rural)...")
        extrair_scr_rural(a.anos).to_parquet(alvo_scr, index=False)

    print("IBGE PAM/LSPA...")
    safra_por_uf().to_parquet(INTERM / "safra_uf.parquet", index=False)

    print("BCB SGS (Selic, IPCA)...")
    macro_mensal().to_parquet(INTERM / "macro.parquet", index=False)
    print("ok")


if __name__ == "__main__":
    main()
