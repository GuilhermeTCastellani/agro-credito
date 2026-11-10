"""Variáveis da LSPA: só levantamentos já publicados, variação contra o ano anterior e revisão de 3 meses."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import features as f  # noqa: E402


def _levantamentos():
    meses = pd.date_range("2023-01-31", "2024-06-30", freq="ME")
    prod = {m: 1_000_000.0 for m in meses}
    prod[pd.Timestamp("2023-12-31")] = 2_000_000.0  # estimativa final de 2023
    prod[pd.Timestamp("2024-01-31")] = 2_200_000.0
    prod[pd.Timestamp("2024-04-30")] = 1_650_000.0  # choque de clima em abril
    return pd.DataFrame({"cod_uf": "43", "levantamento": list(prod), "producao_t": list(prod.values())})


def test_usa_so_o_levantamento_do_mes_anterior():
    v = f.variaveis_lspa(_levantamentos()).set_index("data_base")
    # o choque do levantamento de abril só aparece no mês de maio
    assert np.isclose(v.loc["2024-04-30", "lspa_var_producao_pct"], -50.0)  # abril ainda usa o levantamento de março
    assert np.isclose(v.loc["2024-05-31", "lspa_var_producao_pct"], (1_650_000 / 2_000_000 - 1) * 100)


def test_variacao_contra_a_estimativa_final_do_ano_anterior():
    v = f.variaveis_lspa(_levantamentos()).set_index("data_base")
    assert np.isclose(v.loc["2024-02-29", "lspa_var_producao_pct"], 10.0)  # levantamento de jan/2024: 2,2 mi contra 2,0 mi
    assert v.loc["2024-02-29", "uf"] == "RS"


def test_revisao_nao_volta_para_o_ano_anterior():
    v = f.variaveis_lspa(_levantamentos()).set_index("data_base")
    assert np.isclose(v.loc["2024-02-29", "lspa_revisao_3m_pct"], 0.0)  # janeiro compara consigo mesmo
    # levantamento de abril (usado em maio) compara com janeiro, que é max(abril − 3, janeiro)
    assert np.isclose(v.loc["2024-05-31", "lspa_revisao_3m_pct"], (1_650_000 / 2_200_000 - 1) * 100)


def test_producao_pequena_fica_vazia():
    lev = _levantamentos().assign(producao_t=1_000.0)
    v = f.variaveis_lspa(lev)
    assert v["lspa_var_producao_pct"].isna().all() and v["lspa_revisao_3m_pct"].isna().all()
