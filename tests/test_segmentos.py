"""Segmentos críticos: agregação ponderada e medição contra as regras simples."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from segmentos import agregar, avaliar, medir  # noqa: E402


def test_agregar_pondera_pela_exposicao():
    df = pd.DataFrame({"data_base": ["m"] * 2, "uf": ["RS"] * 2, "submodalidade": ["Custeio"] * 2,
                       "peso": [1.0, 3.0], "previsto": [0.2, 0.0], "atual": [0.1, 0.1], "real": [0.0, 0.4]})
    g = agregar(df).iloc[0]
    assert g["previsto"] == pytest.approx(0.05)
    assert g["real"] == pytest.approx(0.3)
    assert (g["peso"], g["perfis"]) == (4.0, 2)


def test_medir_precisao_recall_e_f1():
    r = medir([True, True, False, False], [True, False, True, False], [1, 1, 1, 1])
    assert (r["precisao"], r["recall"], r["f1"]) == (0.5, 0.5, 0.5)
    assert r["prevalencia"] == 0.5 and r["sinalizados"] == 0.5
    assert medir([True], [False], [1])["f1"] == 0.0


def test_avaliar_cobre_niveis_limites_e_pesos_e_premia_quem_acerta():
    rng = np.random.default_rng(0)
    n = 600
    real = rng.uniform(0, 0.3, n)
    df = pd.DataFrame({"data_base": np.repeat(["a", "b"], n // 2), "uf": rng.choice(["RS", "MT", "PR"], n),
                       "submodalidade": rng.choice(["Custeio", "Investimento"], n), "peso": rng.uniform(0.5, 2, n),
                       "real": real, "previsto": real, "atual": rng.uniform(0, 0.3, n)})
    linhas = avaliar(df)
    assert len(linhas) == 2 * 3 * 2
    for r in linhas:
        assert r["modelo"]["f1"] >= r["nivel_atual"]["f1"]  # quem prevê o real exato nunca perde para a regra simples
        if r["nivel"] == "perfil":
            assert r["modelo"]["f1"] == pytest.approx(1.0)
