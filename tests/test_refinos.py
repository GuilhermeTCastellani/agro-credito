"""Testes das peças de validação, calibração, threshold e histórico do perfil (sem depender do modelo salvo)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from coleta import PAGINA_CMO, SERIES_CMO, ler_aba_conab, ler_cmo, link_cmo  # noqa: E402
from features import CHAVES, ano_safra_publicada, historico_perfil, variaveis_custos, variaveis_precos  # noqa: E402
from modelo import Calibrador  # noqa: E402
from treino import (FIM_TREINO_VAL2, FIM_VAL2, HP_PADRAO, INI_TESTE, INI_VAL2, ajustar_limiar,  # noqa: E402
                    carregar_sintonia, escolher_calibrador, escolher_por_fatia_transferida, f1_maximo, fatia_do_limiar,
                    limiar_da_fatia, metricas_limiar, peso_recencia, rodadas_xgb, sinalizar_por_fatia,
                    sortear_hiperparametros, beneficio_liquido, calibracao_logistica, curva_beneficio, decompor_brier,
                    escolher_por_beneficio, h_measure, intervalos_bootstrap, metricas_sinal)
from treino import caminho_checkpoint, escolher_por_f1_e_auc, melhor_por_brier, salvar_checkpoint  # noqa: E402
from trocar_criterio import trocar  # noqa: E402


def test_segunda_janela_respeita_embargo():
    seis_meses = pd.DateOffset(months=6)
    # os alvos do treino terminam antes do início da janela
    assert pd.Timestamp(FIM_TREINO_VAL2) + seis_meses <= pd.Timestamp(INI_VAL2)
    # os alvos da janela terminam antes dos primeiros alvos do teste
    assert pd.Timestamp(FIM_VAL2) + seis_meses < pd.Timestamp(INI_TESTE) + seis_meses


def test_escolha_usa_a_fatia_transferida():
    validacao = {"a": {"f1": 0.30, "f1_fatia_janela2": 0.10}, "b": {"f1": 0.16, "f1_fatia_janela2": 0.16}}
    assert escolher_por_fatia_transferida(validacao, ["a", "b"]) == "b"


def test_fatia_e_aplicada_mes_a_mes():
    # o mês 2 tem previsões bem mais altas, mas cada mês sinaliza a própria fatia
    p = np.array([0.04, 0.03, 0.02, 0.01, 0.40, 0.30, 0.20, 0.10])
    w = np.ones(8)
    datas = np.array(["m1"] * 4 + ["m2"] * 4)
    assert sinalizar_por_fatia(p, w, datas, 0.25).tolist() == [True, False, False, False, True, False, False, False]


def test_fatia_e_limiar_sao_coerentes():
    rng = np.random.default_rng(5)
    p, w = rng.uniform(0, 1, 5000), rng.uniform(0.5, 2, 5000)
    assert fatia_do_limiar(p, w, limiar_da_fatia(p, w, 0.2)) == pytest.approx(0.2, abs=0.01)


def test_safra_so_usa_ano_ja_publicado():
    datas = pd.Series(pd.to_datetime(["2024-05-31", "2024-09-30", "2024-10-31"]))
    assert ano_safra_publicada(datas).tolist() == [2022, 2022, 2023]


def test_carregar_sintonia_devolve_so_as_versoes_salvas(tmp_path):
    caminho = tmp_path / "metadados.json"
    assert carregar_sintonia(caminho, ["a"]) == {}
    caminho.write_text('{"sintonia": {"a": {"melhor": {}, "f1_validacao": 0.1}}}', encoding="utf-8")
    assert carregar_sintonia(caminho, ["a"]) == {"a": {"melhor": {}, "f1_validacao": 0.1}}
    assert list(carregar_sintonia(caminho, ["a", "b"])) == ["a"]


def test_metricas_limiar_ponderam_pela_exposicao():
    # R$ inadimplentes capturados: 0,5*100 = 50 de um total de 50 + 0,1*100 = 60
    y, w, p = np.array([0.5, 0.1]), np.array([100.0, 100.0]), np.array([0.9, 0.1])
    r = metricas_limiar(y, w, p, 0.5)
    assert r["precisao"] == pytest.approx(0.5)
    assert r["recall"] == pytest.approx(50 / 60)
    assert r["f1"] == pytest.approx(2 * 0.5 * (50 / 60) / (0.5 + 50 / 60))


def test_metricas_limiar_sem_nenhum_positivo_nao_quebra():
    r = metricas_limiar(np.array([0.5]), np.array([1.0]), np.array([0.1]), 0.9)
    assert r["precisao"] == 0 and r["recall"] == 0 and r["f1"] == 0


def test_limiar_maximiza_f1():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 0.4, 3000)
    y, w = rng.binomial(1, p).astype(float), rng.uniform(0.5, 2, 3000)
    t = ajustar_limiar(y, w, p)
    melhor = metricas_limiar(y, w, p, t)["f1"]
    assert all(melhor >= metricas_limiar(y, w, p, g)["f1"] - 1e-12 for g in np.linspace(0.001, 0.399, 200))


def test_limiar_respeita_empates():
    # as duas primeiras linhas têm a mesma probabilidade: não existe limiar que separe uma da outra
    y, w, p = np.array([1.0, 0.0, 0.0]), np.ones(3), np.array([0.5, 0.5, 0.1])
    t = ajustar_limiar(y, w, p)
    assert t == 0.5
    assert metricas_limiar(y, w, p, t)["precisao"] == pytest.approx(0.5)


@pytest.mark.parametrize("metodo", ["platt", "isotonica"])
def test_calibrador_corrige_subestimacao(metodo):
    rng = np.random.default_rng(1)
    real = rng.uniform(0.01, 0.3, 20000)
    bruto = real * 0.5  # modelo prevê metade do risco real
    y, w = rng.binomial(1, real).astype(float), np.ones_like(real)
    cal = Calibrador(metodo).fit(bruto, y, w)
    assert np.average(cal.transform(bruto)) == pytest.approx(real.mean(), abs=0.01)
    assert ((cal.transform(bruto) > 0) & (cal.transform(bruto) < 1)).all()


def test_calibrador_nenhum_nao_altera():
    p = np.array([0.1, 0.2])
    assert np.array_equal(Calibrador("nenhum").fit(p, p, np.ones(2)).transform(p), p)


def test_escolher_calibrador_prefere_nenhum_quando_ja_calibrado():
    rng = np.random.default_rng(2)
    p = rng.uniform(0.01, 0.3, 20000)
    y = rng.binomial(1, p).astype(float)
    datas = pd.Series(pd.date_range("2024-01-31", periods=6, freq="ME").repeat(4000)[:20000])
    metodo, diag = escolher_calibrador(p, y, np.ones_like(p), datas)
    assert metodo in diag
    assert diag[metodo] == min(diag.values())


def test_peso_recencia_cai_pela_metade_a_cada_meia_vida():
    datas = pd.Series(pd.to_datetime(["2024-06-30", "2023-06-30"]))
    assert peso_recencia(datas, 12) == pytest.approx([1.0, 0.5], abs=0.01)


def _painel():
    meses = pd.date_range("2024-01-31", periods=8, freq="ME")
    d = pd.DataFrame({"data_base": meses, "carteira_ativa": 100.0,
                      "carteira_inadimplencia": np.arange(8, dtype=float),
                      "vencido_de_15_ate_90_dias": 2.0, "ativo_problematico": 4.0, "a_vencer_ate_90_dias": 10.0})
    for c in CHAVES:
        d[c] = "x"
    return d


def test_historico_perfil_so_olha_para_tras():
    h = historico_perfil(_painel()).set_index("data_base")
    fim = pd.Timestamp("2024-08-31")
    assert h.loc[fim, "perfil_inad_atual_pct"] == pytest.approx(7)  # valor do próprio mês
    assert h.loc[fim, "perfil_inad_var_6m_pp"] == pytest.approx(6)  # 7 hoje contra 1 em fev/2024
    assert h.loc[fim, "perfil_atraso_15_90_pct"] == pytest.approx(2)
    assert np.isnan(h.loc[pd.Timestamp("2024-06-30"), "perfil_inad_var_6m_pp"])  # não há 6 meses antes
    assert h.loc[fim, "perfil_inad_var_1m_pp"] == pytest.approx(1)
    assert h.loc[fim, "perfil_inad_var_3m_pp"] == pytest.approx(3)
    assert h.loc[fim, "perfil_problematico_pct"] == pytest.approx(4)
    assert h.loc[fim, "perfil_vence_90d_pct"] == pytest.approx(10)
    assert h.loc[fim, "perfil_cresc_6m_pct"] == pytest.approx(0)
    assert h.loc[fim, "pares_inad_atual_pct"] == pytest.approx(7)
    assert h.loc[fim, "mes_alvo"] == 2  # 6 meses depois de agosto
    assert np.isnan(h.loc[fim, "perfil_inad_var_12m_pp"])  # não há 12 meses antes


def test_historico_perfil_nao_multiplica_linhas():
    d = _painel()
    assert len(historico_perfil(d)) == len(d)


def test_f1_maximo_nao_muda_com_transformacao_monotonica():
    rng = np.random.default_rng(3)
    p = rng.uniform(0.01, 0.5, 2000)
    y, w = rng.binomial(1, p).astype(float), rng.uniform(0.5, 2, 2000)
    assert f1_maximo(y, p, w) == pytest.approx(f1_maximo(y, np.log(p / (1 - p)), w))
    assert f1_maximo(y, p, w) == pytest.approx(metricas_limiar(y, w, p, ajustar_limiar(y, w, p))["f1"])


def test_sorteio_de_hiperparametros_e_reprodutivel_e_inclui_o_padrao():
    a, b = sortear_hiperparametros(5), sortear_hiperparametros(5)
    assert a == b and a[0] == HP_PADRAO and len(a) == 5
    assert all(3 <= h["max_depth"] <= 9 and 0.5 <= h["subsample"] <= 1 for h in a)


def test_parada_por_f1_funciona_com_e_sem_offset():
    rng = np.random.default_rng(4)
    meses = pd.date_range("2023-01-31", periods=12, freq="ME")
    d = pd.DataFrame({"data_base": np.repeat(meses, 100)})
    for c in ["uf", "regiao", "segmento", "cliente", "ocupacao_atividade", "porte", "submodalidade", "origem", "indexador"]:
        d[c] = "x"
    d["prazo_remanescente_anos"] = rng.uniform(0, 5, len(d))
    d["valor_medio_operacao_mil"] = rng.uniform(1, 100, len(d))
    d["uf_inad_atual_pct"] = rng.uniform(1, 10, len(d))
    d["alvo_inad"] = np.clip(d["prazo_remanescente_anos"] / 10 + rng.normal(0, 0.05, len(d)), 0, 1)
    d["peso"] = 1.0
    numericas = ["prazo_remanescente_anos", "valor_medio_operacao_mil", "uf_inad_atual_pct"]
    assert rodadas_xgb(d, numericas, False, parar_por_f1=True) >= 1
    assert rodadas_xgb(d, numericas, True, parar_por_f1=True) >= 1


def _precos_base():
    meses = pd.date_range("2021-01-31", "2024-12-31", freq="ME")
    p = pd.DataFrame({"mes": meses})
    for serie in SERIES_CMO.values():
        p[serie] = 100.0
    macro = pd.DataFrame({"mes": meses, "dolar": 1.0})
    return p, macro


def _safra_rs():
    linhas = [("43", 2022, "soja", 100.0), ("43", 2022, "milho", 100.0), ("43", 2023, "soja", 300.0)]
    return pd.DataFrame([{"cod_uf": c, "ano": a, "produto": prod, "var": "producao", "fonte": "PAM", "V": v}
                         for c, a, prod, v in linhas])


def _linha_mes(tab, data):
    return tab[tab["data_base"] == pd.Timestamp(data)].iloc[0]


def test_cesta_usa_os_pesos_do_ultimo_pam_publicado():
    p, macro = _precos_base()
    p.loc[p["mes"] >= "2024-01-31", "milho"] = 200.0
    tab = variaveis_precos(_safra_rs(), p, macro)
    # em maio/2024 vale o PAM 2022 (metade soja, metade milho): milho dobrou em 12 meses, soja ficou igual
    assert _linha_mes(tab, "2024-05-31")["preco_cesta_var_12m_pct"] == pytest.approx(50)
    # em outubro/2024 já vale o PAM 2023, só com soja
    assert _linha_mes(tab, "2024-10-31")["preco_cesta_var_12m_pct"] == pytest.approx(0)


def test_precos_nao_olham_para_o_futuro():
    p, macro = _precos_base()
    antes = _linha_mes(variaveis_precos(_safra_rs(), p, macro), "2024-05-31")
    p.loc[p["mes"] > "2024-05-31", ["soja", "milho", "ureia", "boi"]] = 900.0
    depois = _linha_mes(variaveis_precos(_safra_rs(), p, macro), "2024-05-31")
    pd.testing.assert_series_equal(antes, depois)


def test_precos_sao_convertidos_para_reais_e_calculam_relacao_de_troca():
    p, macro = _precos_base()
    macro.loc[macro["mes"] >= "2024-01-31", "dolar"] = 2.0  # dólar dobra, preços em dólar não mudam
    p.loc[p["mes"] >= "2024-01-31", ["ureia", "dap", "potassio"]] = 200.0  # fertilizante quadruplica em reais
    linha = _linha_mes(variaveis_precos(_safra_rs(), p, macro), "2024-05-31")
    assert linha["preco_cesta_var_12m_pct"] == pytest.approx(100)
    assert linha["fert_var_12m_pct"] == pytest.approx(300)
    assert linha["boi_var_12m_pct"] == pytest.approx(100)
    assert linha["relacao_troca_var_12m_pct"] == pytest.approx(-50)


def test_link_do_banco_mundial_e_lido_da_pagina():
    url = "https://thedocs.worldbank.org/en/doc/abc-001/related/CMO-Historical-Data-Monthly.xlsx"
    assert link_cmo(f'<a href="{url}">Monthly</a>') == url
    with pytest.raises(RuntimeError, match=PAGINA_CMO):
        link_cmo("<a href='outro.xlsx'>nada</a>")


def test_ler_cmo_trata_series_marcadas_e_valores_ausentes(tmp_path):
    from openpyxl import Workbook
    livro = Workbook()
    aba = livro.active
    aba.title = "Monthly Prices"
    for _ in range(4):
        aba.append([None])
    nomes = list(SERIES_CMO)
    aba.append([None] + [n + " **" if n == "Beef" else n + " " for n in nomes])
    aba.append([None] + ["($/mt)"] * len(nomes))
    aba.append(["2026M09"] + [10.0] * len(nomes))
    aba.append(["2026M10"] + ["…"] + [20.0] * (len(nomes) - 1))
    caminho = tmp_path / "cmo.xlsx"
    livro.save(caminho)
    t = ler_cmo(caminho)
    assert t["mes"].tolist() == [pd.Timestamp("2026-09-30"), pd.Timestamp("2026-10-31")]
    assert t["boi"].tolist() == [10.0, 20.0]
    assert np.isnan(t["soja"].iloc[1])


def _safra_soja():
    return pd.DataFrame([{"cod_uf": "43", "ano": ano, "produto": "soja", "var": "producao", "fonte": "PAM", "V": 100.0}
                         for ano in range(2019, 2025)])


def _custos_rs():
    return pd.DataFrame([
        {"produto": "soja", "local": "Cruz Alta", "uf": "RS", "aba": "a", "data_ref": pd.Timestamp("2022-03-31"),
         "produtividade_kg_ha": 3000.0, "custo_total_ha": 2500.0},
        {"produto": "soja", "local": "Cruz Alta", "uf": "RS", "aba": "b", "data_ref": pd.Timestamp("2023-03-31"),
         "produtividade_kg_ha": 3000.0, "custo_total_ha": 5000.0},
    ])


def test_custo_so_entra_depois_da_defasagem_e_calcula_margem():
    p, macro = _precos_base()
    tab = variaveis_custos(_custos_rs(), _safra_soja(), p, macro)
    # custo por saca: 2500 / (3000 / 60) = 50; preço por saca: 100 x 0,06 = 6, então margem de -88%
    assert _linha_mes(tab, "2023-05-31")["conab_margem_pct"] == pytest.approx(6 / 50 * 100 - 100)
    # o custo de março/2023 só vale a partir de junho/2023
    assert _linha_mes(tab, "2023-06-30")["conab_margem_pct"] == pytest.approx(6 / 100 * 100 - 100)
    assert _linha_mes(tab, "2023-06-30")["conab_custo_var_anual_pct"] == pytest.approx(100)


def test_custo_nao_olha_para_o_futuro_e_expira():
    p, macro = _precos_base()
    tab = variaveis_custos(_custos_rs(), _safra_soja(), p, macro)
    assert tab[tab["data_base"] < "2022-06-30"].empty
    assert tab["data_base"].max() <= pd.Timestamp("2025-07-31")
    so_primeiro = variaveis_custos(_custos_rs().iloc[:1], _safra_soja(), p, macro)
    assert _linha_mes(so_primeiro, "2023-05-31")["conab_margem_pct"] == _linha_mes(tab, "2023-05-31")["conab_margem_pct"]


def test_ler_aba_conab_entende_os_dois_formatos():
    antigo = pd.DataFrame([["Produtividade media:", 2880, "kg/ha"], [None, "A PREÇOS DE:", pd.Timestamp("2021-03-15")],
                           ["CUSTO TOTAL (H+I = J)", 1043.25, 21.73]])
    novo = pd.DataFrame([["Mês/Ano: Março/2026", None], ["Produtividade Média: 2880,00 kg/ha", "Ex-Ant"],
                         ["Custo Total (D)", 6329.44]])
    assert ler_aba_conab(antigo) == {"data_ref": pd.Timestamp("2021-03-31"), "produtividade_kg_ha": 2880.0, "custo_total_ha": 1043.25}
    assert ler_aba_conab(novo) == {"data_ref": pd.Timestamp("2026-03-31"), "produtividade_kg_ha": 2880.0, "custo_total_ha": 6329.44}


def test_mcc_e_beneficio_liquido_em_reais():
    y, w = np.array([1.0, 0.0, 1.0, 0.0]), np.ones(4)
    sinal = np.array([True, False, True, False])
    assert metricas_sinal(y, w, sinal)["mcc"] == pytest.approx(1)
    assert metricas_sinal(y, w, ~sinal)["mcc"] == pytest.approx(-1)
    # acerta 2 reais e erra 1 em 4; com 2 alarmes por acerto: (2 - 1 / 2) / 4
    assert beneficio_liquido(y, w, np.array([True, True, True, False]), 2) == pytest.approx(0.375)


def test_curva_de_beneficio_usa_o_limiar_equivalente_ao_custo():
    linhas = curva_beneficio(np.array([1.0, 0.0]), np.ones(2), np.array([0.5, 0.01]))
    assert [round(x["limiar_pct"], 2) for x in linhas] == [25, 16.67, 9.09, 4.76]
    assert all(x["modelo"] == pytest.approx(0.5) for x in linhas)
    assert linhas[0]["todos"] == pytest.approx((1 - 1 / 3) / 2)


def test_h_measure_vai_de_zero_a_um():
    rng = np.random.default_rng(6)
    y = rng.binomial(1, 0.2, 5000).astype(float)
    w = np.ones_like(y)
    assert h_measure(y, w, y * 0.8 + 0.1) == pytest.approx(1, abs=1e-6)
    assert abs(h_measure(y, w, rng.uniform(0, 1, 5000))) < 0.03


def test_calibracao_logistica_reconhece_superestimacao():
    rng = np.random.default_rng(7)
    p = rng.uniform(0.02, 0.5, 30000)
    y, w = rng.binomial(1, p).astype(float), np.ones_like(p)
    bem = calibracao_logistica(y, w, p)
    assert bem["calibracao_intercepto"] == pytest.approx(0, abs=0.05)
    assert bem["calibracao_inclinacao"] == pytest.approx(1, abs=0.05)
    assert calibracao_logistica(y, w, np.clip(p * 2, 0, 0.99))["calibracao_intercepto"] < -0.3


def test_decomposicao_do_brier_fecha_e_mede_habilidade():
    rng = np.random.default_rng(8)
    p = rng.uniform(0.02, 0.5, 30000)
    y, w = rng.binomial(1, p).astype(float), np.ones_like(p)
    d = decompor_brier(y, w, p)
    brier = np.mean((p - y) ** 2)
    assert d["brier_confiabilidade"] - d["brier_resolucao"] + d["brier_incerteza"] == pytest.approx(brier, abs=0.002)
    assert d["brier_skill"] == pytest.approx(1 - brier / d["brier_incerteza"])
    assert d["ece"] < 0.02
    media = np.full_like(p, y.mean())
    assert decompor_brier(y, w, media)["brier_skill"] == pytest.approx(0, abs=1e-9)


def test_criterio_de_beneficio_escolhe_o_maior_na_janela_2():
    validacao = {"a": {"f1_fatia_janela2": 0.3, "beneficio_medio_janela2": 0.01}, "b": {"f1_fatia_janela2": 0.1, "beneficio_medio_janela2": 0.02}}
    assert escolher_por_beneficio(validacao, ["a", "b"]) == "b"
    assert escolher_por_fatia_transferida(validacao, ["a", "b"]) == "a"


def test_intervalos_contem_o_valor_e_a_diferenca_para_si_e_zero():
    rng = np.random.default_rng(9)
    n = 4000
    p = rng.uniform(0.02, 0.5, n)
    y, w = rng.binomial(1, p).astype(float), np.ones(n)
    datas = np.repeat(pd.date_range("2024-01-31", periods=4, freq="ME"), n // 4)
    ufs = np.tile(np.array(["RS", "MT", "PR", "GO", "BA"]), n // 5)
    r = intervalos_bootstrap(y, w, datas, ufs, {"m": (p, p, 0.1)}, "m", n=20)["m"]
    assert r["auc"]["baixo"] <= r["auc"]["alto"]
    assert r["auc"]["dif_baixo"] == 0 and r["auc"]["dif_alto"] == 0


def test_trocar_criterio_copia_modelo_e_metadados(tmp_path):
    for criterio in ("f1", "beneficio"):
        pasta = tmp_path / "por_criterio" / criterio
        pasta.mkdir(parents=True)
        (pasta / "modelo_final.joblib").write_text(criterio)
        (pasta / "metadados.json").write_text(criterio)
    trocar("f1", tmp_path)
    assert (tmp_path / "modelo_final.joblib").read_text() == "f1"
    assert (tmp_path / "metadados.json").read_text() == "f1"
    trocar("beneficio", tmp_path)
    assert (tmp_path / "metadados.json").read_text() == "beneficio"
    with pytest.raises(ValueError):
        trocar("auc", tmp_path)


def test_checkpoint_depende_da_base_e_guarda_o_conteudo(tmp_path):
    import joblib
    a = caminho_checkpoint(tmp_path, "modelo", "base-1")
    assert a != caminho_checkpoint(tmp_path, "modelo", "base-2")
    assert a != caminho_checkpoint(tmp_path, "outro", "base-1")
    salvar_checkpoint(a, {"f1": 0.3})
    assert joblib.load(a) == {"f1": 0.3}
    assert list(tmp_path.glob("*.tmp")) == []


def test_escolha_usa_f1_e_auc_juntos():
    validacao = {"so_f1": {"f1_fatia_janela2": 0.30, "auc_janela2": 0.60},
                 "so_auc": {"f1_fatia_janela2": 0.10, "auc_janela2": 0.90},
                 "equilibrado": {"f1_fatia_janela2": 0.25, "auc_janela2": 0.85}}
    assert escolher_por_f1_e_auc(validacao, validacao) == "equilibrado"


def test_empate_em_f1_e_auc_favorece_o_maior_f1():
    validacao = {"a": {"f1_fatia_janela2": 0.30, "auc_janela2": 0.70}, "b": {"f1_fatia_janela2": 0.20, "auc_janela2": 0.80}}
    assert escolher_por_f1_e_auc(validacao, validacao) == "a"


def test_busca_escolhe_o_menor_brier():
    tentativas = [{"hp": "a", "brier_validacao": 0.02}, {"hp": "b", "brier_validacao": 0.01}]
    assert melhor_por_brier(tentativas)["hp"] == "b"
