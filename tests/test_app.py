"""Teste de fumaça do dashboard e do modelo salvo.  Rodar: python -m pytest tests -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from modelo import carregar_modelo  # noqa: E402


def _linha(**kw):
    base = dict(uf="RS", regiao="Sul", segmento="Cooperativa", cliente="PF", ocupacao_atividade="Autônomo",
                porte="Mais de 5 a 10 salários mínimos", submodalidade="Custeio", origem="Com destinação específica",
                indexador="Prefixado", prazo_remanescente_anos=1.0, valor_medio_operacao_mil=120.0,
                selic=14.0, ipca_12m=4.2, dolar_var12m=-5.0, safra_anomalia_pct=0.0, safra_perda_area_pct=1.0,
                safra_var_producao_pct=0.0, uf_inad_atual_pct=5.0, uf_atraso_15_90_pct=1.5, uf_inad_var_6m_pp=0.0,
                perfil_inad_atual_pct=3.0, perfil_atraso_15_90_pct=1.0, perfil_inad_var_6m_pp=0.0,
                perfil_inad_var_1m_pp=0.0, perfil_inad_var_3m_pp=0.0, perfil_inad_var_12m_pp=0.0,
                perfil_problematico_pct=4.0, perfil_vence_90d_pct=10.0, perfil_cresc_6m_pct=0.0,
                perfil_cresc_12m_pct=0.0, pares_inad_atual_pct=3.0, uf_finalidade_inad_pct=3.0, mes_alvo=2,
                preco_cesta_var_6m_pct=0.0, preco_cesta_var_12m_pct=0.0, fert_var_6m_pct=0.0, fert_var_12m_pct=0.0,
                boi_var_12m_pct=0.0, relacao_troca_var_12m_pct=0.0, conab_margem_pct=10.0, conab_custo_var_anual_pct=0.0)
    base.update(kw)
    return pd.DataFrame([base])


def test_modelo_prob_valida_e_shap_soma():
    m = carregar_modelo(RAIZ)
    x = _linha()
    p = m.prever(x)[0]
    assert 0 < p < 1
    sv = m.explicar(x).iloc[0]
    # SHAP + valor esperado deve reproduzir a log-chance prevista
    # no ensemble, a parte do Random Forest é aproximada, então a soma fecha só dentro do erro medido na instalação
    tolerancia = max(1e-3, 2 * getattr(m, "erro_aditividade", 0) or 0) if getattr(m, "tipo", "") == "ensemble" else 1e-3
    if m.escala_log_chance:
        assert abs((m.valor_base + sv.sum()) - np.log(p / (1 - p))) < tolerancia
    else:
        assert abs((m.valor_base + sv.sum()) - p) < 1e-3


def test_modelo_salvo_tem_calibrador_e_limiar():
    m = carregar_modelo(RAIZ)
    assert m.calibrador is not None
    assert 0 < m.limiar < 1
    assert 0 < m.fatia_pct < 100


def test_regiao_com_mais_inadimplencia_aumenta_risco():
    m = carregar_modelo(RAIZ)
    baixo = m.prever(_linha(uf_inad_atual_pct=2.0))[0]
    alto = m.prever(_linha(uf_inad_atual_pct=10.0))[0]
    assert alto > baixo


def test_dashboard_renderiza_sem_erro():
    at = AppTest.from_file(str(RAIZ / "app.py"), default_timeout=120).run()
    assert not at.exception
    assert len(at.tabs) == 5
    textos = " ".join(md.value for md in at.markdown)
    assert "Probabilidade de inadimplência" in textos


def test_dashboard_troca_para_pj():
    at = AppTest.from_file(str(RAIZ / "app.py"), default_timeout=120).run()
    at.radio[0].set_value("PJ").run()
    assert not at.exception


def test_dashboard_mostra_novas_metricas():
    at = AppTest.from_file(str(RAIZ / "app.py"), default_timeout=120).run()
    assert not at.exception
    colunas = [c for df in at.dataframe for c in df.value.columns]
    for nome in ["Precisão", "Recall", "F1", "Brier", "MCC", "Benefício líquido", "Brier Skill", "AUPRC", "H-measure"]:
        assert nome in colunas


def test_aba_de_segmentos_mostra_a_lista_e_a_medicao():
    at = AppTest.from_file(str(RAIZ / "app.py"), default_timeout=180).run()
    assert not at.exception
    colunas = [c for df in at.dataframe for c in df.value.columns]
    for nome in ["Prevista em 6 meses (%)", "Crítico", "F1 repetindo o nível atual"]:
        assert nome in colunas
    radio = [r for r in at.radio if r.label == "Limite de inadimplência"][0]
    radio.set_value(0.2).run()
    assert not at.exception
