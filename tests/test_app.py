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
    assert "Inadimplência prevista do perfil" in textos


def test_dashboard_troca_para_pj():
    at = AppTest.from_file(str(RAIZ / "app.py"), default_timeout=120).run()
    [r for r in at.radio if r.label == "Tipo de tomador"][0].set_value("PJ").run()
    assert not at.exception


def test_dashboard_mostra_novas_metricas():
    at = AppTest.from_file(str(RAIZ / "app.py"), default_timeout=120).run()
    assert not at.exception
    colunas = [c for df in at.dataframe for c in df.value.columns]
    for nome in ["Precisão", "Recall", "F1", "Brier", "MCC", "Benefício líquido", "Brier Skill", "AUPRC", "H-measure"]:
        assert nome in colunas


def test_aba_de_perfis_criticos_mostra_a_lista_e_a_medicao():
    at = AppTest.from_file(str(RAIZ / "app.py"), default_timeout=180).run()
    assert not at.exception
    assert at.tabs[0].label == "Perfis em alerta"
    colunas = [c for df in at.dataframe for c in df.value.columns]
    for nome in ["Prevista em 6 meses (%)", "Valor em risco (R$ mi)", "Alerta", "Sinalização", "F1 repetindo o nível atual",
                 "Limite"]:
        assert nome in colunas
    rotulos = [m.label for m in at.metric]
    for nome in ["Perfis críticos", "Perfis em atenção"]:
        assert nome in rotulos
    radio = [r for r in at.radio if r.label == "Mostrar"][0]
    radio.set_value("Críticos e em atenção").run()
    assert not at.exception
    alertas = {v for df in at.dataframe if "Alerta" in df.value.columns and "Valor em risco (R$ mi)" in df.value.columns
               for v in df.value["Alerta"]}
    assert alertas <= {"crítico", "atenção"}


def test_acompanhamento_previsto_real():
    import json
    from segmentos import acompanhamento
    df = pd.DataFrame({"data_base": pd.to_datetime(["2024-01-31", "2024-01-31", "2024-02-29"]), "uf": ["RS", "SP", "RS"],
                       "submodalidade": ["Custeio"] * 3, "peso": [1.0, 3.0, 2.0], "janela": ["val1"] * 3,
                       "real": [0.10, 0.02, 0.05], "previsto": [0.08, 0.04, 0.05], "atual": [0.05, 0.01, 0.04]})
    s = pd.DataFrame(acompanhamento(df))
    br_jul = s[(s["uf"] == "BR") & (s["mes_alvo"] == "2024-07")].iloc[0]
    assert abs(br_jul["real_pct"] - 4.0) < 1e-9 and abs(br_jul["previsto_pct"] - 5.0) < 1e-9
    assert set(s.loc[s["uf"] == "RS", "mes_alvo"]) == {"2024-07", "2024-08"}
    meta = json.loads((RAIZ / "modelos" / "metadados.json").read_text(encoding="utf-8"))
    assert {r["janela"] for r in meta["acompanhamento"]["linhas"]} == {"val1", "val2", "teste"}
    proj = [r for r in meta["projecao"]["linhas"] if r["uf"] == "BR"]
    assert len(proj) == 6 and all(r["real_pct"] is None for r in proj)
    assert proj[0]["mes_alvo"] > max(r["mes_alvo"] for r in meta["acompanhamento"]["linhas"])


def test_media_candidata_e_a_media_das_log_chances():
    from modelo import ModeloEnsemble, logit
    m = carregar_modelo(RAIZ)
    media = ModeloEnsemble(m.componentes, {n: 0.25 for n in m.componentes}, 0.0)
    x = _linha()
    esperado = 1 / (1 + np.exp(-np.mean([logit(c.prever(x)) for c in m.componentes.values()], axis=0)))
    assert abs(media.prever(x)[0] - esperado[0]) < 1e-9
    resumo = RAIZ / "modelos" / "ensemble" / "media" / "ensemble.joblib"
    if resumo.exists():
        import joblib
        info = joblib.load(resumo)  # arquivo gerado por este projeto
        assert info["intercepto"] == 0.0 and set(info["coeficientes"].values()) == {0.25}
        assert 0 < info["limiar"] < 1 and 0 < info["fatia_pct"] < 100


def test_limite_principal_e_um_dos_medidos():
    from segmentos import LIMITE_ATENCAO, LIMITE_CRITICO, LIMITE_PRINCIPAL, LIMITES
    assert LIMITE_PRINCIPAL == LIMITE_CRITICO
    assert LIMITE_ATENCAO in LIMITES and LIMITE_CRITICO in LIMITES


def test_nivel_alerta_nos_limites():
    from segmentos import nivel_alerta
    assert list(nivel_alerta([0.0, 0.0499, 0.05, 0.0999, 0.10, 0.5])) == [
        "normal", "normal", "atenção", "atenção", "crítico", "crítico"]


def test_limite_principal_e_medido_no_teste():
    import json
    from segmentos import LIMITE_PRINCIPAL, LIMITES
    assert LIMITE_PRINCIPAL in LIMITES
    meta = json.loads((RAIZ / "modelos" / "metadados.json").read_text(encoding="utf-8"))
    linhas = [r for r in meta["segmentos_teste"]["linhas"]
              if r["nivel"] == "perfil" and abs(r["limite"] - LIMITE_PRINCIPAL) < 1e-9]
    assert {r["pesa"] for r in linhas} == {"reais", "contagem"}
