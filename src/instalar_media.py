"""
Troca o modelo de produção do ensemble empilhado para a média simples das log-chances dos quatro modelos.

Decidido em 2026-11-10, depois do backtest trimestral (src/backtest.py): a média teve o menor erro de nível (1,14 p.p. contra
1,75 do empilhado fixo), com PR-AUC igual e F1 >= 10% 0,008 abaixo, dentro da tolerância de 0,01 fixada antes.

Pré-requisito: python src/comparar_media.py, que grava modelos/ensemble/media/ensemble.joblib (pesos 0,25, intercepto 0,
sem calibrador, fatia e limiar) e comparacao.json (importância SHAP).

1. Backup de modelos/ensemble/final/ensemble.joblib e de modelos/metadados.json em modelos/backup_antes_media/.
2. Copia o resumo da média por cima do de produção (os componentes são os mesmos).
3. Mede a média no teste como os outros modelos (calibração e fatia da validação 1) e reescreve metadados.json.
Depois rode python src/segmentos.py. Para voltar: copie os dois arquivos do backup de volta.
"""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np

import intervalos_colab as ic
import otimizacao as o
from modelo import Calibrador, ModeloEnsemble, logit
from treino import (curva_beneficio, curva_limiar, distribuicao_por_resultado, escolher_calibrador,
                    intervalos_bootstrap, tabela_calibracao)

RAIZ = Path(__file__).resolve().parents[1]
MODELOS = RAIZ / "modelos"
FINAL = MODELOS / "ensemble" / "final"
MEDIA = MODELOS / "ensemble" / "media"
BACKUP = MODELOS / "backup_antes_media"
NOME = "Ensemble média (XGBoost, CatBoost, Random Forest e LightGBM)"
EMPILHADO = "Ensemble empilhado (XGBoost, CatBoost, Random Forest e LightGBM)"
JANELAS = ("val1", "val2", "teste")


def main():
    t0 = time.time()
    BACKUP.mkdir(exist_ok=True)
    for origem in (FINAL / "ensemble.joblib", MODELOS / "metadados.json"):
        if not (BACKUP / origem.name).exists():
            shutil.copy2(origem, BACKUP / origem.name)
    shutil.copy2(MEDIA / "ensemble.joblib", FINAL / "ensemble.joblib")
    modelo = ModeloEnsemble.carregar(FINAL)
    assert modelo.intercepto == 0 and set(modelo.coeficientes.values()) == {0.25} and modelo.calibrador is None
    comp = json.loads((MEDIA / "comparacao.json").read_text(encoding="utf-8"))["candidato_media_logit"]

    janelas = o.carregar_janelas(RAIZ / "dados" / "processados" / "base_modelagem.parquet")
    prev = ic.carregar_previsoes(janelas)
    p = {j: 1 / (1 + np.exp(-np.mean([logit(prev[n][j]) for n in prev], axis=0))) for j in JANELAS}
    empilhado = o.combinar_ensembles(prev, janelas)["previsoes"]["ensemble_empilhado"]
    del prev
    te, v1 = janelas["teste"], janelas["val1"]
    y, w = te["alvo_inad"].to_numpy(), te["peso"].to_numpy()
    y1, w1 = v1["alvo_inad"].to_numpy(), v1["peso"].to_numpy()

    def extras(previsto):
        metodo, _ = escolher_calibrador(previsto["val1"], y1, w1, v1["data_base"])
        return previsto["teste"], Calibrador(metodo).fit(previsto["val1"], y1, w1).transform(previsto["teste"])

    resultado = o.avaliar(p, janelas)
    bruto, pc = extras(p)
    bruto_e, pc_e = extras(empilhado)
    meta = json.loads((MODELOS / "metadados.json").read_text(encoding="utf-8"))
    fatia_e = meta["comparacao"][EMPILHADO]["fatia_pct"] / 100
    intervalos = intervalos_bootstrap(y, w, te["data_base"], te["uf"],
                                      {NOME: (bruto, pc, resultado["fatia_pct"] / 100), EMPILHADO: (bruto_e, pc_e, fatia_e)}, NOME)
    print(f"teste medido ({time.time() - t0:.0f}s): PR-AUC {resultado['pr_auc']:.4f}, AUC {resultado['auc']:.4f}, "
          f"F1 {resultado['f1']:.3f}, Brier {resultado['brier']:.5f}", flush=True)

    meta |= {
        "modelo_escolhido": NOME, "formato": "ensemble", "tipo": "ensemble",
        "criterio_escolha": "backtest_erro_de_nivel",
        "producao": {"combinacao": "media_logit", "calibracao": "nenhuma", "limiar_pct": comp["limiar_pct"],
                     "fatia_pct": comp["fatia_pct"], "janela_de": "2025-09-01", "janela_ate": "2026-02-28",
                     "nota": "média das log-chances com peso 0,25 e intercepto 0, sem calibrador; a fatia vem das previsões "
                             "fora da amostra da janela de teste nos 6 meses mais recentes",
                     "anterior": meta["producao"]},
        "importancia_shap_global": comp["importancia_shap_global"],
        "comparacao": {**meta["comparacao"], NOME: resultado},
        "intervalos_teste": {**meta["intervalos_teste"], **intervalos},
    }
    meta["calibracao_teste"][NOME] = tabela_calibracao(y, w, pc)
    meta["calibracao_teste_bruta"][NOME] = tabela_calibracao(y, w, bruto)
    meta["curva_limiar_teste"][NOME] = curva_limiar(y, w, pc)
    meta["beneficio_teste"][NOME] = curva_beneficio(y, w, pc)
    meta["distribuicao_teste"][NOME] = distribuicao_por_resultado(y, w, pc)
    bt = RAIZ / "modelos" / "backtest" / "resultados.json"
    if bt.exists():
        meta["backtest"] = json.loads(bt.read_text(encoding="utf-8"))
    (MODELOS / "metadados.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"pronto em {time.time() - t0:.0f}s: {NOME}; fatia {comp['fatia_pct']:.1f}%, limiar {comp['limiar_pct']:.2f}%")


if __name__ == "__main__":
    main()
