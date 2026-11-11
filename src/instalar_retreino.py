"""
Instala o retreino do Colab (notebooks/retreino_colab.ipynb): SCR até set/2026 e a decisão sobre a LSPA.

Uso: python src/instalar_retreino.py [caminho do retreino.zip]   (padrão: Downloads)

1. Extrai o zip em modelos/retreino/ e lê decisao.json (com ou sem LSPA, pela regra fixada antes).
2. Backup de modelos/ensemble/final, metadados.json e modelos/colab/previsoes_*.npz em modelos/backup_antes_retreino/.
3. Copia as previsões das janelas para modelos/colab/ e os modelos finais para modelos/ensemble/final/.
4. Calcula o valor de partida e a importância do SHAP numa amostra dos perfis atuais.
5. Mede no teste (agora até mar/2026) os quatro modelos, a média e o empilhado, e reescreve metadados.json.
Depois: python src/segmentos.py. Para voltar: copie o backup de volta.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import intervalos_colab as ic
import otimizacao as o
from modelo import Calibrador, ModeloEnsemble, logit
from treino import (SEED, curva_beneficio, curva_limiar, distribuicao_por_resultado, escolher_calibrador,
                    intervalos_bootstrap, tabela_calibracao)

RAIZ = Path(__file__).resolve().parents[1]
MODELOS = RAIZ / "modelos"
DESTINO = MODELOS / "retreino"
FINAL = MODELOS / "ensemble" / "final"
BACKUP = MODELOS / "backup_antes_retreino"
JANELAS = ("val1", "val2", "teste")
NOMES = {"xgboost": "XGBoost (Optuna, PR-AUC)", "catboost": "CatBoost (Optuna, PR-AUC)",
         "random_forest": "Random Forest (Optuna, PR-AUC)", "lightgbm": "LightGBM (Optuna, PR-AUC)",
         "ensemble_empilhado": "Ensemble empilhado (XGBoost, CatBoost, Random Forest e LightGBM)",
         "media": "Ensemble média (XGBoost, CatBoost, Random Forest e LightGBM)"}


def extrair(zip_caminho: Path) -> dict:
    if DESTINO.exists():
        shutil.rmtree(DESTINO)
    DESTINO.mkdir(parents=True)
    with zipfile.ZipFile(zip_caminho) as z:
        for info in z.infolist():
            alvo = (DESTINO / info.filename).resolve()
            if not str(alvo).startswith(str(DESTINO.resolve())):  # não aceita caminhos para fora da pasta
                raise ValueError(f"caminho inválido no zip: {info.filename}")
        z.extractall(DESTINO)
    return json.loads((DESTINO / "decisao.json").read_text(encoding="utf-8"))


def main(zip_caminho: Path, n_amostra: int = 300):
    t0 = time.time()
    decisao = extrair(zip_caminho)
    variante = "com_lspa" if decisao["usar_lspa"] else "sem_lspa"
    origem = DESTINO / variante
    print(f"decisão: {variante} | sem LSPA PR-AUC {decisao['sem']['pr_auc']:.4f}, com LSPA {decisao['com']['pr_auc']:.4f}", flush=True)

    if not BACKUP.exists():
        BACKUP.mkdir()
        shutil.copytree(FINAL, BACKUP / "final")
        shutil.copy2(MODELOS / "metadados.json", BACKUP / "metadados.json")
        (BACKUP / "colab").mkdir()
        for arq in ic.PASTA.glob("previsoes_*.npz"):
            shutil.copy2(arq, BACKUP / "colab" / arq.name)
    for arq in origem.glob("previsoes_*.npz"):
        shutil.copy2(arq, ic.PASTA / arq.name)
    shutil.rmtree(FINAL)
    shutil.copytree(origem / "producao" / "final", FINAL)
    producao = json.loads((origem / "producao" / "producao.json").read_text(encoding="utf-8"))

    modelo = ModeloEnsemble.carregar(FINAL)
    atual = pd.read_parquet(RAIZ / "dados" / "processados" / "base_atual.parquet")
    amostra = atual.sample(n_amostra, random_state=SEED).reset_index(drop=True)
    p = modelo.prever(amostra)
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), "previsões inválidas"
    modelo.preparar_explicador(amostra)
    importancia = modelo.explicar(amostra).abs().mean().sort_values(ascending=False)
    modelo.salvar(FINAL, componentes=False)
    print(f"modelos carregados; {len(modelo.features)} variáveis; erro da soma (p95) {modelo.erro_aditividade:.3f} "
          f"({time.time() - t0:.0f}s)", flush=True)

    janelas = o.carregar_janelas(RAIZ / "dados" / "processados" / "base_modelagem.parquet")
    prev = ic.carregar_previsoes(janelas)
    previsoes = {**prev, "ensemble_empilhado": o.combinar_ensembles(prev, janelas)["previsoes"]["ensemble_empilhado"],
                 "media": {j: 1 / (1 + np.exp(-np.mean([logit(prev[n][j]) for n in prev], axis=0))) for j in JANELAS}}
    te, v1 = janelas["teste"], janelas["val1"]
    y, w = te["alvo_inad"].to_numpy(), te["peso"].to_numpy()
    y1, w1 = v1["alvo_inad"].to_numpy(), v1["peso"].to_numpy()
    resultados, extras = {}, {}
    for chave, pv in previsoes.items():
        nome = NOMES[chave]
        resultados[nome] = o.avaliar(pv, janelas)
        metodo, _ = escolher_calibrador(pv["val1"], y1, w1, v1["data_base"])
        pc = Calibrador(metodo).fit(pv["val1"], y1, w1).transform(pv["teste"])
        extras[nome] = (pv["teste"], pc, resultados[nome]["fatia_pct"] / 100)
    principal = NOMES["media"]
    intervalos = intervalos_bootstrap(y, w, te["data_base"], te["uf"],
                                      {n: extras[n] for n in (principal, NOMES["ensemble_empilhado"], NOMES["lightgbm"])}, principal)
    r = resultados[principal]
    print(f"teste da média ({time.time() - t0:.0f}s): PR-AUC {r['pr_auc']:.4f}, AUC {r['auc']:.4f}, F1 {r['f1']:.3f}, "
          f"Brier {r['brier']:.5f}", flush=True)

    base = pd.read_parquet(RAIZ / "dados" / "processados" / "base_modelagem.parquet", columns=["data_base"])
    meta = json.loads((MODELOS / "metadados.json").read_text(encoding="utf-8"))
    teste_ate = str(te["data_base"].max().date())
    meta |= {
        "modelo_escolhido": principal, "formato": "ensemble", "tipo": "ensemble", "features": modelo.features,
        "criterio_escolha": "backtest_erro_de_nivel",
        "producao": {**{k: producao[k] for k in ("combinacao", "calibracao", "fatia_pct", "limiar_pct", "janela_de", "janela_ate")},
                     "nota": "média das log-chances com peso 0,25 e intercepto 0, sem calibrador; a fatia vem das previsões "
                             "fora da amostra da janela de teste nos 6 meses mais recentes",
                     "variante": variante},
        "importancia_shap_global": importancia.round(5).to_dict(),
        "dados_ate": str(base["data_base"].max().date()), "n_total": int(len(base)), "n_teste": int(len(te)),
        "teste_ate": teste_ate,
        "comparacao": {**meta["comparacao"], **resultados},
        "intervalos_teste": intervalos,
        "lspa": {k: v for k, v in decisao.items() if k != "por_mes"},
        "nota_comparacao": f"os quatro modelos do Optuna, a média e o empilhado foram medidos no teste até {teste_ate}; os "
                           "modelos anteriores ao ensemble, no teste até 2026-02-28",
    }
    for nome, (bruto, pc, _fatia) in extras.items():
        meta["calibracao_teste"][nome] = tabela_calibracao(y, w, pc)
        meta["calibracao_teste_bruta"][nome] = tabela_calibracao(y, w, bruto)
        meta["curva_limiar_teste"][nome] = curva_limiar(y, w, pc)
        meta["beneficio_teste"][nome] = curva_beneficio(y, w, pc)
        meta["distribuicao_teste"][nome] = distribuicao_por_resultado(y, w, pc)
    (MODELOS / "metadados.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"pronto em {time.time() - t0:.0f}s: {principal} ({variante}); fatia {producao['fatia_pct']:.1f}%, "
          f"limiar {producao['limiar_pct']:.2f}%")
    print(importancia.head(10).round(4).to_string())


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "Downloads" / "retreino.zip")
