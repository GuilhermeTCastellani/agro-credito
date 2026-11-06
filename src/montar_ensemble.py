"""
Instala no dashboard o ensemble empilhado treinado no Colab.

Uso: python src/montar_ensemble.py [caminho do producao_final.zip]   (padrão: Downloads)

1. Faz backup do modelo e dos metadados atuais em modelos/backup_antes_ensemble/.
2. Extrai os modelos em modelos/ensemble/final/ e confere se carregam e preveem.
3. Calcula aqui o valor de partida e a importância da explicação SHAP, numa amostra dos perfis atuais.
4. Reescreve modelos/metadados.json com os resultados do Colab, que vêm de modelos/colab/.
Para voltar ao modelo anterior: python src/trocar_criterio.py f1_auc
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
from modelo import Calibrador, ModeloEnsemble
from treino import (SEED, curva_beneficio, curva_limiar, escolher_calibrador, intervalos_bootstrap,
                    tabela_calibracao)
from treino import distribuicao_por_resultado

RAIZ = Path(__file__).resolve().parents[1]
MODELOS = RAIZ / "modelos"
DESTINO = MODELOS / "ensemble"
NOME = "Ensemble empilhado (XGBoost, CatBoost, Random Forest e LightGBM)"
NOMES_COLAB = {"xgboost": "XGBoost (Optuna, PR-AUC)", "catboost": "CatBoost (Optuna, PR-AUC)",
               "random_forest": "Random Forest (Optuna, PR-AUC)", "lightgbm": "LightGBM (Optuna, PR-AUC)",
               "ensemble_media": "Ensemble média", "ensemble_posicoes": "Ensemble de posições",
               "ensemble_guloso": "Ensemble guloso", "ensemble_empilhado": NOME}


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
    origem = DESTINO / "producao"
    for item in origem.iterdir():
        shutil.move(str(item), DESTINO / item.name)
    origem.rmdir()
    return json.loads((DESTINO / "producao.json").read_text(encoding="utf-8"))


def main(zip_caminho: Path, n_amostra: int = 300):
    t0 = time.time()
    backup = MODELOS / "backup_antes_ensemble"
    backup.mkdir(exist_ok=True)
    for arquivo in ("modelo_final.joblib", "metadados.json"):
        if not (backup / arquivo).exists():
            shutil.copy2(MODELOS / arquivo, backup / arquivo)

    producao = extrair(zip_caminho)
    modelo = ModeloEnsemble.carregar(DESTINO / "final")
    # amostra dos perfis atuais, os mesmos que o dashboard avalia; evita carregar a base inteira duas vezes
    amostra = pd.read_parquet(RAIZ / "dados" / "processados" / "base_atual.parquet").sample(n_amostra, random_state=SEED).reset_index(drop=True)
    p = modelo.prever(amostra)
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), "previsões inválidas"
    modelo.preparar_explicador(amostra)
    importancia = modelo.explicar(amostra).abs().mean().sort_values(ascending=False)
    modelo.salvar(DESTINO / "final", componentes=False)
    print(f"modelos carregados; valor_base {modelo.valor_base:.3f}; erro da soma (p95) {modelo.erro_aditividade:.3f} "
          f"log-chance ({time.time() - t0:.0f}s)", flush=True)

    # resultados do teste, refeitos com as previsões salvas do Colab
    janelas = o.carregar_janelas(RAIZ / "dados" / "processados" / "base_modelagem.parquet")
    previsoes = ic.carregar_previsoes(janelas)
    previsoes |= o.combinar_ensembles(previsoes, janelas)["previsoes"]
    te, v1 = janelas["teste"], janelas["val1"]
    y, w = te["alvo_inad"].to_numpy(), te["peso"].to_numpy()
    resultados, extras = {}, {}
    for chave, prev in previsoes.items():
        nome = NOMES_COLAB[chave]
        resultados[nome] = o.avaliar(prev, janelas)
        metodo, _ = escolher_calibrador(prev["val1"], v1["alvo_inad"].to_numpy(), v1["peso"].to_numpy(), v1["data_base"])
        pc = Calibrador(metodo).fit(prev["val1"], v1["alvo_inad"].to_numpy(), v1["peso"].to_numpy()).transform(prev["teste"])
        extras[nome] = (prev["teste"], pc, resultados[nome]["fatia_pct"] / 100)
    principais = [NOME, NOMES_COLAB["lightgbm"], NOMES_COLAB["random_forest"]]
    intervalos = intervalos_bootstrap(y, w, te["data_base"], te["uf"], {n: extras[n] for n in principais}, NOME)

    meta = json.loads((MODELOS / "metadados.json").read_text(encoding="utf-8"))
    meta.pop("vencedores_por_criterio", None)
    meta.pop("vencedor_beneficio_por_custo", None)
    meta |= {
        "formato": "ensemble", "modelo_escolhido": NOME, "tipo": "ensemble", "features": modelo.features,
        "criterio_escolha": "professor_pr_auc_precisao_recall_f1",
        "producao": {k: producao[k] for k in ("calibracao", "limiar_pct", "fatia_pct", "brier_por_metodo", "janela_de", "janela_ate")},
        "importancia_shap_global": importancia.round(5).to_dict(),
        "comparacao": {**meta["comparacao"], **resultados},
        "intervalos_teste": intervalos,
    }
    for nome, (bruto, pc, _fatia) in extras.items():
        meta["calibracao_teste"][nome] = tabela_calibracao(y, w, pc)
        meta["calibracao_teste_bruta"][nome] = tabela_calibracao(y, w, bruto)
        meta["curva_limiar_teste"][nome] = curva_limiar(y, w, pc)
        meta["beneficio_teste"][nome] = curva_beneficio(y, w, pc)
        meta["distribuicao_teste"][nome] = distribuicao_por_resultado(y, w, pc)
    (MODELOS / "metadados.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"pronto em {time.time() - t0:.0f}s: {NOME}; fatia {producao['fatia_pct']:.1f}%, limiar {producao['limiar_pct']:.2f}%")
    print(importancia.head(10).round(4).to_string())


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "Downloads" / "producao_final.zip")
