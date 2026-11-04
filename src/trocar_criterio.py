"""
Troca o modelo ativo do dashboard entre os critérios de escolha, sem retreinar.

Uso:  python src/trocar_criterio.py f1_auc      (padrão: F1 e AUC-ROC juntos)
      python src/trocar_criterio.py f1
      python src/trocar_criterio.py beneficio
Depois, reinicie o Streamlit, que mantém o modelo em cache.
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

CRITERIOS = ("f1_auc", "f1", "beneficio")
ARQUIVOS = ("modelo_final.joblib", "metadados.json")
MODELOS = Path(__file__).resolve().parents[1] / "modelos"


def trocar(criterio: str, modelos: Path = MODELOS) -> None:
    if criterio not in CRITERIOS:
        raise ValueError(f"critério desconhecido: {criterio}; use {CRITERIOS}")
    origem = modelos / "por_criterio" / criterio
    faltam = [a for a in ARQUIVOS if not (origem / a).exists()]
    if faltam:
        raise FileNotFoundError(f"faltam {faltam} em {origem}; rode src/treino.py")
    for arquivo in ARQUIVOS:
        temporario = modelos / f".{arquivo}.tmp"
        shutil.copy2(origem / arquivo, temporario)
        os.replace(temporario, modelos / arquivo)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("criterio", choices=CRITERIOS)
    criterio = ap.parse_args().criterio
    trocar(criterio)
    print(f"modelo ativo: critério {criterio}. Reinicie o Streamlit.")
