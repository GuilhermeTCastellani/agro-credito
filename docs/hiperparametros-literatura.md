# Hiperparâmetros e ensembles: base na literatura

Pesquisa de 2026-11-04, feita a pedido do professor, para orientar a busca com Optuna (50 tentativas por modelo).

Esses benchmarks usam bases tabulares gerais, com alvo 0/1 e até 500 mil linhas nas maiores. Esta base tem alvo fracionário
e cerca de 1 milhão de perfis-mês. Por isso, os intervalos são um ponto de partida, não uma garantia.

## Espaços de busca nas fontes

Os espaços de XGBoost, LightGBM, CatBoost e RealMLP vêm da Seção C.2 de
[Holzmüller, Grinsztajn & Steinwart (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/hash/2ee1c87245956e3eaa71aaba5f5753eb-Abstract.html).
O de XGBoost é adaptado de Grinsztajn et al., e os de LightGBM e CatBoost, de outros trabalhos citados nessa seção. Todos usam
1000 estimadores e parada antecipada, com 300 rodadas no XGBoost e no LightGBM e `od_wait=300` no CatBoost.

### XGBoost

| Parâmetro | Distribuição |
|---|---|
| `max_depth` | Inteiro uniforme [1, 11] |
| `learning_rate` | Log-uniforme [1e-5, 0,7] |
| `subsample`, `colsample_bytree`, `colsample_bylevel` | Uniforme [0,5, 1] |
| `min_child_weight` | Inteiro log-uniforme [1, 100] |
| `alpha` | Log-uniforme [1e-8, 1e-2] |
| `lambda` | Log-uniforme [1, 4] |
| `gamma` | Log-uniforme [1e-8, 7] |

### LightGBM

| Parâmetro | Distribuição |
|---|---|
| `num_leaves` | Inteiro log-uniforme [1, e⁷] |
| `learning_rate` | Log-uniforme [e⁻⁷, 1] |
| `subsample`, `feature_fraction` | Uniforme [0,5, 1] |
| `min_data_in_leaf` | Inteiro log-uniforme [1, e⁶] |
| `min_sum_hessian_in_leaf` | Log-uniforme [e⁻¹⁶, e⁵] |
| `lambda_l1`, `lambda_l2` | 0 ou log-uniforme [e⁻¹⁶, e²] |

Fixos: `bagging_freq=1`.

### CatBoost

| Parâmetro | Distribuição |
|---|---|
| `learning_rate` | Log-uniforme [e⁻⁵, 1] |
| `bagging_temperature` | Uniforme [0, 1] |
| `l2_leaf_reg` | Log-uniforme [1, 10] |
| `random_strength` | Inteiro uniforme [1, 20] |
| `one_hot_max_size` | Inteiro uniforme [0, 25] |
| `leaf_estimation_iterations` | Inteiro uniforme [1, 20] |

Fixos: `boosting_type=Plain`, `bootstrap_type=Bayesian`, `max_depth=6`.

### RealMLP

| Parâmetro | Opções |
|---|---|
| Embedding numérico | None, PBLD, PL ou PLR |
| Camada de escala | Sim (60%) ou não (40%) |
| `learning_rate` | Log-uniforme [2e-2, 0,3] |
| Dropout | 0 (30%), 0,15 (50%) ou 0,3 (20%) |
| Ativação | ReLU, SELU ou Mish |
| Camadas ocultas | 256×3 (60%), 64×5 (20%) ou 512 (20%) |
| Weight decay | 0 ou 2e-2 |
| Desvio do embedding | Log-uniforme [0,05, 0,5] |
| Label smoothing | 0 (30%) ou 0,1 (70%); no alvo fracionário, deve ficar em 0 (decisão nossa) |

### Random Forest

De [Grinsztajn, Oyallon & Varoquaux (2022)](https://arxiv.org/abs/2207.08815), Apêndice A.3:

| Parâmetro | Distribuição |
|---|---|
| `max_depth` | None (70%), 2, 3 ou 4 (10% cada) |
| `n_estimators` | Inteiro log-uniforme [9,5, 3000,5] |
| `max_features` | sqrt, log2, None ou 0,1 a 0,9 |
| `min_samples_split` | 2 (95%) ou 3 (5%) |
| `min_samples_leaf` | Inteiro log-uniforme [1,5, 50,5] |
| `bootstrap` | Sim ou não |
| `min_impurity_decrease` | 0 (85%), 0,01, 0,02 ou 0,05 (5% cada) |

## Adaptações propostas (decisões nossas, não dos artigos)

- **Taxa de aprendizado do XGBoost:** usar [1e-3, 0,3] em vez de [1e-5, 0,7], porque taxas muito baixas não aprendem em 1000 rodadas.
- **`num_leaves` do LightGBM:** começar em 2, porque uma folha só não forma árvore.
- **Random Forest:** limitar `n_estimators` a 300 e fixar `max_depth` em None ou de 10 a 25, por causa de memória e do tamanho da base.
- **Parada antecipada:** usar 100 rodadas em vez de 300, para caber nas sessões do Colab.
- **Alvo fracionário:** usar `reg:logistic` no XGBoost, `cross_entropy` no LightGBM e `CrossEntropy` no CatBoost. Falta confirmar como
  o RealMLP aceita alvo fracionário e pesos; se não aceitar, cada perfil vira duas linhas com pesos, o que dobra a memória.

## Otimização

- **Optuna:** [Akiba et al. (KDD 2019)](https://arxiv.org/abs/1907.10902). Será usado com o amostrador TPE e semente fixa.
- **Métrica:** PR-AUC, ponderada pela exposição. A [análise do NeurIPS 2024](https://arxiv.org/abs/2401.06091v4) mostra que ela não é
  superior em geral e favorece os grupos com mais inadimplência, então será reportada também por grupo.

## Ensembles

- **Média simples e média de posições:** servem de base de comparação. A segunda funciona melhor quando os modelos têm escalas diferentes.
- **Seleção gulosa:** [Caruana et al. (ICML 2004)](https://mlanthology.org/icml/2004/caruana2004icml-ensemble). Adiciona modelos passo a
  passo, com repetição permitida, escolhendo pelo desempenho na validação.
- **Empilhamento:** regressão logística sobre as previsões de validação dos modelos.

Ganho esperado: [Holzmüller et al.](https://arxiv.org/abs/2407.04491) relatam de 0 a 3% de melhora com seleção de ensemble.
[TabArena (NeurIPS 2025)](https://proceedings.neurips.cc/paper_files/paper/2025/hash/1697e3fb412da11dc9488249f9e7bbc9-Abstract-Datasets_and_Benchmarks_Track.html)
mostra que ensembles entre modelos avançam o estado da arte, mas alerta para sobreajuste à validação.

## O que não foi verificado

- Não foi feita revisão sistemática nem encontrado espaço de busca específico para crédito rural.
- As tabelas de Holzmüller foram lidas por uma extração do artigo; confira no PDF antes de citar valores no trabalho.
- De Caruana et al., foram usados o resumo e fontes secundárias.
