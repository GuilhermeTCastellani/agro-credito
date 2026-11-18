# Previsão de Inadimplência por Perfil de Crédito Rural — PoC

O produto principal é a **previsão por perfil**: para cada perfil da carteira de crédito rural (UF, tipo de instituição, PF/PJ,
ocupação ou atividade, porte, finalidade, origem dos recursos e indexador), o modelo prevê a inadimplência (atraso > 90 dias)
**6 meses à frente** e dá um nível de alerta: **crítico** a partir de 10%, **atenção** de 5% a 10% e normal abaixo disso. A
inadimplência prevista é a fração dos reais emprestados ao perfil que deve estar com atraso acima de 90 dias. O resultado é a lista
de **perfis em alerta** da carteira, ordenada pelo valor em risco.

Treinado **exclusivamente com dados oficiais e públicos** (Banco Central e IBGE), sem dados sintéticos. Dashboard em Streamlit
com explicabilidade via SHAP.

## Como rodar

Há três níveis, do mais rápido ao mais completo. Para a maioria das pessoas, basta o primeiro.

| Quero... | Precisa de | Tempo |
|---|---|---|
| 1. Abrir o dashboard | Python e este repositório | 10 minutos |
| 2. Atualizar com meses novos do Banco Central | o nível 1 e internet | cerca de 1 hora |
| 3. Retreinar os modelos | o nível 2 e o Google Colab (GPU) | algumas horas |

### 1. Abrir o dashboard

**Requisitos:** Python 3.14 (foi o testado; versões a partir de 3.12 devem funcionar) e cerca de 4 GB de memória livre.
Feche programas pesados antes: o computador de desenvolvimento tem 7,4 GB e o dashboard já caiu por falta de memória.

**a) Baixe o projeto e instale as dependências.** No Windows (PowerShell), na pasta do projeto:

```powershell
git clone https://github.com/GuilhermeTCastellani/agro-credito.git
cd agro-credito
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

No Linux ou no macOS, troque `.venv\Scripts\python` por `.venv/bin/python`.

Os modelos em produção (`modelos/ensemble/final/`, cerca de 130 MB) e os dados de que o dashboard precisa
(`dados/processados/`) já vêm no repositório, por isso o clone demora um pouco.

**b) Suba o dashboard:**

```powershell
.venv\Scripts\python -m streamlit run app.py
```

Ele abre em http://localhost:8501. A primeira carga leva alguns segundos, porque o modelo vai para a memória.

**c) Opcional: confira que está tudo certo** (cerca de 2,5 minutos, 87 testes):

```powershell
.venv\Scripts\python -m pytest tests -q
```

### O que o dashboard mostra

- **Perfis em alerta**: quanto da carteira mais recente está em cada nível, a lista dos perfis com mais dinheiro em risco, a
  medição do teste e a visão agregada por UF e finalidade.
- **Avaliar perfil**: a pessoa escolhe o perfil e a operação e recebe o veredito (normal, atenção ou crítico), a inadimplência
  prevista numa régua de risco e os fatores que mais pesaram (SHAP). O cenário da UF vem com os dados oficiais mais recentes e
  pode ser alterado para testar cenários.
- **Avaliação em lote**: baixa uma planilha de exemplo (abre direto no Excel), preenche uma linha por produtor e envia. O
  resultado traz o nível de alerta, a inadimplência prevista, o principal fator de risco e o principal atenuante de cada linha.
- **Desempenho do modelo**: a medição por perfil, a comparação dos modelos, o previsto contra o real mês a mês e a calibração.
- **Dados e método**: fontes, método e limitações.

## Resultado principal: perfis em alerta (teste)

Um perfil é sinalizado quando a inadimplência prevista passa do limite: 10% mede o nível crítico e 5% mede atenção ou crítico. O teste usa perfis de jan/2025 a mar/2026 (alvos de jul/2025 a
set/2026), com a média dos quatro modelos calibrada na primeira janela de validação. Precisão, recall e F1 pesam por **reais de carteira**.

| Perfil passa de | Precisão | Recall | **F1 do modelo** | F1 repetindo o nível atual | F1 sinalizando tudo | Carteira sinalizada |
|---|---|---|---|---|---|---|
| **5% (atenção ou crítico)** | **0,735** | **0,863** | **0,794** | **0,691** | **0,506** | **40%** |
| **10% (crítico)** | **0,704** | **0,627** | **0,664** | **0,562** | **0,301** | **16%** |
| 20% | 0,731 | 0,380 | 0,500 | 0,469 | 0,133 | 4% |

O F1 em 5% sai maior em parte porque mais carteira passa de 5% (34% no teste, contra 18% em 10%): sinalizar tudo já dá 0,506. O
ganho sobre repetir o nível atual é praticamente o mesmo nos dois limites. O 10% marca o nível crítico porque a lista de 5% é larga
demais para priorizar e porque 5% fica abaixo da inadimplência média atual (6,5%).

Contando perfis em vez de reais, o F1 em 10% é 0,634 (contra 0,552 repetindo o nível atual). Agregando em segmentos UF × finalidade,
o F1 em 10% é 0,751 contando segmentos e 0,593 pesando por reais. A medição completa fica em `modelos/metadados.json`, na chave
`segmentos_teste`, e é gerada por `python src/segmentos.py`.

**Limites desta medição.** Os limites foram escolhidos depois de ver o teste, e o modelo também foi escolhido no teste, por orientação
do professor. Os números são otimistas até virem meses novos do Banco Central. Em 20%, o ganho sobre repetir o nível atual é pequeno.

**Na carteira atual.** Com os perfis de set/2026, o modelo prevê 7,8% de inadimplência em 6 meses, contra 6,5% hoje. Dos 21.937 perfis,
10.233 ficam críticos (25,6% da carteira; no teste, eram 16%), 3.230 em atenção (23,1%) e 8.474 normais (51,4%). O nível previsto é incerto
(veja abaixo), e quem passa dos limites depende dele: use a lista mais como ordem de prioridade do que como contagem exata.

**Primeira conferência fora da amostra (set/2026).** Antes do retreino, a média projetava 7,4% para set/2026; o real foi 6,5% (a
inadimplência caiu de 7,2% em agosto). O empilhamento anterior projetava 8,9%. O erro caiu de 2,4 para 0,9 p.p., mas ainda para cima.

**Previsto × real, mês a mês.** Nos alvos de jul/2024 a set/2026 (janelas de validação e teste), o modelo errou o nível para os dois lados,
sem um viés fixo: previu alto demais de jan a abr/2025 e de fev a abr/2026 (até 0,6 p.p.) e baixo demais de ago a dez/2025 (até 0,8 p.p.) e
em ago–set/2026 (6,2% previstos contra 7,2% reais em agosto; 5,9% contra 6,5% em setembro). As maiores superestimações por UF foram no RS logo depois das enchentes de 2024 e nas UFs
de grãos na colheita (fev–abr); o modelo subestima de forma sistemática MA, TO, PI, GO, MS e MT e a finalidade comercialização.

**Projeção.** O modelo em produção, aplicado aos perfis de abr a set/2026, projeta a inadimplência da carteira em 7,7% em out/2026,
com pico de 8,8% em jan/2027 e 7,8% em mar/2027 (hoje: 6,5%). A série histórica fica em `modelos/metadados.json`, na chave `acompanhamento`, e a
projeção, na chave `projecao`; as duas aparecem no gráfico da aba de desempenho, por UF. Pesam pela carteira, sem teto, e são geradas
por `python src/segmentos.py`. Parte da diferença entre set/2026 (último mês conhecido) e out/2026 (primeiro projetado) vem da troca do
modelo do teste pelo de produção.

## Modelo de base: média de quatro modelos

A previsão por perfil vem da média simples das log-chances de XGBoost, CatBoost, Random Forest e LightGBM, com hiperparâmetros
buscados pelo Optuna pela PR-AUC (50 tentativas por modelo) e treino final no Colab. Medido como ordenação de reais de carteira com o
alvo fracionário, no teste (até mar/2026): PR-AUC 0,273, precisão 0,417, recall 0,191, F1 0,262, AUC 0,817 e Brier 0,0525. Esse F1
responde a outra pergunta e não se compara com o F1 por perfil. Em produção (retreino com o SCR até set/2026): sem calibrador, fatia de
9,9% e limiar de 18,15%.

**Safra corrente (LSPA) testada e deixada de fora.** Duas variáveis por UF, só com levantamentos já publicados (variação da safra de
grãos estimada contra o ano anterior e revisão da estimativa em 3 levantamentos), foram comparadas num backtest de 9 origens com e sem
elas. Regra fixada antes: entra se a PR-AUC subir sem piorar o nível nem o alerta crítico. Resultado: PR-AUC 0,2963 com e 0,2970 sem,
erro de nível 1,06 p.p. nos dois, F1 ≥ 10% 0,665 com e 0,667 sem. A safra corrente não acrescentou informação ao que o histórico do
perfil e da UF já carrega. Decisão registrada em `modelos/metadados.json`, chave `lspa`.

**Por que a média, e não o empilhamento.** Até 10/11/2026 a produção usava um empilhamento (regressão logística sobre os quatro
modelos, com pesos da validação de 2024). Os pesos somavam 1,12 e tinham intercepto positivo, o que inflava o nível nos modelos de
produção. Um backtest trimestral (`src/backtest.py`) simulou o procedimento de produção em 8 origens, de mar/2024 a dez/2025: treino
até 6 meses antes de cada origem e previsão dos 3 meses seguintes. Avaliado a partir de mar/2025, com a regra fixada antes de ver o
resultado (menor erro de nível, sem perder mais de 0,01 em PR-AUC e F1 no alerta crítico):

| Forma de combinar | Erro de nível (p.p.) | Viés | PR-AUC | AUC | F1 ≥ 10% | F1 ≥ 5% |
|---|---|---|---|---|---|---|
| **Média** | **1,14** | **−0,24** | **0,296** | **0,818** | **0,658** | **0,788** |
| Empilhamento (anterior) | 1,75 | +1,17 | 0,296 | 0,818 | 0,666 | 0,780 |
| Empilhamento reajustado a cada origem | 2,14 | +1,43 | 0,289 | 0,815 | 0,653 | 0,774 |
| Média com ajuste de nível móvel | 2,17 | +1,20 | 0,285 | 0,811 | 0,638 | 0,758 |
| Empilhamento + calibração móvel | 2,55 | +1,96 | 0,288 | 0,814 | 0,642 | 0,766 |
| Média + calibração móvel | 2,65 | +1,99 | 0,288 | 0,813 | 0,639 | 0,763 |

O período avaliado coincide com o teste, que já tinha sido visto: o backtest escolhe o procedimento, e a confirmação limpa vem dos
meses novos do Banco Central. Resultados em `modelos/backtest/resultados.json`. Para voltar ao empilhamento, copie os arquivos de
`modelos/backup_antes_media/` de volta (`ensemble.joblib` para `modelos/ensemble/final/`) e rode `python src/segmentos.py`.

## Histórico: modelo anterior (XGBoost, antes do ensemble)

Resultados do modelo que estava em produção antes do ensemble, mantidos para comparação.

Critérios definidos com o professor: os hiperparâmetros são buscados pelo **Brier**, os modelos são decididos por **F1 e
AUC-ROC juntos** e a comparação usa **várias métricas**, sem julgar por uma só.

Há duas janelas de **validação**, sempre com embargo de 6 meses. A primeira treina com perfis até jun/2023 e avalia em
jan–jun/2024: nela são feitas a busca, a calibração e a escolha da **fatia da carteira** a sinalizar, que é a fração sinalizada
pelo threshold de melhor F1. A segunda treina com perfis até dez/2023 e avalia em jul–dez/2024: vence a melhor posição média em
F1 e AUC-ROC. Essa forma de combinar as duas métricas é uma decisão nossa; em empate, vence o modelo mais equilibrado. O **teste**
treina com perfis até jun/2024 e avalia em perfis de jan/2025 a fev/2026 (alvos de jul/2025 a ago/2026), período em que a
inadimplência rural saltou de ~2% para ~7%. Inadimplência real no teste: 6,2%.

Precisão, recall, F1 e MCC medem **reais de carteira**, não número de produtores. Em cada mês, são sinalizados os perfis de maior
risco até completar a fatia. Benefício é o benefício líquido médio com 3, 5, 10 e 20 alarmes falsos aceitos por real inadimplente
encontrado. Carteira é a fração sinalizada, e Previsto é a inadimplência média prevista.

| Modelo | AUC | Brier | Brier Skill | Precisão | Recall | F1 | MCC | AUPRC | H-measure | Benefício | Carteira | Previsto |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Referência (taxa atual da UF) | 0,645 | 0,0576 | 0,009 | 0,111 | 0,447 | 0,178 | 0,117 | 0,114 | 0,009 | 0,0049 | 25,0% | 3,9% |
| Regressão Logística | 0,742 | 0,0562 | 0,034 | 0,211 | 0,172 | 0,190 | 0,143 | 0,138 | 0,019 | 0,0086 | 5,1% | 4,4% |
| Random Forest | 0,702 | 0,0572 | 0,017 | 0,276 | 0,194 | 0,228 | 0,190 | 0,131 | 0,020 | 0,0079 | 4,4% | 3,7% |
| XGBoost | 0,746 | 0,0567 | 0,025 | 0,282 | 0,137 | 0,184 | 0,160 | 0,171 | 0,033 | 0,0070 | 3,0% | 3,3% |
| XGBoost relativo à UF | 0,760 | 0,0547 | 0,061 | 0,363 | 0,112 | 0,171 | 0,174 | 0,183 | 0,038 | 0,0127 | 1,9% | 6,1% |
| XGBoost + histórico do perfil | 0,802 | 0,0546 | 0,062 | 0,336 | 0,209 | 0,258 | 0,228 | 0,218 | 0,057 | 0,0131 | 3,9% | 4,2% |
| XGBoost relativo à UF + histórico | 0,782 | 0,0546 | 0,061 | 0,344 | 0,171 | 0,228 | 0,208 | 0,168 | 0,034 | 0,0120 | 3,1% | 6,7% |
| XGBoost relativo à UF + histórico + recência | 0,786 | 0,0539 | 0,073 | 0,336 | 0,173 | 0,229 | 0,207 | 0,198 | 0,047 | 0,0136 | 3,2% | 6,6% |
| XGBoost + SCR ampliado | 0,808 | 0,0532 | 0,086 | 0,341 | 0,189 | 0,243 | 0,218 | 0,230 | 0,060 | 0,0160 | 3,4% | 5,2% |
| XGBoost relativo à UF + SCR ampliado | 0,797 | 0,0534 | 0,082 | 0,296 | 0,249 | 0,271 | 0,228 | 0,200 | 0,048 | 0,0150 | 5,2% | 6,6% |
| XGBoost + SCR ampliado + preços | 0,806 | 0,0533 | 0,084 | 0,314 | 0,226 | 0,262 | 0,225 | 0,222 | 0,057 | 0,0160 | 4,5% | 5,2% |
| XGBoost + SCR ampliado + Conab | 0,810 | 0,0528 | 0,092 | 0,320 | 0,244 | 0,277 | 0,238 | 0,249 | 0,065 | 0,0162 | 4,7% | 4,9% |
| XGBoost + histórico (Brier) | 0,801 | 0,0544 | 0,065 | 0,390 | 0,154 | 0,221 | 0,216 | 0,234 | 0,061 | 0,0133 | 2,5% | 4,2% |
| XGBoost relativo à UF + histórico (Brier) | 0,792 | 0,0543 | 0,067 | 0,335 | 0,212 | 0,260 | 0,229 | 0,214 | 0,055 | 0,0151 | 3,9% | 6,4% |
| XGBoost + SCR ampliado (Brier) | 0,807 | 0,0530 | 0,089 | 0,373 | 0,162 | 0,226 | 0,215 | 0,236 | 0,061 | 0,0161 | 2,7% | 5,2% |
| XGBoost relativo à UF + SCR ampliado (Brier) | 0,797 | 0,0534 | 0,083 | 0,360 | 0,135 | 0,196 | 0,190 | 0,225 | 0,053 | 0,0156 | 2,3% | 7,5% |
| **XGBoost + SCR ampliado + Conab (Brier), em produção** | **0,807** | **0,0534** | **0,082** | **0,327** | **0,230** | **0,270** | **0,235** | **0,226** | **0,060** | **0,0158** | **4,4%** | **5,0%** |

Modelo em produção: intervalos de 95% no teste de AUC 0,800–0,814, Brier 0,050–0,057, F1 0,261–0,275 e benefício 0,014–0,018,
reamostrando grupos de UF e mês, o que não captura toda a dependência entre os meses. Calibração no teste: intercepto +0,24 (subestima;
alvo 0), inclinação 1,12 (alvo 1) e ECE 0,013. AUC por recorte: PF 0,77 · PJ 0,91 · cooperativas 0,75 · custeio 0,77. AUPRC: PF 0,23 · PJ 0,16.
Detalhes completos em `modelos/metadados.json` e `modelos/treino.log`.

**O que mais ajudou.** O histórico do perfil e o SCR ampliado elevaram o melhor AUC de 0,760 para 0,810. Os custos da Conab trouxeram
ganho moderado (sem busca: F1 de 0,243 para 0,277; Brier de 0,0532 para 0,0528), mas pesam pouco no SHAP. Os preços internacionais
não trouxeram ganho claro.

**O que a escolha não captou.** O modelo vencedor foi o mesmo nos três critérios, mas a versão da Conab sem busca teve AUC, Brier e F1
melhores no teste, com intervalos da diferença que não incluem zero. Ela não foi adotada para não escolher pelo teste. No conjunto, a
busca pelo Brier melhorou algumas versões e piorou outras. O vencedor do benefício líquido também muda conforme o custo aceito.

**Brier Skill.** O ganho sobre prever sempre a média é de só 8,2%, o que relativiza o Brier baixo. Ele é pequeno em parte porque o alvo é
fracionário: nem o próprio alvo como previsão passa de F1 0,414 no teste (precisão 0,401), e superar esse teto exige dados por contrato.

**Limites desta avaliação.** Esta é a quinta regra de escolha definida depois de ver o teste, então ele não é mais cego. Uma medição limpa
depende dos meses que o Banco Central ainda vai publicar.

**Produção.** O modelo final usa calibração Platt e fatia de 10,6%, ajustadas nos perfis de ago/2025 a fev/2026 com um modelo treinado 12
meses antes, o que dá threshold de 22,94% na carteira de ago/2026. Essa fatia não foi medida fora da amostra. O modelo prevê 9,6% de
inadimplência em 6 meses, contra 7,2% hoje.

## Dados

| Fonte | Uso |
|---|---|
| [SCR.data v2 — BCB](https://dadosabertos.bcb.gov.br/dataset/scr_data) (2022–2026) | Carteira e inadimplência rural por UF × tipo de instituição × PF/PJ × ocupação/CNAE × porte × finalidade × origem × indexador. Alvo e contexto regional. |
| [IBGE PAM, tabela 5457](https://sidra.ibge.gov.br/tabela/5457) (2014–2025) | Área plantada, colhida e produção de grãos por UF → anomalia de produtividade, perda de área, variação da produção. |
| [IBGE LSPA, tabela 6588](https://sidra.ibge.gov.br/tabela/6588) (2006–2026) | Estimativa da safra corrente, por levantamento mensal. Testada no backtest e deixada fora do modelo (não melhorou a PR-AUC). |
| BCB SGS 432, 433, 1 | Selic, IPCA e dólar. |
| [Banco Mundial, Pink Sheet](https://www.worldbank.org/en/research/commodity-markets) | Preços internacionais mensais de produtos agrícolas, boi e fertilizantes, convertidos para reais. Antes de uso comercial, confirme a licença. |
| [Conab, custos de produção](https://www.gov.br/conab/pt-br/atuacao/informacoes-agropecuarias/custos-de-producao/planilhas-de-custos-de-producao) | Custo total de soja e milho por município e safra. Antes de uso comercial, confirme a licença. |

**Unidade de observação:** cada perfil de crédito rural do SCR.data num mês (1.009.929 observações).
**Alvo:** fração da carteira daquele perfil com atraso > 90 dias seis meses depois, ponderada pela exposição em reais
(cada perfil entra como R$ adimplentes e R$ inadimplentes). As métricas usam essa mesma ponderação.

## Atualizar e retreinar

### 2. Atualizar com meses novos do Banco Central

O SCR.data sai com cerca de 30 dias de defasagem. Para incluir um mês novo sem retreinar os modelos:

```powershell
$env:PYTHONUTF8 = "1"
Remove-Item dados\intermediarios\scr_rural.parquet   # força a extração de novo; sem isso a coleta pula o SCR
.venv\Scripts\python src\coleta.py      # baixa SCR.data (~800 MB em dados/brutos), IBGE, SGS, Banco Mundial e Conab
.venv\Scripts\python src\features.py    # remonta a base, o contexto por UF e o histórico dos perfis
.venv\Scripts\python src\segmentos.py   # refaz os níveis de alerta, o previsto × real e a projeção
```

Depois reinicie o dashboard: ele guarda o modelo e os dados em cache.

### 3. Retreinar os modelos

Os quatro modelos (XGBoost, CatBoost, Random Forest e LightGBM) são treinados no Google Colab, com GPU, porque não cabem na
memória de um computador comum. O caminho é:

1. Rodar `notebooks/retreino_colab.ipynb` no Colab. O notebook clona este repositório (precisa do secret `GITHUB_TOKEN` na conta)
   e salva o resultado em `MyDrive/agro-credito/progresso/retreino_set2026/retreino.zip`.
2. Baixar o `retreino.zip` para a pasta Downloads e rodar `.venv\Scripts\python src\instalar_retreino.py`, que instala os modelos
   em `modelos/ensemble/final/` (com backup do anterior), mede o teste e reescreve `modelos/metadados.json`.
3. Rodar `.venv\Scripts\python src\segmentos.py` e os testes.

Os outros notebooks (`treino_colab.ipynb` e `backtest_colab.ipynb`) refazem a busca de hiperparâmetros e o backtest trimestral.

### Referência: todos os scripts

```bash
python src/coleta.py      # baixa SCR.data (~800 MB), IBGE, SGS, Banco Mundial e Conab; extrai a carteira rural
python src/features.py    # monta a base de modelagem e o contexto atual por UF
python src/treino.py      # modelos de base: busca pelo Brier, decide por F1 e AUC-ROC, calibra e salva com o SHAP
python src/montar_ensemble.py  # instala o ensemble treinado no Colab em modelos/ensemble/final/
python src/comparar_media.py   # prepara a média simples dos quatro modelos em modelos/ensemble/media/
python src/instalar_media.py   # põe a média em produção (backup em modelos/backup_antes_media/)
python src/backtest.py         # compara as formas de combinar os modelos com as previsões do backtest do Colab
python src/instalar_retreino.py  # instala o retreino do Colab (retreino.zip) em modelos/ensemble/final/
python src/segmentos.py   # mede no teste os perfis e segmentos acima de 5%, 10% e 20%
python src/trocar_criterio.py f1  # opcional: alterna o critério entre f1_auc (padrão), f1 e beneficio, sem retreinar
python -m pytest tests -q # testes de fumaça (modelo, aditividade do SHAP, dashboard)
```

Se `dados/intermediarios/scr_rural.parquet` já existir, a coleta pula a extração; apague-o para refazer.

## Estrutura

```
app.py                  dashboard Streamlit
src/coleta.py           download e extração das fontes oficiais
src/features.py         engenharia de variáveis e alvo 6 meses à frente
src/modelo.py           pré-processamento, offset, calibrador e classe ModeloRisco (prever + explicar)
src/segmentos.py        níveis de alerta (atenção 5%, crítico 10%) e medição dos perfis e segmentos
src/montar_ensemble.py  instala os quatro modelos do ensemble treinados no Colab
src/instalar_media.py   põe em produção a média simples dos quatro modelos
src/backtest.py         backtest trimestral das formas de combinar os modelos
src/trocar_criterio.py  alterna o modelo ativo entre os critérios de escolha
src/treino.py           validação e teste fora do tempo, calibração, threshold, modelo final e SHAP
modelos/                modelo treinado, metadados, log do treino e backup da versão anterior
dados/processados/      base de modelagem, contexto por UF, histórico por perfil e perfis do último mês
tests/                  testes de fumaça e das peças de calibração, threshold e histórico
```

## Limitações

- **Risco do perfil, não do indivíduo.** O Brasil não publica inadimplência por produtor (sigilo bancário). O modelo
  aprende o risco de cada perfil na carteira nacional. Para um score individual, treine o mesmo pipeline com a base
  interna de uma cooperativa ou banco: basta montar uma tabela com as mesmas colunas e um alvo 0/1 por contrato
  (peso 1). Histórico de pagamento do próprio produtor seria a variável mais forte a acrescentar.
- **Categorias do SCR.** Ocupação e porte são os declarados pelas instituições; produtores PF aparecem em geral como
  Autônomo, Empresário ou Outros.
- **Mudança de ciclo.** O nível previsto oscila: no histórico, o modelo previu alto demais no início de 2025 e baixo demais desde
  meados de 2025. Para os perfis de set/2026, prevê 7,8% de inadimplência em 6 meses, contra 6,5% hoje, e não dá para conferir esse
  número ainda; ele pode estar alto ou baixo. Esse nível decide quantos perfis ficam em alerta. Retreine e recalibre mensalmente e
  acompanhe o previsto × real.
- **Preços e custos.** O preço é internacional, não o recebido pelo produtor, e a margem da Conab cobre só soja e milho e está
  disponível em 13 UFs hoje; nas demais, o modelo preenche o valor. Os custos entram com a premissa de 3 meses de defasagem.
- **Safra defasada.** Para evitar vazamento, cada mês usa só o último PAM já publicado, com a premissa de que ele sai em
  outubro do ano seguinte. A safra em curso ainda não entra no modelo.
- **Histórico do perfil no dashboard.** Vem do SCR mais recente para aquela combinação de UF, instituição, tomador,
  ocupação, porte, finalidade, origem e indexador. Sem essa combinação, o dashboard usa o patamar da UF.
- **Portabilidade do modelo salvo.** O arquivo do XGBoost guarda um estado interno que depende do sistema em que foi
  treinado: o modelo original, treinado no Linux, não abria no Windows. O atual foi treinado no Windows e não foi testado
  no Linux; se falhar, rode `python src/treino.py` no ambiente de destino.
- **Safra no nível de UF.** Choques de safra locais (município) não são capturados.
- **Uso.** Ferramenta de apoio à análise; decisões de crédito devem considerar a política da instituição e a
  regulamentação aplicável (incluindo LGPD e a revisão de decisões automatizadas).

## Próximos passos sugeridos

1. Treinar com a base interna de uma cooperativa (nível contrato) usando `src/modelo.py` sem alterações.
2. Cruzar com o SICOR (BCB) por município e cultura para trazer quebra de safra municipal e programa (Pronaf/Pronamp).
3. Adicionar preços de commodities (CEPEA) e índices climáticos (INMET/CEMADEN).
4. Monitorar estabilidade (PSI) e recalibrar mensalmente.
