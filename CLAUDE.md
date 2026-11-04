# Risco de Crédito Rural (agro-credito)

PoC do Trabalho de IA: estima a probabilidade de inadimplência (atraso > 90 dias) de uma operação de crédito rural
**6 meses à frente**, a partir do perfil do produtor, da operação e do cenário da UF. Usa só dados públicos oficiais
(Banco Central e IBGE) e tem dashboard em Streamlit com explicações via SHAP.

Responda em português do Brasil. **Nunca faça commit nem push sem pedido explícito.**

## Estado atual (2026-11-04)

- Modelo em produção: **XGBoost + SCR ampliado + Conab (Brier)**, vencedor nos três critérios; 46 testes passando.
- **Orientação mais recente do professor, ainda não implementada (siga sem trocar por conta própria):** otimizar com PR-AUC no Optuna,
  com cerca de 50 tentativas e hiperparâmetros da literatura (`docs/hiperparametros-literatura.md`); decidir no teste por PR-AUC, precisão,
  recall e F1; testar ensembles. Rodar no Colab com T4. Ela substitui a anterior (Brier no tuning, F1 e AUC-ROC na escolha), que gerou o modelo atual.
- O teste já foi visto em cinco regras de escolha: não o use para novos ajustes. A medição limpa depende dos perfis a partir de
  mar/2026, cujos alvos começam em set/2026 e ainda não entraram na base.
- **Colab:** `notebooks/treino_colab.ipynb` chama `src/otimizacao.py` (XGBoost, LightGBM, CatBoost, RealMLP, Random Forest e ensembles).
  Ainda não rodou no Colab, e o RealMLP nunca foi executado. O progresso fica no Drive. Com o vencedor, retreine localmente, porque o modelo
  treinado no Colab pode não abrir no Windows; se vencer RealMLP ou ensemble, o SHAP do dashboard precisará mudar.
- **Git:** o `.git` ativo está em `C:\Users\guilh` e aponta para outro projeto. Nunca faça commit ou push por ele; use um repositório próprio deste projeto.
- Este PC tem só 7,4 GB de RAM: feche programas antes de treinar. O treino salva o progresso em `modelos/checkpoint/`.

### Pontos em aberto

- Confirmar com o professor a combinação de F1 e AUC-ROC.
- Antes de uso comercial, confirmar as licenças do Banco Mundial e da Conab.
- Estratégia comercial em debate em `docs/estrategia-comercial.md`.

## Comandos (PowerShell, na raiz do projeto)

O ambiente é o `.venv` local, com Python 3.14 no Windows.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt

# dashboard em http://localhost:8501
.venv\Scripts\python -m streamlit run app.py

# testes
.venv\Scripts\python -m pytest tests -q

# reconstruir tudo
$env:PYTHONUTF8 = "1"
.venv\Scripts\python src\coleta.py    # baixa as fontes; dados/brutos ocupa 777 MB
.venv\Scripts\python src\features.py  # monta a base, o contexto por UF e o histórico por perfil
.venv\Scripts\python src\treino.py    # reaproveita buscas salvas; --refazer-busca refaz todas
.venv\Scripts\python src\trocar_criterio.py f1  # alterna entre f1_auc (padrão), f1 e beneficio, sem retreinar
```

`coleta.py` não refaz a extração se `dados/intermediarios/scr_rural.parquet` existir. Apague esse arquivo para refazer.

## Estrutura

| Caminho | Função |
|---|---|
| `app.py` | Dashboard: avaliar produtor, avaliação em lote (CSV), desempenho do modelo, dados e método |
| `src/coleta.py` | Baixa SCR.data, IBGE PAM/LSPA, BCB SGS, Banco Mundial e Conab; extrai a carteira rural |
| `src/features.py` | Engenharia de variáveis, alvo a 6 meses e histórico do perfil |
| `src/modelo.py` | Pré-processamento, `Calibrador` e `ModeloRisco`; o joblib precisa deste módulo para carregar o modelo |
| `src/treino.py` | Janelas, métricas, fatia, busca pelo Brier, escolha, intervalos e modelos finais |
| `src/trocar_criterio.py` | Alterna o modelo ativo entre `modelos/por_criterio/{f1_auc,f1,beneficio}` |
| `tests/test_app.py` | Fumaça do modelo salvo e do dashboard |
| `tests/test_refinos.py` | Métricas, fatia, calibração, histórico, preços, custos, critérios e checkpoint |
| `modelos/` | `modelo_final.joblib`, `metadados.json`, `treino.log`, `backup_antes_refinos/`, `backup_antes_f1/`, `backup_antes_janela2/`, `backup_antes_pacote/`, `backup_antes_precos/`, `backup_antes_conab/`, `por_criterio/` |
| `dados/processados/` | `base_modelagem.parquet`, `base_atual.parquet`, `contexto_atual.json`, `perfil_atual.parquet` |

`modelo_final.joblib` é um pickle: só carregue arquivos gerados por este projeto.

## Dados

| Fonte | Uso |
|---|---|
| SCR.data v2 (BCB), 2022–2026 | Carteira e inadimplência rural; alvo e contexto regional |
| IBGE PAM, tabela 5457 (2014–2025) | Área e produção de grãos por UF |
| IBGE LSPA, tabela 6588 (2026) | Estimativa da safra corrente |
| BCB SGS 432, 433 e 1 | Selic, IPCA e dólar |
| Banco Mundial, Pink Sheet (mensal) | Preços de produtos agrícolas, boi e fertilizantes, convertidos para reais; licença comercial a confirmar |
| Conab, custos de produção | Custo de soja e milho por município e safra; licença comercial a confirmar; o portal `portaldeinformacoes` tem certificado inválido, use o gov.br |

- **Unidade:** um perfil em um mês (1.009.929 observações). O perfil combina UF, tipo de instituição, PF/PJ,
  ocupação/CNAE, porte, finalidade, origem e indexador.
- **Alvo:** fração da carteira do perfil com atraso > 90 dias seis meses depois, ponderada pela exposição.
- **Limite principal:** o modelo mede o risco do **perfil**, não do produtor individual, porque não há dado público por
  CPF/CNPJ. Para o nível individual, é preciso treinar com a base interna de um banco ou cooperativa.

## Método

Variáveis: categorias do perfil, prazo e valor médio, Selic/IPCA/dólar, safra da UF, inadimplência e atraso da UF, histórico do perfil
(`PERFIL`), SCR ampliado (`PERFIL_AMPLIADO`), preços (`PRECOS`) e custos da Conab (`CUSTOS`).

| Janela | Treino (perfis) | Avaliação (perfis) | Papel |
|---|---|---|---|
| Validação 1 | até jun/2023 | jan–jun/2024 | Busca pelo Brier; calibrador e fatia usados no teste |
| Validação 2 | até dez/2023 | jul–dez/2024 | Aplica a fatia da validação 1; a melhor posição média em F1 e AUC-ROC escolhe o modelo |
| Teste | até jun/2024 | jan/2025–fev/2026 | Só mede |
| Produção | toda a base | — | Calibrador e fatia vêm dos ~6 meses finais, previstos por modelo treinado 12 meses antes; o threshold é o que gera a fatia no último mês |

- **Fatia:** fração da exposição sinalizada pelo threshold de melhor F1. No teste, é aplicada mês a mês sobre a previsão sem calibrar.
- **SCR ampliado (`PERFIL_AMPLIADO`):** variação em 1, 3 e 12 meses, ativo problemático, vencimentos em 90 dias, crescimento em
  6 e 12 meses, inadimplência de perfis parecidos e da finalidade na UF, e mês do alvo.
- **Ajustes:** as árvores usam perda logística e parada por log-loss. A busca aleatória (20 tentativas por versão) minimiza o Brier, e o F1
  define a fatia. Versões antigas com sufixo "(F1)" foram substituídas pelas "(Brier)".
- **Conab:** margem (preço internacional em reais / custo por saca de 60 kg) e variação anual do custo, por UF, com pesos do último PAM
  publicado e premissa de 3 meses de defasagem. Custos de local inativado expiram após 24 meses.
- **Calibração:** nenhuma, Platt ou isotônica, comparadas pelo Brier: ajusta na primeira metade da janela e avalia na segunda.
- **Métricas:** AUC, KS, Gini, Brier, precisão, recall e F1 são ponderados por exposição, então medem **reais de carteira**.

## Resultados no teste (última execução)

Inadimplência real no teste: 6,2%. Benefício: média de 3, 5, 10 e 20 alarmes falsos aceitos por real inadimplente encontrado.

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

Tabela completa, intervalos e métricas de calibração em `modelos/metadados.json` e `modelos/treino.log`.

## Problemas conhecidos

- **Versão sem busca melhor no teste:** XGBoost + SCR ampliado + Conab teve AUC 0,810, Brier 0,0528 e F1 0,277, contra 0,807, 0,0534 e
  0,270 do modelo em produção, com intervalos da diferença sem zero. Não troque com base no teste.
- **Busca pelo Brier:** melhorou algumas versões e piorou outras; o ganho no teste não é consistente.
- **Nível:** no teste, prevê 5,0% contra 6,2% (intercepto +0,24). Em produção, prevê 9,6% para ago/2026, contra 7,2% hoje, ainda sem conferência.
- **Fatia instável:** 4,4% no teste, vinda da validação, e 10,6% em produção, nunca medida fora da amostra.
- **Recall baixo no teste:** 0,230, porque a fatia vem de um período de inadimplência baixa.
- **Risco alto:** no decil do topo do teste, previu 24,8% contra 35,5% reais.
- **Teto do F1:** com o alvo fracionário, nem o próprio alvo como previsão passa de F1 0,414 no teste (precisão 0,401).
- **Ganho real pequeno:** o Brier Skill é de só 8,2%.
- **Intervalos:** reamostram UF e mês, sem capturar toda a dependência entre os meses.
- **Buscas antigas:** nenhuma busca atual vem de antes da correção do vazamento, mas os resultados "(F1)" antigos ficam só nos backups.
- **Variação de 12 meses:** não existe nos primeiros meses da base, então fica vazia na parada antecipada do treino da validação 1
  (aviso do `sklearn`). Isso afeta só a contagem de rodadas nessa etapa.
- **Peso:** o corte no percentil 99 usa a base inteira, inclusive o período de teste. O efeito tende a ser pequeno, mas não foi medido.

## Armadilhas técnicas

- **Portabilidade do modelo:** o pickle do XGBoost guarda um `rng_state` que depende do sistema. O modelo treinado no Linux
  não abriu no Windows (`input stream corrupted`). O atual foi treinado no Windows e não foi testado no Linux; se falhar, retreine lá.
- **Codificação:** leia e grave JSON sempre com `encoding="utf-8"`.
- **Ordem das variáveis:** `log_valor` aplica o log na **2ª variável numérica**, que precisa ser `valor_medio_operacao_mil`.
  Acrescente novas variáveis depois dela.
- **SHAP e calibração:** `explicar()` inclui a coluna `calibracao`, para que `valor_base + soma = previsão final`. Ela não é
  uma variável de entrada: `descrever` a trata à parte e os gráficos de importância a removem.
- **Sem vazamento:** no mês t, use só dados já publicados até t e respeite o embargo de 6 meses nas janelas. A safra segue
  `ano_safra_publicada`, com a premissa de que o PAM de um ano sai em outubro do ano seguinte.
- **Novas variáveis do perfil:** acrescente em `PERFIL_AMPLIADO` (`src/modelo.py`) e também em `ROTULOS`, `UNIDADE` e `COLS_EXTRAS`
  (`app.py`); variáveis de cenário por UF vão em `COLS_PRECOS` (`app.py`). `dados/processados/base_atual.parquet` precisa ser regenerado por `features.py`.
- **Comparações:** para comparar com resultados anteriores, mantenha as versões sem otimização, que param por log-loss.
- **Cache:** o Streamlit guarda o modelo com `st.cache_resource`. Reinicie o servidor depois de retreinar.

## Regras de análise

- Calibração e threshold vêm só das janelas de validação. O modelo é decidido no teste por orientação do professor: registre sempre que
  isso deixa o número otimista e confirme com meses novos do Banco Central.
- Reporte F1, Brier e nível previsto × real juntos, sem destacar só a métrica que melhorou.
- Diga sempre quando um número foi medido na mesma janela usada para ajustar.
- Decisões de crédito exigem revisão humana e atenção à LGPD. O dashboard é apoio à análise, não decisão automática.

## Hipótese comercial (não validada com clientes)

Vender um **radar de risco de carteira** rural, não um score individual. Primeiro público a testar: gestoras de Fiagro e
CRA e securitizadoras; cooperativas de crédito são o caminho para chegar ao nível individual com dados internos.
Rascunho completo e pontos para debate em `docs/estrategia-comercial.md`.
