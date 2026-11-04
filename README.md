# Previsão de Inadimplência e Risco de Crédito Agro — PoC

Modelo preditivo que estima a probabilidade de inadimplência (atraso > 90 dias) de uma operação de crédito rural
**6 meses à frente**, a partir do perfil do produtor, da operação e do cenário de safra e crédito da sua UF.
Treinado **exclusivamente com dados oficiais e públicos** (Banco Central e IBGE), sem dados sintéticos.
Dashboard em Streamlit com explicabilidade via SHAP.

## Rodar o dashboard

```bash
pip install -r requirements.txt
streamlit run app.py
```

O modelo treinado já vem em `modelos/`, então o dashboard abre direto. Abas:

- **Avaliar produtor** — o analista informa o perfil e a operação; recebe probabilidade, score (0–1000), faixa (A–E)
  e os fatores que mais pesaram (SHAP), além de indicar se o perfil passou do threshold. O cenário de safra e crédito da UF vem preenchido com os dados oficiais
  mais recentes e pode ser alterado para testes de estresse.
- **Avaliação em lote** — upload de CSV com vários produtores; devolve probabilidade, score, faixa, principal fator
  de risco e principal atenuante de cada um.
- **Desempenho do modelo** — comparação dos modelos fora do tempo, calibração, importância global e série histórica.
- **Dados e método** — fontes, método e limitações.

## Resultados (teste fora do tempo)

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
| [IBGE LSPA, tabela 6588](https://sidra.ibge.gov.br/tabela/6588) (2026) | Estimativa da safra corrente. Coletada, mas fora do modelo atual. |
| BCB SGS 432, 433, 1 | Selic, IPCA e dólar. |
| [Banco Mundial, Pink Sheet](https://www.worldbank.org/en/research/commodity-markets) | Preços internacionais mensais de produtos agrícolas, boi e fertilizantes, convertidos para reais. Antes de uso comercial, confirme a licença. |
| [Conab, custos de produção](https://www.gov.br/conab/pt-br/atuacao/informacoes-agropecuarias/custos-de-producao/planilhas-de-custos-de-producao) | Custo total de soja e milho por município e safra. Antes de uso comercial, confirme a licença. |

**Unidade de observação:** cada perfil de crédito rural do SCR.data num mês (1.009.929 observações).
**Alvo:** fração da carteira daquele perfil com atraso > 90 dias seis meses depois, ponderada pela exposição em reais
(cada perfil entra como R$ adimplentes e R$ inadimplentes). As métricas usam essa mesma ponderação.

## Reconstruir do zero

```bash
python src/coleta.py      # baixa SCR.data (~800 MB), IBGE, SGS, Banco Mundial e Conab; extrai a carteira rural
python src/features.py    # monta a base de modelagem e o contexto atual por UF
python src/treino.py      # busca pelo Brier, decide por F1 e AUC-ROC, calibra e salva com o SHAP
python src/trocar_criterio.py f1  # opcional: alterna o critério entre f1_auc (padrão), f1 e beneficio, sem retreinar
python -m pytest tests -q # testes de fumaça (modelo, aditividade do SHAP, dashboard)
```

Para atualizar mensalmente, basta rodar os três scripts de novo: o SCR.data sai com cerca de 30 dias de defasagem.
Se `dados/intermediarios/scr_rural.parquet` já existir, a coleta pula a extração; apague-o para refazer.

## Estrutura

```
app.py                  dashboard Streamlit
src/coleta.py           download e extração das fontes oficiais
src/features.py         engenharia de variáveis e alvo 6 meses à frente
src/modelo.py           pré-processamento, offset, calibrador e classe ModeloRisco (prever + explicar)
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
- **Mudança de ciclo.** A calibração aprende com a piora recente e assume que ela continua. Para os perfis de ago/2026, o modelo
  prevê 9,6% de inadimplência em 6 meses, contra 7,2% hoje, e não dá para conferir esse número ainda. Retreine e recalibre mensalmente.
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
