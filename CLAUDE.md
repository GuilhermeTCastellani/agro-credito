# Risco de Crédito Rural (agro-credito)

PoC do Trabalho de IA: estima a probabilidade de inadimplência (atraso > 90 dias) de uma operação de crédito rural
**6 meses à frente**, a partir do perfil do produtor, da operação e do cenário da UF. Usa só dados públicos oficiais
(Banco Central e IBGE) e tem dashboard em Streamlit com explicações via SHAP.

Responda em português do Brasil. **Nunca faça commit nem push sem pedido explícito.**

## Estado atual (2026-11-10)

- **Produto principal (decidido em 2026-11-09):** previsão por **perfil** em dois níveis de alerta, pela inadimplência prevista em 6 meses:
  **crítico** a partir de 10% e **atenção** de 5% a 10% (`LIMITE_CRITICO`, `LIMITE_ATENCAO` e `nivel_alerta` em `src/segmentos.py`). No teste,
  com perfis e pesando por reais: ≥ 10% precisão 0,691, recall 0,628, F1 0,658 (0,551 repetindo o nível atual, 0,296 sinalizando tudo; 16% da
  carteira); ≥ 5% precisão 0,729, recall 0,858, F1 0,789 (0,681 e 0,502; 39% da carteira). O F1 maior em 5% vem em parte da prevalência maior;
  o 10% ficou como crítico porque a lista de 5% é larga demais para priorizar e 5% fica abaixo da média atual (7,2%). Em 20%: 0,493 contra
  0,461, ganho pequeno. Medição em `metadados.json` (`segmentos_teste`), gerada por `src/segmentos.py`. Na carteira de ago/2026: 10.829 perfis
  críticos (33% da carteira), 2.921 em atenção (20,8%) e 8.360 normais (46,2%); essa contagem depende do nível previsto, que é incerto (veja o acompanhamento abaixo). As métricas do ensemble
  abaixo são do **modelo de base**, que ordena reais de carteira, e não se comparam com o F1 por perfil.
- **Dashboard:** a 1ª aba é "Perfis em alerta" (contagem por nível, medição do teste, lista filtrável e segmentos UF × finalidade como visão
  agregada); "Avaliar perfil" dá o veredito crítico, atenção ou normal, e o lote tem a coluna `nivel_alerta`; a aba de desempenho abre com a
  medição por perfil.
- **Modelo em produção (trocado em 2026-11-10, decisão do usuário após o backtest):** **média simples das log-chances** de XGBoost,
  CatBoost, Random Forest e LightGBM (peso 0,25, intercepto 0, sem calibrador), nome `Ensemble média (XGBoost, CatBoost, Random Forest e
  LightGBM)`. Componentes do Colab Pro (L4) em `modelos/ensemble/final/` (instalados por `src/montar_ensemble.py`); o resumo da média foi
  posto lá por `src/instalar_media.py`, e `metadados.json` tem `producao.combinacao = media_logit` e `criterio_escolha =
  backtest_erro_de_nivel` (o `src/segmentos.py` e o `app.py` leem isso). Backup do empilhado em `modelos/backup_antes_media/`.
- **Teste (média):** PR-AUC 0,271, precisão 0,409, recall 0,199, F1 0,268, AUC 0,817, Brier 0,0518 (empilhado: 0,269; 0,402; 0,205;
  0,272; 0,817; 0,0518). Os modelos foram escolhidos no teste, por orientação do professor; os números são otimistas.
- **Produção (supersedido pelo retreino com set/2026, veja "Retreino com LSPA" abaixo):** sem calibrador, fatia 9,1%, limiar 21,82%. Prevê 8,8% para fev/2027 (perfis de ago/2026), contra 7,2% hoje; projeção
  7,4% (set/2026) a 8,8% (fev/2027). Alertas em ago/2026: 10.411 críticos (31,9% da carteira), 3.171 em atenção (19,5%), 8.528 normais.
  Por perfil no teste: ≥ 10% F1 0,657 (precisão 0,691, recall 0,627), ≥ 5% F1 0,789, ≥ 20% F1 0,501; segmentos ≥ 10%: 0,767 contando, 0,607 em reais.
- **SHAP do ensemble:** soma as explicações dos componentes com os pesos (0,25 cada); o Random Forest usa o SHAP aproximado
  (o exato leva ~1 s por linha). Erro da soma até 0,28 em log-chance (p95).
- **Voltar ao modelo anterior:** `python src/trocar_criterio.py f1_auc` (o backup está em `modelos/backup_antes_ensemble/`).
- **Checagem fora do teste (2026-11-09):** a validação 2 não serve para escolher, porque definiu os hiperparâmetros (Optuna) e os pesos do
  empilhamento. Na validação 1 (inadimplência 1,9%), pela posição média em PR-AUC e AUC, os quatro ensembles ficam à frente dos quatro modelos
  sozinhos, e o empilhado empata em 1º com a média simples (PR-AUC 0,095 e AUC 0,770; diferenças entre ensembles na 3ª casa). O Random Forest,
  com a melhor PR-AUC no teste (0,282), é o pior na validação 1. Precisão, recall e F1 não entram, porque a fatia vem da validação 1.
- **Erros por UF e mês (2026-11-09; ensemble, val1+val2+teste, alvos jul/2024–ago/2026, calibração da val1):** quase não houve queda de
  inadimplência que o modelo não previu (4 UF-meses, só em AM, AL e SE). No Brasil, pesando pela carteira sem teto (chave `acompanhamento`), o erro
  troca de sinal: alto demais em jan–abr/2025 e fev–abr/2026 (até 0,6 p.p.), baixo demais em ago–dez/2025 (até 0,8 p.p.) e em ago/2026 (6,2% contra
  7,2%). Os números por UF a seguir vieram de uma análise com o teto do percentil 99, que puxa o nível para cima. Superestimações maiores: RS de jul/2024 a fev/2025 (0,4–0,6 p.p.,
  logo após as enchentes, o que sugere prorrogações) e UFs de grãos em fev–abr de 2025 e 2026 (GO até 1,4 p.p.; MT, MS, SP, SC), na colheita, o que
  sugere safra corrente ou sazonalidade. Subestima de forma sistemática MA, TO, GO, MS, MT e PI, e a finalidade comercialização (prevista 6,2%, real 8,9%).
- **Nível inflado da produção e candidata média simples (2026-11-09 à noite; resolvido em 2026-11-10 com a troca para a média):** os pesos do
  empilhamento (val2, 2024; soma 1,12, intercepto +0,56) amplificam o nível dos componentes de produção, que acertam sozinhos. Reajustar os
  pesos com dados recentes piora (soma 1,4–1,7), porque as previsões fora da amostra vêm de modelos defasados numa alta. `src/comparar_media.py`
  compara sem trocar o modelo ativo e grava `modelos/ensemble/media/` (`ensemble.joblib` só com o resumo: pesos 0,25 nas log-chances, intercepto 0,
  sem calibrador, fatia 9,1%, limiar 21,82%; e `comparacao.json`). Resultados: no teste fora da amostra, média = empilhado (PR-AUC 0,2705 × 0,2691,
  AUC 0,817, F1 ≥ 10% 0,657 × 0,658, erro de nível 0,37 p.p. nos dois); na produção, alvos set/2025–ago/2026 (dentro da amostra), viés +0,72 p.p. do
  empilhado contra +0,03 da média; projeção fev/2027 11,2% → 8,8%, sem o salto (set/2026: 8,9% → 7,4%); alertas em ago/2026 quase iguais (crítico
  33,0% → 31,9% da carteira; atenção 20,8% → 19,5%). Para trocar: backup de `modelos/ensemble/final/ensemble.joblib`, copiar o da `media/` por cima,
  ajustar `metadados.json` (`producao`, `importancia_shap_global`, nome do modelo) e rodar `src/segmentos.py`. Os slides só mudam se o usuário decidir.
- **Backtest trimestral (2026-11-10, em andamento, sem LSPA):** `src/backtest.py` simula a produção em 8 origens (mar/2024 a dez/2025): treino
  até t − 6 meses com o teto do peso só do treino, previsão de t a t + 2, hiperparâmetros de `modelos/hiperparametros_ensemble.json` (versionado,
  porque `modelos/colab/` é ignorado). Avalia a partir de mar/2025 seis formas de combinar (empilhado fixo, empilhado fixo + calibração móvel,
  empilhado móvel, média, média + ajuste de nível móvel, média + calibração móvel); regra fixada antes: menor erro absoluto de nível, com PR-AUC e
  F1 ≥ 10% até 0,01 abaixo do empilhado fixo. Colab: `notebooks/backtest_colab.ipynb` (piloto de 1 origem, depois as 8; salva em
  `MyDrive/agro-credito/progresso/backtest`). PC: baixar `backtest.zip`, extrair os .npz em `modelos/backtest/` e rodar `python src/backtest.py`.
  Precisa de commit e push antes do Colab. A LSPA fica para depois, se der tempo.
  **Resultado (2026-11-10, origens mar/2025–dez/2025, `modelos/backtest/resultados.json`):** vence a **média** (erro abs. de nível 1,14 p.p.,
  viés −0,24, pior −1,71; PR-AUC 0,2956, AUC 0,818, F1 ≥ 10% 0,658, F1 ≥ 5% 0,788) contra o empilhado fixo (1,75; +1,17; +3,40; 0,2955; 0,818;
  0,666; 0,780). Empilhado móvel, ajuste de nível móvel e as calibrações móveis ficam piores em nível e em ordenação. O usuário aprovou a
  troca e a média entrou em produção no mesmo dia. Dashboard e README atualizados; **slides ainda não**.
- **Próximo passo:** conferir o nível previsto quando vierem meses novos do Banco Central; ele decide quantos perfis ficam em alerta.
  Segmento ≥ 10% no teste: F1 0,754 contando segmentos e 0,557 pesando por reais.
- **Git:** esta pasta tem repositório próprio e privado (`GuilhermeTCastellani/agro-credito`). A pasta do usuário tem outro `.git`, de outro
  projeto: confira com `git rev-parse --show-toplevel` antes de qualquer commit. Commits e push só com pedido do usuário.
- Este PC tem só 7,4 GB de RAM: feche programas antes de treinar. O Colab do usuário agora é Pro, numa conta pessoal; o progresso fica em
  `MyDrive/agro-credito/progresso`, e o secret `GITHUB_TOKEN` precisa existir na conta usada.

### Retomada (parou em 2026-11-17 à noite)

- **Sessão de 17/11:** não haverá demonstração ao vivo; o slide 12 do deck mostra duas capturas do dashboard (alertas e um perfil
  crítico em MA, banco). Slide 13 ganhou "Integrar: uma API" como próximo passo (a API não foi construída, decisão do usuário).
  Nomes do grupo já estão na capa. Roteiro em `apresentacao/roteiro-apresentacao.docx` (falas, tempos, perguntas prováveis,
  checklist); falta preencher quem fala em cada parte. A aba "Avaliação em lote" foi refeita (planilha com colunas em português,
  `;` e UTF-8 com BOM para o Excel, validação por linha, resultado legível; saíram score e faixa). README ganhou "Como rodar" em
  três níveis; `requirements.txt` ganhou catboost, lightgbm e optuna. `modelos/ensemble/final/` (130 MB, os modelos em produção) passou
  a ir para o Git, a pedido do usuário (exceção no `.gitignore`, `-text` no `.gitattributes`); o resto de `modelos/ensemble/` segue fora.
  Os slides em `apresentacao/novo/` foram sincronizados com a versão publicada (inclui as edições feitas pela página). O dashboard foi derrubado por falta de memória ao fim do dia.
- **Inconsistência conhecida:** o quadro azul da aba "Perfis em alerta" diz que set/2026 foi previsto em 5,9% (modelo da janela
  de teste, chave `acompanhamento`), e o slide 11 diz 7,4% (modelo de produção da época). Explicado no roteiro, não corrigido.

- **Deck novo (2026-11-16):** https://claude.ai/artifact/StSM7i4HRX3Jd63qACBuq3, 13 slides para ~15 min (demonstração incluída), com
  abertura macro do agro, números do retreino (média, set/2026), backtest, conferência de set/2026 e a faixa de ±1 ponto. Fonte em
  `apresentacao/novo/project/`; desenho aprovado em `docs/superpowers/specs/2026-11-16-deck-novo-design.md`. Tem animação de entrada
  em todos os slides. O deck antigo (`apresentacao/project/`, LKgKTSATgyZBpZmezjAvZM) ficou intacto e desatualizado.
  **Pendente no deck:** o usuário ainda não confirmou o visual dos slides com gráfico (3, 4 e 10); preencher `[Nomes dos 6 integrantes]`;
  conferir no boletim do MAPA os 48,5% de participação do agro nas exportações (veio de notícia).
- **Inadimplência oficial × a nossa (2026-11-16):** a série do BC para crédito rural de pessoa física (SGS 21148) marca 9,67% em ago/2026
  (recorde desde 2011) e 8,68% em set/2026; a nossa (SCR.data, PF + PJ) marca 7,20% e 6,48% (só PF em set/2026: 7,8%). A diferença de
  0,9 ponto em PF não foi apurada. O slide 3 cita a série oficial como contexto. O BC regravou mai–set/2026 entre 10 e 14/11, sem mudar a
  inadimplência rural (conferido); outubro ainda não saiu.
- **Dashboard redesenhado (2026-11-16):** tema novo em `.streamlit/config.toml` (cores e fontes do deck), veredito com régua de risco em
  "Avaliar perfil", barra da carteira por nível em "Perfis em alerta" e paleta nova nos gráficos. Os 87 testes passam. **O usuário ainda
  não disse se o visual ficou bom**: começar a próxima sessão perguntando isso (fontes, régua, barra da carteira).
- `modelos/retreino/` (164 MB, download do Colab) fica fora do Git, como `modelos/colab/`.

### Sessão anterior (parou em 2026-11-10 à noite)

- **Mensagem para a apresentação (pedida pelo usuário em 2026-11-10; ainda não está em nenhum slide):** "O modelo acompanha a tendência
  de alta, mas fica atrasado nas viradas. Subestimou as subidas de ago–dez/2025 e de ago/2026 e superestimou o começo de 2025. A
  projeção atual também sobe, enquanto o real acabou de cair. É por isso que a taxa prevista deve ser lida como uma faixa de ±1 ponto."
  Gráfico de apoio (real × previsto fora do tempo × projeção atual): https://claude.ai/artifact/SuoBA9uFFVN8ywuGHpLK2o (fonte no
  scratchpad da sessão, fora do projeto). **O usuário quer montar um deck novo e melhor** (não editar o atual); começar por isso na próxima sessão.
- **Apresentação em 2026-11-19**, grupo de 6, slides + demonstração ao vivo. Deck: https://claude.ai/artifact/LKgKTSATgyZBpZmezjAvZM
  (17 slides, fonte em `apresentacao/project/`, fora do Git). Os slides ainda mostram o empilhado (projeção 11,2%, salto no gráfico): só mudam
  depois do backtest e da decisão do usuário.
- **Onde parou:** commits `3e07918` (produto por perfil, previsto × real, projeção, comparação da média) e `0bf4299` (backtest) enviados ao GitHub.
  No Colab, o piloto rodou a origem 2024-03 (XGBoost 25 s, CatBoost 20 s, RF 14 s, LightGBM 93 s; 427 mil linhas de treino), então os 4 arquivos dela
  devem estar em `MyDrive/agro-credito/progresso/backtest`. Estimativa das 8 origens: 25–40 min. A célula "Todas as origens" **não funcionou**
  por um motivo que o usuário ainda não descreveu: amanhã, pedir a mensagem de erro ou o que aconteceu (sessão caiu, GPU indisponível, erro de
  memória, Drive desmontado) antes de mexer em código.
- **Depois do backtest:** baixar `backtest.zip` para Downloads, extrair os .npz em `modelos/backtest/`, rodar `python src/backtest.py`, mostrar a
  tabela e o vencedor pela regra; o usuário decide se troca o modelo de produção (a média simples já está pronta em `modelos/ensemble/media/`) e se
  atualiza dashboard, README e slides.
- **Modelo ativo:** a média simples desde 2026-11-10 (veja acima). Dashboard parado. Os números do "Produto principal" e dos
  "Problemas conhecidos" mais abaixo ainda são do empilhado; os da média estão na linha "Produção".
- **Retreino com LSPA e SCR de set/2026 (decidido em 2026-11-10):** o BCB atualizou `scrdata_2026.zip` em 05/11 (traz set/2026).
  LSPA por levantamento (tabela 6588 tem 241 períodos, 2006-09 a 2026-09), premissa: levantamento do mês m publicado em m+1.
  Variáveis por UF: `lspa_var_producao_pct` (grãos estimados no ano corrente / estimativa final do ano anterior) e
  `lspa_revisao_3m_pct` (revisão da estimativa do ano nos últimos 3 levantamentos). **Regra fixada antes de ver resultado:** a LSPA
  entra se a PR-AUC subir no backtest com a média, sem piorar o erro de nível mais de 0,1 p.p. nem o F1 ≥ 10% mais de 0,01.
  **Conferência de set/2026 (primeiro mês fora da amostra):** real 6,48% (caiu de 7,20% em ago); a média previu 7,37% (+0,9 p.p.),
  o empilhado teria previsto 8,9% (+2,4). Superestimou em 21 de 27 UFs (maiores: RR, PE, RJ; subestimou AP e PI).
  **Resultado e instalação (2026-11-10, noite):** retreino feito no Colab (`notebooks/retreino_colab.ipynb`) e instalado por
  `src/instalar_retreino.py` (backup em `modelos/backup_antes_retreino/`). **LSPA ficou de fora** pela regra: backtest de 9 origens,
  média, PR-AUC 0,2970 sem × 0,2963 com, erro de nível 1,06 p.p. nos dois, F1 ≥ 10% 0,667 × 0,665 (chave `lspa` no metadados; as
  variáveis continuam na base e no código, desligadas por `otimizacao.definir_variaveis(False)`). Produção atual: média sem LSPA,
  35 variáveis, treinada até set/2026, fatia 9,9%, limiar 18,15%, erro da soma do SHAP 0,49 (p95). Teste até mar/2026: PR-AUC 0,273,
  AUC 0,817, precisão 0,417, recall 0,191, F1 0,262, Brier 0,0525; por perfil ≥ 10% F1 0,664 (precisão 0,704, recall 0,627), ≥ 5% 0,794,
  ≥ 20% 0,500; segmentos ≥ 10% 0,751 contando e 0,593 em reais. Carteira de set/2026: prevê 7,8% (hoje 6,5%); 10.233 críticos (25,6%),
  3.230 atenção (23,1%), 8.474 normais (51,4%). Projeção out/2026 7,7%, pico jan/2027 8,8%, mar/2027 7,8%. README, app e testes (87)
  atualizados; **não commitado** e **slides ainda não atualizados**. SCR antigo em `dados/brutos/scrdata_2026_ate_ago.zip.bak` e
  `dados/intermediarios/scr_rural_ate_ago.parquet.bak`.
- **Da sessão anterior (2026-11-05):**

- **Enviado ao professor:** `docs/resultados.xlsx` (renomeada pelo usuário; gerada por um script temporário; abas: método e limites, comparação no
  teste com posições em fórmula, intervalos 95%, pessoa física, segmentos críticos, hiperparâmetros). Aguardando as respostas às
  4 perguntas da aba "Método e limites": desempate pela posição média, escolha no teste, ensemble × LightGBM sozinho e
  segmentos críticos como resultado complementar.
- **Pessoa física no teste (ensemble):** AUC 0,777, PR-AUC 0,271, precisão 0,404, recall 0,202, F1 0,270; calibração boa de 3% a 12%,
  subestima acima de 25% previsto (previu 32%, real 49%).
- **Pendências:** resumo de 1 a 2 páginas para o professor, se ele pedir;
  decidir se apaga `previsoes_*.npz` e `resultados_optuna.json` da raiz (cópias de `modelos/colab/`, ignoradas no Git);
  `docs/resultados.xlsx` e as anotações não foram enviados ao GitHub.
- **Dashboard:** foi encerrado por falta de memória; só subir de novo se o usuário pedir.

### Pontos em aberto

- Confirmar com o professor a regra de desempate entre PR-AUC, precisão, recall e F1.
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
| `app.py` | Dashboard: perfis em alerta, avaliar perfil, avaliação em lote (CSV), desempenho do modelo, dados e método |
| `src/segmentos.py` | Níveis de alerta (atenção 5%, crítico 10%) e medição no teste de perfis e segmentos acima de 5%, 10% e 20% |
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
- **Nível:** o ensemble errou o nível para os dois lados (série em `metadados.json`, chave `acompanhamento`, gerada por `src/segmentos.py`;
  gráfico na aba de desempenho): alto demais em jan–abr/2025, baixo demais em ago–dez/2025 e jul–ago/2026. Em produção, prevê 11,2% para
  ago/2026, contra 7,2% hoje, ainda sem conferência. Não diga que o nível está "inflado": a evidência recente aponta subestimação.
  Projeção do modelo em produção (chave `projecao`, perfis de mar–ago/2026): 8,9% em set/2026 a 11,2% em fev/2027. O gráfico do previsto × real
  pesa pela carteira sem teto (bate com a taxa oficial); as métricas por perfil (`segmentos_teste`) usam o teto do treino.
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
