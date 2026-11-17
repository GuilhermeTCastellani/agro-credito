# Deck novo da apresentação (19/11/2026)

**Para quê:** apresentação do Trabalho de IA, grupo de 6, cerca de 15 minutos com a demonstração ao vivo incluída.
O deck atual (17 slides, ~22 min) ficou desatualizado (empilhado, ago/2026) e não ambienta a plateia.

**Objetivos:** abrir com a visão macro do agro no Brasil; uma mensagem por slide; números do retreino (média simples,
set/2026); contar backtest, conferência de set/2026 e a faixa de ±1 ponto.

**Fora:** editar o deck atual (fica intacto); mexer em modelo, dashboard ou README.

**Abordagens consideradas:** (a) atualizar o deck atual, descartada porque o usuário pediu um novo; (b) deck novo com a
mesma sequência de 17 slides, descartada porque não cabe em 15 min; (c) deck novo de 13 slides com abertura macro,
escolhida e aprovada pelo usuário em 2026-11-16.

## Estrutura aprovada

| # | Slide | Mensagem | Quem | Tempo |
|---|---|---|---|---|
| 1 | Capa | Radar de inadimplência rural por perfil | 1 | 0:20 |
| 2 | O tamanho do agro | Peso no PIB e nas exportações, volume do crédito rural | 1 | 1:00 |
| 3 | O que mudou | Inadimplência de 0,4% (2022) a 7,2% (ago/2026); Selic, preços, clima, recuperações judiciais | 1 | 1:10 |
| 4 | O risco não é igual | Diferença por UF e finalidade; pergunta do projeto | 2 | 1:10 |
| 5 | Dados e perfil | 5 fontes públicas; perfil (8 características); alvo a 6 meses | 2 | 1:20 |
| 6 | Como o modelo aprende | Grupos de variáveis; treino no passado, teste no futuro | 3 | 1:15 |
| 7 | Os modelos | Da referência (AUC 0,645) à média de 4 modelos (0,817) | 3 | 1:15 |
| 8 | Três níveis de alerta | Normal, atenção, crítico; teste contra regras simples (F1 0,664 no crítico) | 4 | 1:15 |
| 9 | A carteira hoje | Set/2026: 25,6% crítica, 23,1% atenção; prevê 7,8% contra 6,5% hoje | 4 | 1:00 |
| 10 | Previsto × real | Acompanha a tendência, atrasa nas viradas; faixa de ±1 ponto | 5 | 1:20 |
| 11 | Como escolhemos | Backtest: média × empilhado; LSPA descartada por regra fixada antes | 5 | 1:10 |
| 12 | Demonstração | Dashboard: perfis em alerta e avaliar perfil | 6 | 2:30 |
| 13 | Limites e próximos passos | Perfil não é indivíduo; números otimistas; conferir com meses novos | 6 | 1:00 |

## Decisões do construtor (não do usuário)

- Gráficos a partir das bases do projeto (SCR, Selic, preços); números externos (PIB do agro, Plano Safra,
  recuperações judiciais) pesquisados, com fonte no slide.
- Notas de fala com `[Apresentador N]` e tempo por slide.
- Artifact novo do tipo Slides; fonte em `apresentacao/novo/`.
