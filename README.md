# ClassificadorSentimentosBERT

Este projeto visa utiliza a arquitetura BERT (Bidirectional Encoder Representations from Transformers) para analisar comentários de vídeos sobre um ataque de onça ocorrido no Brasil em 2025.

## Objetivo

Treinar 3 modelos BERT separados para classificar o sentimento (positivo, neutro, negativo) dos comentários do YouTube em relação a três classes: **Onça**, **Caseiro** (a vítima) e **Notícia**.

## Abordagem e Otimizações

O modelo base utilizado é o neuralmind/bert-base-portuguese-cased, ajustado para a tarefa de classificação de 3 classes. Os seguintes valores foram definidos:

- **LR** = 2e-5
- **Weight Decay** = 0.05
- **epoch** = 10

## Dataset

 `oncas_comentarios.csv` – 4569 comentários de 3 vídeos analisados