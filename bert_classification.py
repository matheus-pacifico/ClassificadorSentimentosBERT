# -*- coding: utf-8 -*-
"""# Importações"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import re

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import torch
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

import copy

"""# Criação do dataframe"""

url_arquivo = "oncas_comentarios.csv"

df = pd.read_csv(url_arquivo, encoding='utf-8')

CLASSES_A_CLASSIFICAR = ['onca', 'caseiro', 'notícia']

df = df[CLASSES_A_CLASSIFICAR + ['comment_text']]

"""# Pré-processamento"""

def substituir_nomes_de_usuarios(comentario):
  regex_nomes_em_parenteses = r'@+\S*'
  return re.sub(regex_nomes_em_parenteses, " [USUARIO] ", comentario)

def escapar_aspas_escapadas(comentario):
  comentario = re.sub('&quot;', '"', comentario)
  return re.sub("&#39;", "'", comentario)

def remover_tag_a_html(comentario):
  regex_link = r'<a\s+[^>]*>(https?:\/\/[^<]+|www\.[^<]+)+<\/a>'
  comentario = re.sub(regex_link, '', comentario)
  # Regex para substituir ponto por ponto e espaço se seguido de uma letra maiúscula
  regex_busca = r'<a\s+[^>]*>([^<]*\.[A-Z][^<]*)<\/a>'
  comentario = remover_varias_combinacoes_regex(regex_busca, r'\.(?=[A-Z])', comentario, '. ')
  # Regex para substituir ponto por espaço se seguido de algum caracter que nao seja letra maiuscula ou espaco
  regex_busca = r'<a\s+[^>]*>([^<]*\.[^A-Z|\s][^<]*)<\/a>'
  comentario = remover_varias_combinacoes_regex(regex_busca, r'\.(?![A-Z]|\s)', comentario, ' ')
  return re.sub(r'<a\s+[^>]*>(.*?)<\/a>', '', comentario)

def substituir_html(comentario):
  comentario = remover_tag_a_html(comentario)
  regex_tags_html = r'<[^>]+>'
  comentario = re.sub(regex_tags_html, " ", comentario)
  return escapar_aspas_escapadas(comentario)

def remover_varias_combinacoes_regex(regex_busca, regex_substituicao, texto, substituicao=''):
  matches = re.findall(regex_busca, texto)
  if matches:
    for match_ in matches:
      conteudo_modificado = re.sub(regex_substituicao, substituicao, match_)
      texto = re.sub(regex_busca, f'{conteudo_modificado}', texto, count=1)
  return texto

def remover_ruidos(comentario):
  #substitui espaçoes consecutivos e remove 0xa0 e u+200b ​
  comentario = comentario.replace('\xA0', ' ').replace('​', ' ')
  return re.sub(r'\s+', ' ', comentario).strip()

def normalizar(comentario):
  comentario = substituir_html(comentario)
  comentario = remover_ruidos(comentario)
  comentario = substituir_nomes_de_usuarios(comentario)
  comentario = re.sub(r'(?<=[^\s.])(\.)(?=[^\s.A-Z])', ' ', comentario)
  return re.sub(r'(?<=[^\s.])(\.)(?=[A-Z])', '. ', comentario)

TAMANHO_MINIMO_TEXTO = 15

df.dropna(inplace=True)
df.drop_duplicates(subset=["comment_text"], inplace=True)
df['comment_text'] = df['comment_text'].apply(normalizar)
df = df[df['comment_text'].str.strip().astype(bool)]
df.drop_duplicates(subset=["comment_text"], inplace=True)
df = df[df['comment_text'].str.len() >= TAMANHO_MINIMO_TEXTO]

"""# Codificação das classes"""

LABEL_MAPS = {
  'onca': {'negativo': 0, 'neutro': 1, 'positivo': 2},
  'caseiro': {'negativo': 0, 'neutro': 1, 'positivo': 2},
  'notícia': {'ruim': 0, 'neutra': 1, 'boa': 2}
}

for classe in CLASSES_A_CLASSIFICAR:
  df[classe] = df[classe].str.strip().map(LABEL_MAPS[classe])

LABEL_MAPS['notícia'] = {'negativo': 0, 'neutro': 1, 'positivo': 2}

REVERSE_LABEL_MAPS = {
    label_name: {v: k for k, v in map_dict.items()}
    for label_name, map_dict in LABEL_MAPS.items()
}

"""# Divisão do conjunto"""

def criar_datasets(X, y, classe):
  X_treino_val, X_teste, y_treino_val, y_teste = train_test_split(
      X, y, test_size=0.15, random_state=42, stratify=y
  )
  X_treino, X_validacao, y_treino, y_validacao = train_test_split(
      X_treino_val, y_treino_val, test_size=(0.15/0.85), random_state=42, stratify=y_treino_val
  )
  return X_treino, X_validacao, X_teste, y_treino, y_validacao, y_teste

X = df['comment_text']
X_dataframes = {}
y_dataframes = {}
for classe in CLASSES_A_CLASSIFICAR:
  y = df[classe]
  X_treino, X_validacao, X_teste, y_treino, y_validacao, y_teste = criar_datasets(X, y, classe)

  X_dataframes[classe] = {'X_treino': X_treino, 'X_validacao': X_validacao, 'X_teste': X_teste}
  y_dataframes[classe] = {'y_treino': y_treino, 'y_validacao': y_validacao, 'y_teste': y_teste}

  print(f"Amostras de treino {classe}: {len(X_treino)} = {len(X_treino)/len(df):.0%}")
  print(f"Amostras de validação {classe}: {len(X_validacao)} = {len(X_validacao)/len(df):.0%}")
  print(f"Amostras de teste {classe}: {len(X_teste)} = {len(X_teste)/len(df):.0%}\n")

"""# Definição de parêmetros"""

MODEL_NAME = 'neuralmind/bert-base-portuguese-cased'

NOVOS_TOKENS = ['[USUARIO]']

WEIGHT_DECAY = 0.05

BATCH_SIZE = 16
EPOCAS = 10
LEARNING_RATE = 2e-5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Usando **{device}**')

"""# Tokenizer, Modelos, DataLoaders
# Optimizers, Schedulers
"""

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME, do_lower_case=False)
tokenizer.add_tokens(NOVOS_TOKENS)

def tokenize_data(texts, labels, tokenizer, max_len=512):
  encoded_data = tokenizer.batch_encode_plus(
    texts.tolist(),
    add_special_tokens=True,
    max_length=max_len,
    padding='max_length',
    return_attention_mask=True,
    return_tensors='pt',
    truncation=True
  )
  input_ids = encoded_data['input_ids']
  attention_masks = encoded_data['attention_mask']
  labels_tensor = torch.tensor(labels.values)
  return TensorDataset(input_ids, attention_masks, labels_tensor)

data_loaders = {}

for classe in CLASSES_A_CLASSIFICAR:
  treino_dataset = tokenize_data(X_dataframes[classe]['X_treino'], y_dataframes[classe]['y_treino'], tokenizer)
  validacao_dataset = tokenize_data(X_dataframes[classe]['X_validacao'], y_dataframes[classe]['y_validacao'], tokenizer)
  teste_dataset = tokenize_data(X_dataframes[classe]['X_teste'], y_dataframes[classe]['y_teste'], tokenizer)

  data_loaders[classe] = {
    'treino': DataLoader(treino_dataset, BATCH_SIZE, shuffle=True),
    'validacao': DataLoader(validacao_dataset, BATCH_SIZE),
    'teste': DataLoader(teste_dataset, BATCH_SIZE)
  }

modelos = {}
optimizers = {}
schedulers = {}

for classe in CLASSES_A_CLASSIFICAR:
  NUM_CLASSES = len(LABEL_MAPS[classe])

  modelo = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_CLASSES,
    output_attentions=False,
    output_hidden_states=False
  )
  modelo.resize_token_embeddings(len(tokenizer))
  modelo.to(device)
  modelos[classe] = modelo

  optimizers[classe] = {'AdamW': AdamW(modelo.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)}
  total_steps = len(data_loaders[classe]['treino']) * EPOCAS
  schedulers[classe] = {
    'linear': get_linear_schedule_with_warmup(optimizers[classe]['AdamW'], num_warmup_steps=0, num_training_steps=total_steps),
    'cosine': get_cosine_schedule_with_warmup(optimizers[classe]['AdamW'], num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
  }

"""# Treinamento e Validação"""

melhor_modelo = {}

for classe in CLASSES_A_CLASSIFICAR:
  print(f'--- INICIANDO TREINAMENTO PARA CLASSE: {classe} ---\n')
  modelo = modelos[classe]
  acc_melhor_modelo=0
  optimizer = optimizers[classe]['AdamW']
  scheduler = schedulers[classe]['linear']
  treino_dataloader = data_loaders[classe]['treino']
  validacao_dataloader = data_loaders[classe]['validacao']
  estatistica_treino = []

  for i in range(0, EPOCAS):
    # --- TREINAMENTO ---
    print(f'Treinando... época {i+1}')
    modelo.train()
    loss_total_treino = 0

    for step, batch in tqdm(enumerate(treino_dataloader), total=len(treino_dataloader)):
      b_input_ids = batch[0].to(device)
      b_input_mask = batch[1].to(device)
      b_labels = batch[2].to(device)

      optimizer.zero_grad()

      outputs = modelo(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
      loss = outputs.loss

      loss_total_treino += loss.item()

      loss.backward()
      torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
      optimizer.step()
      scheduler.step()

    media_treino_loss = loss_total_treino / len(treino_dataloader)

    # --- VALIDAÇÃO ---
    print(f'Validando... época {i+1}')
    modelo.eval()
    total_eval_loss = 0
    total_eval_acc = 0
    total_samples = 0

    for step, batch in tqdm(enumerate(validacao_dataloader), total=len(validacao_dataloader)):
      b_input_ids = batch[0].to(device)
      b_input_mask = batch[1].to(device)
      b_labels = batch[2].to(device)

      with torch.no_grad():
        outputs = modelo(b_input_ids, attention_mask=b_input_mask, labels=b_labels)

      loss = outputs.loss
      logits = outputs.logits

      total_eval_loss += loss.item()

      logits = logits.detach().cpu().numpy()
      label_ids = b_labels.to('cpu').numpy()

      pred_flat = np.argmax(logits, axis=1).flatten()
      total_eval_acc += np.sum(pred_flat == label_ids)
      total_samples += len(label_ids)

    media_loss_val = total_eval_loss / len(validacao_dataloader)
    media_acc_val = total_eval_acc / total_samples

    print(f"Época {i + 1}/{EPOCAS} | Treino Loss: {media_treino_loss:.4f} | Val. Loss: {media_loss_val:.4f} | Val. Acc: {media_acc_val:.4f}\n")

    estatistica_treino.append(
      {
        'epoca': i + 1,
        'Treino Loss': media_treino_loss,
        'Valid Loss': media_loss_val,
        'Valid Accur': media_acc_val
      }
    )
    if media_acc_val > acc_melhor_modelo:
      acc_melhor_modelo = media_acc_val
      melhor_modelo[classe] = copy.deepcopy(modelo)

  df_stats = pd.DataFrame(data=estatistica_treino)
  plt.figure(figsize=(10,5))
  plt.plot(df_stats['epoca'], df_stats['Treino Loss'], label="Loss do Treinamento")
  plt.plot(df_stats['epoca'], df_stats['Valid Loss'], label="Loss da Validação")
  plt.title(f"Evolução do Loss ({classe})")
  plt.xlabel("Época")
  plt.ylabel("Loss")
  plt.xticks(df_stats['epoca'])
  plt.legend()
  plt.show()
  print()

"""# Avaliação"""

resultados = {}

for classe in CLASSES_A_CLASSIFICAR:
  teste_dataloader = data_loaders[classe]['teste']
  modelo = melhor_modelo[classe]
  modelo.eval()
  all_pred = []
  all_true = []

  for batch in teste_dataloader:
    b_input_ids = batch[0].to(device)
    b_input_mask = batch[1].to(device)
    b_labels = batch[2].to(device)

    with torch.no_grad():
      outputs = modelo(b_input_ids, attention_mask=b_input_mask)

    logits = outputs.logits
    logits = logits.detach().cpu().numpy()
    label_ids = b_labels.to('cpu').numpy()

    pred_flat = np.argmax(logits, axis=1).flatten()

    all_pred.extend(pred_flat)
    all_true.extend(label_ids)

  print(f'\n--- Classification Report para {classe} ---')
  target_names = [REVERSE_LABEL_MAPS[classe][i] for i in sorted(REVERSE_LABEL_MAPS[classe].keys())]
  report = classification_report(
    all_true,
    all_pred,
    target_names=target_names,
    zero_division=0,
    output_dict=True
  )
  print(classification_report(
    all_true,
    all_pred,
    target_names=target_names,
    zero_division=0))

  resultados[classe] = {
    'accuracy': report['accuracy'],
    'y_true': all_true,
    'y_pred': all_pred,
    'X_test': X_dataframes[classe]['X_teste']
  }

print('RESULTADOS E ERROS DE CLASSIFICAÇÃO')

for classe in CLASSES_A_CLASSIFICAR:
  resultado = resultados[classe]
  print(f"\n--- CLASSIFICADOR ({classe}) ---")
  print(f"Acurácia no Teste: {resultado['accuracy']:.4f}")

  error_analysis_df = pd.DataFrame({
    'comment_text': resultado['X_test'],
    'true': [REVERSE_LABEL_MAPS[classe][l] for l in resultado['y_true']],
    'pred': [REVERSE_LABEL_MAPS[classe][l] for l in resultado['y_pred']]
  })

  erros_df = error_analysis_df[error_analysis_df['true'] != error_analysis_df['pred']]

  if erros_df.empty:
    print("Nenhum erro de classificação encontrado neste conjunto de teste.")
  else:
    print(f"Total de erros: {len(erros_df)} de {len(resultado['X_test'])}")
    print(f"Exemplos de Erros para '{classe}':")
    for i in range(min(3, len(erros_df))):
      erro = erros_df.iloc[i]
      print(f"\n    Comentário: \"{erro['comment_text']}\"")
      print(f"    Rótulo Correto (True): {erro['true']} | Previsão (Pred): {erro['pred']}")
  print()