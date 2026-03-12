# ==============================================================================
# PROJETO: Predição de Idade via ECG (TCC Engenharia de Computação - Feevale)
# AUTOR: Marcio Jose Gomes Bastos Filho
# DESCRICAO: Script de diagnóstico de hardware. Verifica se a biblioteca PyTorch
#            consegue acessar a GPU via drivers CUDA. Essencial para garantir
#            a performance necessária no treinamento de Redes Neurais.
# DEPENDENCIAS: torch
# INSTRUCOES: Executar em qualquer novo ambiente ou servidor antes de iniciar
#             o script 'train.py'.
# ==============================================================================

import torch # Importo a biblioteca principal de Tensores e Deep Learning.

# Verifico se o driver CUDA está instalado e acessível pelo PyTorch.
# O resultado deve ser True para que o treinamento utilize a aceleração da GPU.
print("CUDA disponível?", torch.cuda.is_available())

# Identifico a quantidade de placas de vídeo (GPUs) presentes no sistema.
print("Quantas GPUs detectadas:", torch.cuda.device_count())

# Se houver uma GPU disponível, recupero e exibo as especificações do hardware.
if torch.cuda.is_available():
    # Exibo o nome oficial do modelo da GPU (ex: NVIDIA GeForce RTX 3060).
    # O índice 0 representa a primeira placa de vídeo detectada.
    print("Nome da GPU:", torch.cuda.get_device_name(0))