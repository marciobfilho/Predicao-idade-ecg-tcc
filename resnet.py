# ==============================================================================
# PROJETO: Predição de Idade via ECG (TCC Engenharia de Computação - Feevale)
# AUTOR: Marcio Jose Gomes Bastos Filho
# DESCRICAO: Implementação da arquitetura de Rede Neural Residual 1D (ResNet1d).
#            Esta classe define os blocos residuais e a estrutura da rede para 
#            processamento de sinais unidimensionais (ECG).
# DEPENDENCIAS: torch, numpy
# INSTRUCOES: Esta classe é um módulo auxiliar. Deve ser importada por scripts
#             de treinamento ou avaliação (ex: train.py ou evaluate.py).
# ==============================================================================

# Importo o módulo de redes neurais do PyTorch
import torch.nn as nn 

# Importo o NumPy para operações matemáticas de suporte (cálculo de dimensões)
import numpy as np 

# --- FUNÇÕES AUXILIARES DE DIMENSIONAMENTO ---

def _padding(downsample, kernel_size):
    """
    Calcula o padding necessário para manter a consistência temporal do sinal.
    Garante que a saída tenha o tamanho correto após a convolução.
    """
    # Uso a fórmula matemática para garantir que o tamanho da saída não mude 
    # (quando downsample=1) ou seja reduzido proporcionalmente ao stride.
    padding = max(0, int(np.floor((kernel_size - downsample + 1) / 2)))
    return padding 

def _downsample(n_samples_in, n_samples_out):
    """
    Calcula o fator de redução (stride) entre duas camadas consecutivas.
    """
    downsample = int(n_samples_in // n_samples_out) 
    # Validação: o número de amostras deve sempre diminuir ou permanecer igual.
    if downsample < 1:
        raise ValueError("O número de amostras deve sempre diminuir.") 
    # Validação: a redução deve ser por um fator inteiro para evitar perda de dados desalinhada.
    if n_samples_in % n_samples_out != 0:
        raise ValueError("A redução entre blocos deve ser um fator inteiro.") 
    return downsample 

# --- DEFINIÇÃO DOS BLOCOS DA REDE ---

class ResBlock1d(nn.Module):
    """
    Implementação do Bloco Residual 1D. 
    Contém o caminho convolucional principal e a conexão de atalho (skip connection).
    """

    def __init__(self, n_filters_in, n_filters_out, downsample, kernel_size, dropout_rate):
        # O kernel deve ser ímpar para que o padding seja simétrico.
        if kernel_size % 2 == 0:
            raise ValueError("A implementação suporta apenas kernel_size ímpares.") 
        
        # Inicializo a classe base do PyTorch
        super(ResBlock1d, self).__init__() 

        # --- CAMINHO PRINCIPAL (PARTE 1) ---
        # Primeira convolução: mantém o tamanho temporal do sinal.
        padding = _padding(1, kernel_size) 
        self.conv1 = nn.Conv1d(n_filters_in, n_filters_out, kernel_size, padding=padding, bias=False) 
        self.bn1 = nn.BatchNorm1d(n_filters_out) # Normalização para estabilizar o treino.
        self.relu = nn.ReLU() # Ativação não-linear.
        self.dropout1 = nn.Dropout(dropout_rate) # Regularização para evitar overfitting.

        # --- CAMINHO PRINCIPAL (PARTE 2) ---
        # Segunda convolução: aplica o downsample (redução da resolução temporal).
        padding = _padding(downsample, kernel_size) 
        self.conv2 = nn.Conv1d(n_filters_out, n_filters_out, kernel_size,
                                 stride=downsample, padding=padding, bias=False) 
        self.bn2 = nn.BatchNorm1d(n_filters_out) 
        self.dropout2 = nn.Dropout(dropout_rate) 

        # --- CONEXÃO DE ATALHO (SKIP CONNECTION / RESIDUAL) ---
        skip_connection_layers = [] 

        # Se houver redução de amostragem, o atalho também deve reduzir para permitir a soma.
        if downsample > 1: 
            maxpool = nn.MaxPool1d(downsample, stride=downsample) 
            skip_connection_layers += [maxpool] 

        # Se o número de filtros (canais) mudar, aplico uma convolução 1x1 para alinhar dimensões.
        if n_filters_in != n_filters_out: 
            conv1x1 = nn.Conv1d(n_filters_in, n_filters_out, 1, bias=False) 
            skip_connection_layers += [conv1x1] 

        # Agrupo as camadas de ajuste do atalho em um bloco sequencial.
        if skip_connection_layers: 
            self.skip_connection = nn.Sequential(*skip_connection_layers) 
        else:
            self.skip_connection = None 

    def forward(self, x, y):
        """
        Executa a passagem de dados (Forward Pass) pelo bloco.
        x: entrada para convoluções | y: entrada para o atalho residual.
        """
        # Ajusto a dimensão do atalho 'y' se necessário.
        if self.skip_connection is not None:
            y = self.skip_connection(y) 

        # Processamento convolucional.
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        # Segunda convolução e soma residual (identidade do modelo ResNet).
        x = self.conv2(x) 
        x += y # A soma mágica: integra a informação original com a processada.
        
        y = x # Atualizo o estado residual para o próximo bloco.
        
        # Finalização do bloco.
        x = self.bn2(x) 
        x = self.relu(x) 
        x = self.dropout2(x) 
        return x, y 

# --- ARQUITETURA COMPLETA DA REDE ---

class ResNet1d(nn.Module):
    """
    Estrutura macro da ResNet1d. 
    Empilha vários ResBlocks de acordo com as dimensões fornecidas.
    """

    def __init__(self, input_dim, blocks_dim, n_classes, kernel_size=17, dropout_rate=0.8):
        super(ResNet1d, self).__init__() 

        # Extração de parâmetros de entrada e do primeiro nível de filtros.
        n_filters_in, n_filters_out = input_dim[0], blocks_dim[0][0] 
        n_samples_in, n_samples_out = input_dim[1], blocks_dim[0][1] 

        # Configuração da camada de entrada (Head da rede).
        downsample = _downsample(n_samples_in, n_samples_out) 
        padding = _padding(downsample, kernel_size) 
        self.conv1 = nn.Conv1d(n_filters_in, n_filters_out, kernel_size, bias=False,
                                 stride=downsample, padding=padding) 
        self.bn1 = nn.BatchNorm1d(n_filters_out) 

        # Criação dinâmica dos blocos residuais.
        self.res_blocks = [] 
        for i, (n_filters, n_samples) in enumerate(blocks_dim): 
            n_filters_in, n_filters_out = n_filters_out, n_filters 
            n_samples_in, n_samples_out = n_samples_out, n_samples 

            downsample = _downsample(n_samples_in, n_samples_out) 
            resblk1d = ResBlock1d(n_filters_in, n_filters_out, downsample, kernel_size, dropout_rate) 

            # Registro do bloco no PyTorch para que os parâmetros sejam treináveis.
            self.add_module('resblock1d_{0}'.format(i), resblk1d) 
            self.res_blocks += [resblk1d] 

        # Camada de Saída (FCN - Fully Connected Network).
        n_filters_last, n_samples_last = blocks_dim[-1] 
        last_layer_dim = n_filters_last * n_samples_last # Flatten do tensor.
        self.lin = nn.Linear(last_layer_dim, n_classes) # Camada linear para regressão da idade.

    def forward(self, x):
        """
        Fluxo completo de dados pela rede.
        """
        # Camada inicial.
        x = self.conv1(x) 
        x = self.bn1(x) 

        y = x # Ponto de partida para a primeira conexão residual.

        # Passagem por todos os blocos residuais empilhados.
        for blk in self.res_blocks: 
            x, y = blk(x, y) 

        # Flatten: converte o mapa de características 1D em um vetor para a camada linear.
        x = x.view(x.size(0), -1) 

        # Predição final.
        x = self.lin(x) 
        return x