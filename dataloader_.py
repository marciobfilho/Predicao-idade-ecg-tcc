# ==============================================================================
# PROJETO: Predição de Idade via ECG (TCC Engenharia de Computação - Feevale)
# AUTOR: Marcio Jose Gomes Bastos Filho
# DESCRICAO: Implementação da classe BatchDataloader para carregamento de lotes.
#            Esta classe automatiza a entrega de dados filtrados por máscaras,
#            sendo fundamental para separar os dados de treino e validação
#            mantendo a eficiência no consumo de memória.
# DEPENDENCIAS: torch, numpy, math
# INSTRUCOES: A classe deve ser utilizada como um iterador em loops de treino.
#             Ela suporta múltiplos tensores de entrada (sinais, idades, pesos).
# ==============================================================================

import math # Importo o modulo para funcoes matematicas.
import torch # Importo o PyTorch para computacao com tensores.
import numpy as np # Importo o NumPy para operacoes com arrays.

class BatchDataloader:
    """
    Dataloader customizado responsável por gerar batches de dados.
    Utiliza uma máscara booleana para filtrar quais registros do dataset
    devem ser entregues em cada iteração (ex: apenas registros de treino).
    """

    # Construtor da classe. Recebe os "tensores" (ECG, idades, pesos), 
    # o tamanho do lote (bs) e uma mascara booleana.
    def __init__(self, *tensors, bs=1, mask=None):
        # Localizo os índices onde a máscara é True para definir o intervalo de dados úteis.
        nonzero_idx, = np.nonzero(mask) 

        # Armazeno os tensores brutos (geralmente referências a arquivos HDF5 ou arrays NumPy).
        self.tensors = tensors 
    
        # Defino a quantidade de amostras por iteração (batch size).
        self.batch_size = bs 

        # Armazeno a máscara que define o split (Treino, Validação ou Teste).
        self.mask = mask 

        # Otimização: Defino o primeiro e o último índice válido para evitar percorrer o dataset inteiro.
        if nonzero_idx.size > 0:
            self.start_idx = min(nonzero_idx) 
            self.end_idx = max(nonzero_idx) + 1 # +1 para compatibilidade com fatiamento (slicing) do Python.
        else:
            self.start_idx = 0 
            self.end_idx = 0 

    # Método de iteração: Define como o próximo lote de dados é extraído e transformado.
    def __next__(self):
        # Condição de parada: Se o ponteiro de leitura 'start' atingiu o limite 'end_idx'.
        if self.start == self.end_idx:
            raise StopIteration 

        # Determino o fim do lote atual, garantindo que não ultrapasse o limite total.
        end = min(self.start + self.batch_size, self.end_idx) 

        # Extraio a fatia correspondente da máscara para validar os itens deste lote.
        batch_mask = self.mask[self.start:end] 

        # Caso o lote atual não contenha nenhum item validado pela máscara, pulo para o próximo.
        while sum(batch_mask) == 0: 
            self.start = end 
            end = min(self.start + self.batch_size, self.end_idx) 
            batch_mask = self.mask[self.start:end] 

        # Extraio os dados brutos de cada tensor (ECG, idade, etc.) no intervalo definido.
        batch = [np.array(t[self.start:end]) for t in self.tensors] 

        # Atualizo o ponteiro de início para a próxima chamada do iterador.
        self.start = end 

        # Acumulo a contagem de elementos processados para controle interno de progresso.
        self.sum += sum(batch_mask) 

        # Transformação final: Filtro os dados pela máscara e converto para tensores PyTorch float32.
        # Isso prepara os dados para serem processados pela GPU durante o treinamento.
        return [torch.tensor(b[batch_mask], dtype=torch.float32) for b in batch] 

    # Reinicializa o iterador. Chamado automaticamente ao iniciar um loop 'for'.
    def __iter__(self):
        self.start = self.start_idx # Reseta o ponteiro para o início do split válido.
        self.sum = 0 # Zera o contador de elementos processados.
        return self 

    # Calcula e retorna o número total de lotes que este dataloader irá entregar.
    def __len__(self):
        count = 0 
        start = self.start_idx 

        # Percorro o intervalo para contar apenas os batches que possuem ao menos um item válido.
        while start != self.end_idx: 
            end = min(start + self.batch_size, self.end_idx) 
            batch_mask = self.mask[start:end] 

            if sum(batch_mask) != 0: 
                count += 1 
            
            start = end 

        return count