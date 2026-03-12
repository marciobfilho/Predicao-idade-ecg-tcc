# ==============================================================================
# PROJETO: Predição de Idade via ECG (TCC Engenharia de Computação - Feevale)
# AUTOR: Marcio Jose Gomes Bastos Filho
# DESCRICAO: Implementação do BatchDataloader. Esta classe gerencia o carregamento
#            iterativo de dados (batches), permitindo o processamento de grandes
#            volumes de sinais (ECG) e rótulos (idades) de forma eficiente.
# DEPENDENCIAS: torch, numpy, math
# INSTRUCOES: Esta classe deve ser instanciada passando os tensores de dados
#             e uma máscara booleana que define quais exemplos pertencem
#             ao conjunto atual (treino ou validação).
# ==============================================================================

import math # Importo o modulo para funcoes matematicas.
import torch # Importo o PyTorch para computacao com tensores.
import numpy as np # Importo o NumPy para operacoes com arrays.


class BatchDataloader:
    """
    Classe responsável por particionar e entregar os dados em lotes (batches).
    Funciona como um iterador, otimizando o consumo de memória.
    """
    
    # Construtor da classe. Recebo os tensores (ex: sinais, idades, pesos), 
    # o tamanho do batch (bs) e a mascara de validade.
    def __init__(self, *tensors, bs=1, mask=None):
        # Encontro os indices onde a mascara e True (elementos validos para este loader).
        nonzero_idx, = np.nonzero(mask) 
        self.tensors = tensors # Armazeno a tupla de tensores de dados.
        self.batch_size = bs # Armazeno o tamanho do lote definido no parser de argumentos.
        self.mask = mask # Armazeno a mascara booleana global.
        
        # Defino os limites de leitura para evitar iterar sobre o dataset inteiro desnecessariamente.
        if nonzero_idx.size > 0: 
            self.start_idx = min(nonzero_idx) # Primeiro indice valido (ex: inicio do split).
            self.end_idx = max(nonzero_idx)+1 # Ultimo indice valido +1 para fatiamento (slicing).
        else:
            self.start_idx = 0 
            self.end_idx = 0 

    # Metodo especial que define a lógica de entrega do proximo batch (invocado pelo loop for)
    def __next__(self):
        # Verifico se o ponteiro de leitura chegou ao fim do intervalo definido.
        if self.start == self.end_idx: 
            raise StopIteration # Encerro a iteracao da epoca.

        # Calculo o indice final pretendido para o batch atual.
        end = min(self.start + self.batch_size, self.end_idx) 
        
        # Extraio a fatia correspondente da mascara para validar os itens deste batch.
        batch_mask = self.mask[self.start:end] 
        
        # Lógica de Pulo (Skip): Se um batch inteiro for falso na mascara, avanco ate achar dados validos.
        while sum(batch_mask) == 0: 
            self.start = end 
            end = min(self.start + self.batch_size, self.end_idx) 
            batch_mask = self.mask[self.start:end] 

        # Extraio os dados brutos (fatias) de todos os tensores fornecidos no construtor.
        batch = [np.array(t[self.start:end]) for t in self.tensors] 
        
        # Avanco o ponteiro de inicio para o local onde o batch atual terminou.
        self.start = end 
        
        # Acumulo a contagem de elementos processados para controle interno.
        self.sum += sum(batch_mask) 
        
        # Converto os arrays NumPy filtrados pela mascara em tensores PyTorch prontos para a GPU.
        return [torch.tensor(b[batch_mask], dtype=torch.float32) for b in batch] 

    # Torna a classe um iterador reinicializavel.
    def __iter__(self):
        # Sempre que um novo loop 'for' inicia, volto o ponteiro para o inicio do split.
        self.start = self.start_idx 
        self.sum = 0 # Reseto o contador de elementos.
        return self 

    # Retorna a quantidade total de lotes (batches) que este loader entregara.
    def __len__(self):
        count = 0 
        start = self.start_idx 
        # Simulo a iteracao para contar quantos batches possuem ao menos um elemento valido.
        while start != self.end_idx: 
            end = min(start + self.batch_size, self.end_idx) 
            batch_mask = self.mask[start:end] 
            if sum(batch_mask) != 0: 
                count += 1 
            start = end 
        return count