# ==============================================================================
# PROJETO: Predição de Idade via ECG (TCC Engenharia de Computação - Feevale)
# AUTOR: Marcio Jose Gomes Bastos Filho
# DESCRICAO: Script de Avaliação e Inferência. Carrega um modelo pré-treinado 
#            e gera predições de idade para um dataset de ECGs em formato HDF5.
# DEPENDENCIAS: torch, h5py, pandas, numpy, tqdm, resnet_marcio.py
# INSTRUCOES: Executar via terminal passando a pasta do modelo e o arquivo HDF5.
#             Ex: python evaluate_.py ./model/ ./dados/test_ECG.hdf5 --output resultados.csv
# ==============================================================================

# =======================
# IMPORTACAO DE MODULOS
# =======================

from resnet_marcio import ResNet1d # Importo a arquitetura ResNet1d definida no arquivo resnet_marcio.
import tqdm # Uso tqdm para mostrar a barra de progresso durante os loops de inferência.
import h5py # Uso h5py para abrir e manipular arquivos HDF5 de forma eficiente.
import torch # Importo o PyTorch para lidar com tensores e execução na GPU.
import os # Uso os para manipular caminhos de diretórios e arquivos.
import json # Leio arquivos JSON para recuperar os hiperparâmetros do modelo.
import numpy as np # Uso NumPy para operações vetoriais e armazenamento das predições.
import argparse # Uso argparse para interface de linha de comando.
from warnings import warn # Emito avisos caso argumentos desconhecidos sejam passados.
import pandas as pd # Uso pandas para exportar os resultados finais em formato CSV.

# =======================
# BLOCO PRINCIPAL
# =======================
 
if __name__ == "__main__":
    # Configuro o parser para capturar os argumentos necessários para a execução.
    parser = argparse.ArgumentParser(add_help=False) 

    # Caminho da pasta que contém o arquivo 'model.pth' e 'config.json'.
    parser.add_argument('mdl', type=str, help='Pasta contendo o modelo e configurações.') 

    # Caminho do arquivo HDF5 contendo os sinais de ECG para teste.
    parser.add_argument('path_to_traces', type=str, default='../data/ecg_tracings.hdf5',
                        help='Caminho para o arquivo HDF5 com os traçados.') 

    # Tamanho do batch: quantos exames processar simultaneamente na memória.
    parser.add_argument('--batch_size', type=int, default=8, help='Número de exames por batch.') 

    # Nome do arquivo CSV onde as idades preditas serão salvas.
    parser.add_argument('--output', type=str, default='predicted_age.csv', help='Arquivo de saída.') 

    # Nome do dataset interno do HDF5 onde estão os sinais.
    parser.add_argument('--traces_dset', default='tracings', help='Nome do dataset de sinais no HDF5.')

    # Nome do dataset interno do HDF5 onde estão os identificadores únicos (opcional).
    parser.add_argument('--ids_dset', help='Nome do dataset de IDs no HDF5.')

    # Processo os argumentos fornecidos pelo usuário.
    args, unk = parser.parse_known_args() 
    if unk:
        warn("Argumentos desconhecidos ignorados: " + str(unk) + ".") 

    # Seleção automática de hardware: GPU (CUDA) se disponível, caso contrário CPU. 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 

    # =======================
    # CARREGAMENTO DO MODELO
    # =======================

    # Carrego o checkpoint (.pth) contendo os pesos sinápticos treinados.
    ckpt = torch.load(
        os.path.join(args.mdl, 'model.pth'), 
        map_location=lambda storage, loc: storage, # Mapeamento dinâmico para o hardware atual.
        weights_only=False 
    )

    # Leio o arquivo de configuração para reconstruir a arquitetura idêntica à do treino.
    config_path = os.path.join(args.mdl, 'config.json') 
    with open(config_path, 'r') as f:
        config_dict = json.load(f) 

    # O modelo foi treinado com o padrão de 12 derivações clássicas.
    N_LEADS = 12 

    # Instancio a estrutura da ResNet1d com os hiperparâmetros recuperados.
    model = ResNet1d(
        input_dim=(N_LEADS, config_dict['seq_length']), 
        blocks_dim=list(zip(config_dict['net_filter_size'], config_dict['net_seq_lengh'])), 
        n_classes=1, # Saída única (regressão da idade).
        kernel_size=config_dict['kernel_size'], 
        dropout_rate=config_dict['dropout_rate'] 
    )

    # Injeto os pesos aprendidos no modelo instanciado.
    model.load_state_dict(ckpt["model"]) 

    # Transfiro o modelo para o hardware de execução (GPU ou CPU).
    model = model.to(device) 

    # =======================
    # CARREGAMENTO DOS DADOS
    # =======================

    # Abro o arquivo HDF5 em modo de leitura.
    ff = h5py.File(args.path_to_traces, 'r') 

    # Acesso o ponteiro para os sinais de ECG (sem carregar tudo na RAM ainda).
    traces = ff[args.traces_dset] 
    n_total = len(traces) # Quantidade total de exames a serem processados.

    # Gestão de Identificadores: uso os IDs do arquivo ou crio uma sequência numérica.
    if args.ids_dset:
        ids = ff[args.ids_dset] 
    else:
        ids = range(n_total) 

    # Coloco o modelo em modo 'eval' (desabilita Dropout para garantir predições determinísticas).
    model.eval() 

    # Capturo as dimensões para o loop: total de exames, pontos por sinal e derivações.
    n_total, n_samples, n_leads = traces.shape 

    # Calculo o número total de iterações necessárias com base no batch_size.
    n_batches = int(np.ceil(n_total / args.batch_size)) 

    # Vetor NumPy para armazenar o resultado final das predições.
    predicted_age = np.zeros((n_total,)) 
    end = 0 

    # =======================
    # LOOP DE INFERENCIA
    # =======================

    for i in tqdm.tqdm(range(n_batches)):
        start = end 
        end = min((i + 1) * args.batch_size, n_total) 

        # Otimização: desativo o cálculo de gradientes para economizar memória e tempo.
        with torch.no_grad(): 
            # Leio o batch do HDF5, converto para tensor e transponho para (Batch, Canais, Seq).
            x = torch.tensor(traces[start:end, :, :]).transpose(-1, -2) 

            # Ajusto o tipo de dado para float32 e envio para a GPU.
            x = x.to(device, dtype=torch.float32) 

            # Realizo a passagem direta (Forward Pass) para obter a predição.
            y_pred = model(x) 

        # Removo o resultado do fluxo computacional e salvo no vetor de saída.
        predicted_age[start:end] = y_pred.detach().cpu().numpy().flatten() 

    # =======================
    # SALVO OS RESULTADOS
    # =======================

    # Organizo os IDs e as Idades Preditas em uma tabela (DataFrame).
    df_results = pd.DataFrame({'ids': ids, 'predicted_age': predicted_age}) 
    df_results = df_results.set_index('ids') 

    # Exportação para CSV para posterior análise estatística ou visualização.
    df_results.to_csv(args.output) 
    print(f"Predições concluídas. Resultados salvos em: {args.output}")