# ==============================================================================
# PROJETO: Predição de Idade via ECG (TCC Engenharia de Computação - Feevale)
# AUTOR: Marcio Jose Gomes Bastos Filho
# DESCRICAO: Script principal de treinamento do modelo. Gerencia o loop de 
#            épocas, cálculo de métricas, otimização e persistência do modelo.
# DEPENDENCIAS: torch, numpy, pandas, h5py, tqdm, resnet.py, dataloader.py
# INSTRUCOES: Executar via terminal passando os caminhos do HDF5 e CSV.
#             Ex: python train.py --cuda ./dados/ecg.hdf5 ./dados/metadados.csv
# ==============================================================================

import os 
import json 
from warnings import warn 
import numpy as np 
import pandas as pd 
from tqdm import tqdm 
import torch 
import torch.optim as optim 
from resnet import ResNet1d 
from dataloader import BatchDataloader # Ajustado para o nome padrão do arquivo

# --- FUNÇÕES DE CÁLCULO E PESOS ---

def compute_loss(ages, pred_ages, weights):
    """
    Calcula a perda (Loss) utilizando o Erro Quadrático Médio Ponderado (W-MSE).
    A ponderação é essencial para lidar com desequilíbrios na distribuição das idades.
    """
    diff = ages.flatten() - pred_ages.flatten() 
    # Multiplicação do erro ao quadrado pelo peso correspondente à idade.
    loss = torch.sum(weights.flatten() * diff * diff) 
    return loss 

def compute_weights(ages, max_weight=np.inf):
    """
    Calcula pesos baseados na frequência de cada idade (Inverse Frequency Weighting).
    Idades menos frequentes ganham mais importância para evitar viés no modelo.
    """
    _, inverse, counts = np.unique(ages, return_inverse=True, return_counts=True)
    weights = 1 / counts[inverse] # Peso inversamente proporcional à contagem.
    normalized_weights = weights / np.sum(weights) 
    w = len(ages) * normalized_weights 
    
    if max_weight < np.inf: 
        w = np.minimum(w, max_weight) # Limita pesos extremos para manter a estabilidade.
        w = len(ages) * w / np.sum(w) 
    return w 

def compute_metrics(y_true, y_pred, weights=None):
    """
    Calcula métricas de desempenho: MAE (Erro Absoluto Médio) e RMSE (Raiz do Erro Quadrático Médio).
    Retorna versões ponderadas e não ponderadas.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    err = y_pred - y_true 
    
    mae = float(np.mean(np.abs(err))) 
    rmse = float(np.sqrt(np.mean(err ** 2))) 
    
    if weights is None:
        w_mae, w_rmse = mae, rmse 
    else:
        w = np.asarray(weights).reshape(-1)
        wsum = float(np.sum(w)) if float(np.sum(w)) != 0.0 else 1.0 
        w_mae = float(np.sum(w * np.abs(err)) / wsum)
        w_rmse = float(np.sqrt(np.sum(w * err ** 2) / wsum))
    return mae, rmse, w_mae, w_rmse 

# --- LOOP DE TREINO E VALIDAÇÃO ---

def train_one_epoch(ep, dataload, model, optimizer, device):
    """
    Executa uma única iteração (época) sobre todo o conjunto de treinamento.
    """
    model.train() # Habilita Dropout e Batch Normalization.
    total_loss, n_entries = 0.0, 0
    desc = "Epoch {:2d}: train - Loss: {:.6f}"
    
    pbar = tqdm(initial=0, leave=True, total=len(dataload), desc=desc.format(ep, 0.0), position=0)
    
    for traces, ages, weights in dataload:
        # Reorganiza o sinal: (Batch, Canais, Comprimento) para compatibilidade com Conv1d.
        traces = traces.transpose(1, 2) 
        traces, ages, weights = traces.to(device), ages.to(device), weights.to(device)
        
        optimizer.zero_grad(set_to_none=True) # Reseta gradientes para nova etapa.
        pred_ages = model(traces) 
        loss = compute_loss(ages, pred_ages, weights) 
        
        loss.backward() # Retropropagação dos erros.
        optimizer.step() # Atualização dos pesos sinápticos.
        
        total_loss += float(loss.detach().cpu().numpy())
        n_entries += len(traces)
        pbar.desc = desc.format(ep, total_loss / max(n_entries, 1))
        pbar.update(1)
        
    pbar.close()
    return total_loss / max(n_entries, 1)

@torch.no_grad()
def evaluate(ep, dataload, model, device):
    """
    Avalia o modelo no conjunto de validação sem atualizar os pesos.
    """
    model.eval() # Desabilita Dropout para inferência estável.
    total_loss, n_entries = 0.0, 0
    desc = "Epoch {:2d}: valid - Loss: {:.6f}"
    
    pbar = tqdm(initial=0, leave=True, total=len(dataload), desc=desc.format(ep, 0.0), position=0)
    y_true_all, y_pred_all, w_all = [], [], []
    
    for traces, ages, weights in dataload:
        traces = traces.transpose(1, 2)
        traces, ages, weights = traces.to(device), ages.to(device), weights.to(device)
        
        pred_ages = model(traces)
        loss = compute_loss(ages, pred_ages, weights)
        
        total_loss += float(loss.detach().cpu().numpy())
        n_entries += len(traces)
        pbar.desc = desc.format(ep, total_loss / max(n_entries, 1))
        pbar.update(1)
        
        y_true_all.append(ages.detach().cpu().numpy().reshape(-1))
        y_pred_all.append(pred_ages.detach().cpu().numpy().reshape(-1))
        w_all.append(weights.detach().cpu().numpy().reshape(-1))
        
    pbar.close()
    return total_loss / max(n_entries, 1), np.concatenate(y_true_all), np.concatenate(y_pred_all), np.concatenate(w_all)

# --- EXECUÇÃO PRINCIPAL ---

if __name__ == "__main__":
    import h5py 
    import argparse 

    parser = argparse.ArgumentParser(description='Treinamento do modelo de predição de idade via ECG.')
    parser.add_argument('--epochs', type=int, default=70)
    parser.add_argument('--cuda', action='store_true') # Flag para processamento em GPU.
    parser.add_argument('--folder', default='model/') # Onde salvar os resultados.
    parser.add_argument('path_to_traces', help='Caminho do arquivo HDF5.')
    parser.add_argument('path_to_csv', help='Caminho do arquivo CSV.')
    # Outros argumentos possuem defaults baseados na arquitetura ResNet original.
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--net_filter_size', type=int, nargs='+', default=[64, 128, 196, 256, 320])
    parser.add_argument('--net_seq_lengh', type=int, nargs='+', default=[4096, 1024, 256, 64, 16])
    parser.add_argument('--dropout_rate', type=float, default=0.8)
    parser.add_argument('--kernel_size', type=int, default=17)
    parser.add_argument('--n_valid', type=int, default=100)
    parser.add_argument('--age_col', default='age')

    args, unk = parser.parse_known_args()
    if unk: warn(f"Argumentos desconhecidos ignorados: {unk}")

    device = torch.device('cuda:0' if args.cuda and torch.cuda.is_available() else 'cpu')
    os.makedirs(args.folder, exist_ok=True)

    # Persistência das configurações da execução.
    with open(os.path.join(args.folder, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Carregamento e preparação dos dados.
    tqdm.write("Carregando bases de dados...")
    df = pd.read_csv(args.path_to_csv)
    ages = df[args.age_col].values
    with h5py.File(args.path_to_traces, 'r') as f:
        traces = f['tracings']
        
        # Divisão simples entre treino e validação.
        valid_mask = np.arange(len(df)) < args.n_valid
        weights = compute_weights(ages)

        train_loader = BatchDataloader(traces, ages, weights, bs=args.batch_size, mask=~valid_mask)
        valid_loader = BatchDataloader(traces, ages, weights, bs=args.batch_size, mask=valid_mask)

        # Inicialização da Rede Neural.
        model = ResNet1d(input_dim=(traces.shape[2], 4096), 
                         blocks_dim=list(zip(args.net_filter_size, args.net_seq_lengh)),
                         n_classes=1, kernel_size=args.kernel_size, dropout_rate=args.dropout_rate)
        model.to(device)

        optimizer = optim.Adam(model.parameters(), args.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=7, factor=0.1)

        # Histórico de treinamento.
        history_path = os.path.join(args.folder, "history.csv")
        best_val = float('inf')

        # --- LOOP PRINCIPAL DE ÉPOCAS ---
        for ep in range(args.epochs):
            train_loss = train_one_epoch(ep, train_loader, model, optimizer, device)
            valid_loss, y_true, y_pred, w_val = evaluate(ep, valid_loader, model, device)
            mae, rmse, w_mae, w_rmse = compute_metrics(y_true, y_pred, w_val)

            # Salva o melhor modelo baseado na perda de validação.
            if valid_loss < best_val:
                torch.save({"model": model.state_dict(), "valid_loss": valid_loss}, 
                           os.path.join(args.folder, 'model.pth'))
                best_val = valid_loss

            # Registro do progresso no CSV.
            log_data = pd.DataFrame([[ep, train_loss, valid_loss, mae, rmse]], 
                                    columns=['epoch', 'train_loss', 'valid_loss', 'mae', 'rmse'])
            log_data.to_csv(history_path, mode='a', header=not os.path.exists(history_path), index=False)

            tqdm.write(f"Ep {ep:2d} | Val Loss: {valid_loss:.4f} | MAE: {mae:.2f} anos")
            scheduler.step(valid_loss)

    # Salva configuração final da arquitetura.
    config = {"seq_length": args.seq_length, "filter_size": args.net_filter_size, "kernel": args.kernel_size}
    with open(os.path.join(args.folder, "config.json"), "w") as f:
        json.dump(config, f, indent=2)