# ==============================================================================
# PROJETO: Predição de Idade via ECG (TCC Engenharia de Computação - Feevale)
# AUTOR: Marcio Jose Gomes Bastos Filho
# DESCRICAO: Script de Visualização de Performance. Gera gráficos das curvas de
#            aprendizado, correlacionando o Erro Absoluto Médio (MAE) e a 
#            variação da Taxa de Aprendizado (Learning Rate) por época.
# DEPENDENCIAS: matplotlib, pandas, numpy, argparse
# INSTRUCOES: Executar via terminal passando o arquivo history.csv gerado no treino.
#             Ex: python plot_learning_curves.py ./model/history.csv --save curva.png
# ==============================================================================

import matplotlib.pyplot as plt # Importo a biblioteca para geração de gráficos.
import pandas as pd # Importo a biblioteca para manipulação e leitura do histórico em CSV.
import numpy as np # Importo a biblioteca para operações em arrays.

# Bloco executado apenas se o script for chamado diretamente via terminal
if __name__ == "__main__":
    import argparse # Módulo para tratamento de argumentos de linha de comando.

    # Configuro o parser para receber os parâmetros do usuário.
    parser = argparse.ArgumentParser(description='Script para traçar curvas de aprendizado do modelo.')

    # Caminho obrigatório para o arquivo de histórico (gerado pelo train.py).
    parser.add_argument('history_file', type=str, help="Caminho para o arquivo history.csv.")

    # Opção para aplicar estilos visuais do matplotlib (ex: ggplot, seaborn).
    parser.add_argument('--plot_style', nargs='*', default=[], help='Estilos de plotagem opcionais.')

    # Opção para salvar o gráfico diretamente em um arquivo de imagem.
    parser.add_argument('--save', default='', help='Caminho/nome do arquivo para salvar o gráfico.')

    # Realizo o parse dos argumentos fornecidos.
    args = parser.parse_args()

    # Aplico estilos visuais caso tenham sido informados.
    if args.plot_style:
        plt.style.use(args.plot_style)

    # Leitura dos dados: carrego o CSV contendo métricas por época (epoch, mae, lr).
    df = pd.read_csv(args.history_file)

    # ====================
    # CONFIGURAÇÃO DO PLOT: MAE (Eixo Y Primário)
    # ====================
    
    # Crio a base do gráfico (Figura e Eixo principal).
    fig, ax = plt.subplots()

    # Ploto a curva do Erro Absoluto Médio (MAE) em azul.
    # Somo +1 em 'epoch' apenas para o gráfico começar na Época 1 em vez de 0.
    ax.plot(df['epoch']+1, df['mae'], label='Erro de Treino (MAE)', color='blue')

    # Configurações de labels do eixo principal.
    ax.set_xlabel('Épocas de Treinamento')
    ax.set_ylabel('MAE (Erro em Anos)', color='blue')

    # ====================
    # CONFIGURAÇÃO DO PLOT: Taxa de Aprendizado (Eixo Y Secundário)
    # ====================

    # Crio um eixo Y espelhado (twinx) que compartilha o mesmo eixo X (Épocas).
    axt = ax.twinx()

    # Ploto a Taxa de Aprendizado (Learning Rate) usando o estilo 'step' (degraus).
    # Uso 'alpha' para transparência e cor preta ('k') para diferenciar do MAE.
    axt.step(df['epoch']+1, df['lr'], label='Learning Rate', alpha=0.4, color='k')

    # Defino a escala do eixo secundário como Logarítmica, pois o LR varia em ordens de grandeza.
    axt.set_yscale('log')
    axt.set_ylabel('Learning Rate (Escala Log)', alpha=0.4, color='k')

    # Ajusto os limites visuais da taxa de aprendizado conforme os padrões do scheduler.
    axt.set_ylim((1e-8, 1e-2))

    # Título do gráfico para identificação do experimento.
    plt.title('Curvas de Aprendizado: Evolução do Erro vs. Taxa de Ajuste')

    # ====================
    # FINALIZAÇÃO E EXPORTAÇÃO
    # ====================

    # Se o usuário definiu um caminho em --save, gravo o arquivo no disco.
    if args.save:
        plt.savefig(args.save, bbox_inches='tight')
        print(f"Gráfico salvo com sucesso em: {args.save}")
    else:
        # Caso contrário, apenas exibo o gráfico em uma janela interativa.
        plt.show()