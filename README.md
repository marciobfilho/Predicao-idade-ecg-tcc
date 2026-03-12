Replicação: Predição de Idade via ECG e seu uso como Preditor de Mortalidade
Este repositório contém a replicação do estudo que utiliza Redes Neurais Profundas (ResNet1d) para estimar a idade cronológica de pacientes através de sinais de Eletrocardiograma (ECG). O projeto foi desenvolvido como Trabalho de Conclusão de Curso (TCC) em Engenharia de Computação na Universidade Feevale.

Resumo do Projeto
A predição de idade via ECG permite identificar o envelhecimento precoce do sistema cardiovascular, servindo como um biomarcador para risco de mortalidade. Minha replicação validou a arquitetura residual original, utilizando PyTorch para processar grandes volumes de dados biomédicos e validar a eficácia da rede neural na predição etária precisa.

Documentação Acadêmica
O trabalho completo, incluindo fundamentação teórica, metodologia detalhada e análise estatística dos resultados, está disponível no repositório institucional:


TCC (PDF): REPLICAÇÃO DO ESTUDO DA UTILIZAÇÃO DE INTELIGÊNCIA ARTIFICIAL NA ANÁLISE DE EXAMES DE ECG DE 12 DERIVAÇÕES 

Referência Científica
Este projeto baseia-se no artigo publicado na Nature Communications:

Lima, E.M., Ribeiro, A.H., Paixão, G.M.M. et al. Deep neural network-estimated electrocardiographic age as a mortality predictor. Nat Commun 12, 5117 (2021). DOI: 10.1038/s41467-021-25351-7

Estrutura do Repositório
O pipeline de Machine Learning está organizado nos seguintes módulos:


resnet.py: Definição da arquitetura residual profunda (ResNet1d) adaptada para sinais de ECG. 


train.py: Script principal para orquestração do treinamento e otimização. 


dataloader.py: Implementação otimizada do BatchDataloader para leitura de arquivos de alta densidade HDF5. 


formulate_problem.py: Lógica de particionamento de dados (Train/Val/Test) e análise da distribuição demográfica. 


evaluate_.py: Módulo de inferência para aplicação do modelo em novos datasets. 


plot_learning_curves.py: Ferramenta de análise visual de métricas (MAE/RMSE) e decaimento do Learning Rate. 


teste_cuda.py: Script de diagnóstico para validação de aceleração por GPU. 

Tecnologias Utilizadas

PyTorch: Framework principal de Deep Learning. 


H5py: Manipulação de datasets HDF5. 


Pandas/NumPy: Engenharia de atributos e processamento numérico. 


Matplotlib/Seaborn: Visualização científica de dados. 

Acesso aos Dados
Para reprodução dos experimentos, é necessário obter as coortes originais:

CODE-15% (Test Set): Zenodo DOI: 10.5281/zenodo.4916206

SaMi-Trop (Validação Externa): Zenodo DOI: 10.5281/zenodo.4905618

Instruções de Execução
Instalação de Dependências:

Bash
pip install -r requirements.txt
Validação de Ambiente (GPU):

Bash
python teste_cuda.py
Treinamento do Modelo:

Bash
python train.py --cuda --epochs 70 caminho/para/sinais.hdf5 caminho/para/labels.csv
Reconhecimento

Autor: Marcio Jose Gomes Bastos Filho (Engenheiro de Computação) 


Prêmio: Recebedor do Mérito Acadêmico - Universidade Feevale. 

Orientadora: Dra. Marta Rosecler Bez.