# ==============================================================================
# PROJETO: Predição de Idade via ECG (TCC Engenharia de Computação - Feevale)
# AUTOR: Marcio Jose Gomes Bastos Filho
# DESCRICAO: Script de pré-processamento e divisão de dados (Data Splitting).
#            Realiza a separação dos exames em conjuntos de treino, validação e
#            teste, garantindo que exames do mesmo paciente não apareçam em
#            grupos diferentes (evitando vazamento de dados).
# DEPENDENCIAS: pandas, numpy, seaborn, matplotlib
# INSTRUCOES: Executar via terminal fornecendo o arquivo CSV de metadados.
#             Ex: python formulate_problem.py dados.csv --splits 0.15 0.05
# ==============================================================================

import pandas as pd # Importo a biblioteca pandas para ler arquivos CSV com os metadados.
import numpy as np # Importo a biblioteca numpy para manipular arrays e realizar operacoes matematicas.

# --- LÓGICA DE DIVISÃO DOS DADOS ---

def get_splits(age_at_exam, patient_ids, exam_ids, splits, min_age_valid=16, max_age_valid=85, seed=0):
    """
    Divide o dataset de forma estratificada por paciente e prioriza faixas etárias.
    """
    rng = np.random.RandomState(seed) # Gerador com seed fixa para reprodutibilidade dos splits.

    # Validação inicial: a soma das proporções não pode exceder 100%.
    if sum(splits) > 1.0:
        raise ValueError('As divisoes devem ser somadas em um numero menor que um.') 

    n_exams = len(exam_ids) 
    patients = np.unique(patient_ids) # Identifico os indivíduos únicos na base.
    n_patients = len(patients) 

    # Mapeamentos (Hashing) para busca rápida de índices por ID.
    hash_exams = dict(zip(exam_ids, range(n_exams))) 
    hash_patients = dict(zip(patients, range(n_patients))) 
    inverse_hash_patients = dict(zip(range(n_patients), patients)) 

    # Agrupamento: Associa cada paciente à sua lista de exames realizados.
    patient_exams = [[] for _ in range(n_patients)] 
    for exam_idx in range(n_exams): 
        patient_idx = hash_patients[patient_ids[exam_idx]]
        patient_exams[patient_idx].append(exam_ids[exam_idx]) 

    # Análise de idades únicas presentes na base.
    ages, _, _ = np.unique(age_at_exam, return_inverse=True, return_counts=True) 

    # Seleção representativa: Escolhe um único exame aleatório por paciente para 
    # definir sua idade de referência no agrupamento por faixas etárias.
    patient_idx_per_age = {a: [] for a in ages} 
    patient_single_exam = np.zeros(n_patients, dtype=int) 

    for patient_idx in range(n_patients): 
        id_exam = rng.choice(patient_exams[patient_idx]) 
        patient_single_exam[patient_idx] = id_exam 
        a = age_at_exam[hash_exams[id_exam]]
        patient_idx_per_age[a].append(patient_idx) 

    # Cálculo da quantidade absoluta de pacientes em cada partição (Test/Val/Train).
    n_splits = [int(np.floor(s * n_patients)) for s in splits] 
    n_splits += [n_patients - sum(n_splits)] 

    # Embaralhamento para garantir aleatoriedade dentro das idades.
    rng.shuffle(ages) 
    for a, patient_idx in patient_idx_per_age.items():
        rng.shuffle(patient_idx) 
    
    # Montagem da lista prioritária: Primeiro pacientes na faixa etária válida (16-85 anos).
    all_patient_idx = [] 
    
    # Loop para selecionar pacientes dentro da faixa etária saudável/válida para os splits.
    for a in ages[(ages >= min_age_valid) & (ages <= max_age_valid)]:
        while patient_idx_per_age[a]:
            all_patient_idx.append(patient_idx_per_age[a].pop())

    # Inclusão dos demais pacientes (ex: muito jovens ou muito idosos) no restante da lista.
    for a in ages[(ages < min_age_valid) | (ages > max_age_valid)]:
        while patient_idx_per_age[a]:
            all_patient_idx.append(patient_idx_per_age[a].pop())

    # Alocação final dos IDs de pacientes e exames nas listas de retorno por split.
    patients_in_splits = [[] for _ in n_splits] 
    single_exam_in_split = [[] for _ in n_splits] 
    exams_in_splits = [[] for _ in n_splits] 

    for i, patient_idx in enumerate(all_patient_idx): 
        last_n = 0
        for s, n in enumerate(np.cumsum(n_splits)): 
            if last_n <= i < n: 
                patients_in_splits[s].append(inverse_hash_patients[patient_idx]) 
                single_exam_in_split[s].append(patient_single_exam[patient_idx]) 
                exams_in_splits[s] += patient_exams[patient_idx] 
            last_n = n

    return patients_in_splits, single_exam_in_split, exams_in_splits 

# --- INTERFACE DE LINHA DE COMANDO E VISUALIZAÇÃO ---

if __name__ == "__main__":
    import argparse 
    import warnings 

    parser = argparse.ArgumentParser(description='Gera sumário e divisões de dados para o TCC.') 
    parser.add_argument('file', help='Arquivo CSV de entrada.') 
    parser.add_argument('--exam_id_col', default='N_exame', help='Coluna do ID do exame.') 
    parser.add_argument('--age_col', default='Idade', help='Coluna da idade.') 
    parser.add_argument('--patient_id_col', default='N_paciente_univoco', help='Coluna do ID do paciente.') 
    parser.add_argument('--splits', default=[0.15, 0.05], nargs='*', type=float, help='Proporções dos splits.') 
    parser.add_argument('--splits_names', default=['Teste', 'Validação', 'Treino'], nargs='*', type=str) 
    parser.add_argument('--no_plot', action='store_true', help='Desativa gráficos.') 

    args, unk = parser.parse_known_args() 
    if unk: warnings.warn(f"Argumentos desconhecidos: {unk}")

    # Carregamento do CSV com separador ponto-e-vírgula (padrão comum em exports brasileiros).
    df = pd.read_csv(args.file, low_memory=False, sep=';') 
    print(f"Total de registros: {len(df)}") 
    print(f"Total de pacientes únicos: {df[args.patient_id_col].nunique()}") 

    # Limpeza: Remove duplicatas de exames para garantir unicidade.
    df.drop_duplicates(args.exam_id_col, inplace=True) 

    # Execução da divisão.
    p_splits, s_splits, e_splits = get_splits(
        np.array(df[args.age_col]), 
        np.array(df[args.patient_id_col], dtype=int), 
        np.array(df[args.exam_id_col], dtype=int), 
        args.splits
    )

    # --- PLOTAGEM DOS RESULTADOS ---

    if not args.no_plot: 
        import seaborn as sns 
        import matplotlib.pyplot as plt 

        n = len(args.splits) + 1 
        fig, ax = plt.subplots(nrows=n, figsize=(10, 5 * n)) 

        # Gera histogramas para cada conjunto gerado.
        for i in range(n): 
            age_single = np.array(df[df[args.exam_id_col].isin(s_splits[i])][args.age_col])
            age_all = np.array(df[df[args.exam_id_col].isin(e_splits[i])][args.age_col])

            title = f"{args.splits_names[i]}: {len(p_splits[i])} Pacientes, {len(e_splits[i])} Exames" 
            
            # Comparação visual entre "Todos os Exames" vs "Exames Únicos por Paciente".
            sns.histplot(age_all, ax=ax[i], kde=False, bins=range(0, 100, 1), label='Todos Exames') 
            sns.histplot(age_single, ax=ax[i], kde=False, bins=range(0, 100, 1), color='red', alpha=0.5, label='1 Exame/Paciente') 
            
            ax[i].set_title(title) 
            ax[i].legend() 
            ax[i].set_xlabel("Idade (anos)") 
            ax[i].set_ylabel("Frequência") 

        plt.tight_layout() 
        plt.show()