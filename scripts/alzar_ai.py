"""
# **Alzar AI**
Este script realiza uma análise de dados completa, compara múltiplos modelos de Machine Learning e utiliza o melhor modelo para prever as chances de um jogo ser o vencedor do "Game of the Year" (GOTY) no evento The Game Awards (TGA).

Compara Regressão Logística, Floresta Aleatória e Gradient Boosting para selecionar o modelo com melhor performance.

1. Importação das Bibliotecas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib  # Biblioteca para salvar e carregar o modelo
import warnings

# Importando os modelos
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Importando as métricas
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Configurações gerais
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['figure.dpi'] = 100

"""2. Carregamento e Preparação dos Dados"""

file_path = 'data/Alzar_AI_Base_de_Dados.csv'

try:
    df = pd.read_csv(file_path)
    print(">>> Passo 2: Arquivo carregado com sucesso!")
except FileNotFoundError:
    print(f"ERRO: Arquivo não encontrado no caminho '{file_path}'.")
    df = pd.DataFrame()

if not df.empty:
    df.columns = df.columns.str.strip()
    if 'Hype' in df.columns and df['Hype'].isnull().any():
        df['Hype'].fillna(df['Hype'].median(), inplace=True)
    if 'Nota Metacritic' in df.columns and df['Nota Metacritic'].isnull().any():
        df['Nota Metacritic'].fillna(df['Nota Metacritic'].median(), inplace=True)
    if 'Indicacoes' in df.columns and df['Indicacoes'].isnull().any():
        df['Indicacoes'].fillna(df['Indicacoes'].median(), inplace=True)
    print("Dados preparados para análise.\n")

"""3. Análise Exploratória de Dados"""

print(">>> Passo 3: Análise Exploratória de Dados (AED)...")

print("\nAnalisando o impacto do Número de Indicações...")
plt.figure(figsize=(12, 8))
# Combinação de Violin Plot com Strip Plot para máxima clareza
#sns.violinplot(x='Vencedor', y='Indicacoes', data=df, inner=None, palette=['#4374B3', '#FF8C00'])
sns.stripplot(x='Vencedor', y='Indicacoes', data=df, jitter=True, size=6, color='black', alpha=0.7)
plt.title('Impacto do Nº de Indicações para Vencer o GOTY', fontsize=18, pad=20)
plt.xticks([0, 1], ['Apenas Indicado', 'Vencedor do GOTY'], fontsize=12)
plt.xlabel('')
plt.ylabel('Total de Indicações Recebidas', fontsize=12)
plt.show()

# Gráficos individuais para as outras variáveis
features_to_compare = {'Nota Metacritic': 'Nota do Metacritic', 'Review_Composto_Total': 'Review Composto Total', 'Hype': 'Nível de Hype'}
for feature, title in features_to_compare.items():
    print(f"\nAnalisando o impacto de: {title}...")
    plt.figure(figsize=(10, 7))
    sns.boxplot(x='Vencedor', y=feature, data=df, palette=['#4374B3', '#FF8C00'])
    plt.title(f'Impacto de "{title}" para Vencer o GOTY', fontsize=16)
    plt.xticks([0, 1], ['Apenas Indicado', 'Vencedor do GOTY'])
    plt.xlabel('')
    plt.ylabel(title)
    plt.show()

"""4. Análise de Correlação"""

print("\n>>> Passo 4: Matriz de Correlação...")
features_for_model = ['Indicacoes', 'Nota Metacritic', 'Review_Composto_Total', 'Hype']
cols_for_corr = features_for_model + ['Vencedor']
correlation_matrix = df[cols_for_corr].corr()
plt.figure(figsize=(10, 7))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Matriz de Correlação das Variáveis do Modelo', fontsize=16)
plt.show()

"""5. Preparação para o Modelo

Aqui há uma divisão fixa de 80/20
São 62 jogos. Portanto, 49 foram usados para treino e os outros 13 para teste.
"""

print("\n>>> Passo 5: Preparando dados para o modelo...")
X = df[features_for_model]
y = df['Vencedor']
n_test_samples = 13
X_train, y_train = X[:-n_test_samples], y[:-n_test_samples]
X_test, y_test = X[-n_test_samples:], y[-n_test_samples:]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"Dados divididos e padronizados: {len(X_train)} para treino, {len(X_test)} para teste.\n")

"""6. Comparação de Modelos de Machine Learning"""

print(">>> Passo 6: Treinando e Comparando Múltiplos Modelos...")

models = {
    "Regressão Logística": LogisticRegression(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

results = {}

for name, model in models.items():
    # Treina o modelo
    model.fit(X_train_scaled, y_train)

    # Faz as previsões no conjunto de teste
    y_pred = model.predict(X_test_scaled)

    # Calcula a acurácia
    accuracy = accuracy_score(y_test, y_pred)

    # Armazena o modelo treinado e sua acurácia
    results[name] = {'accuracy': accuracy, 'model': model}

    # Imprime os resultados para este modelo
    print("\n" + "="*40)
    print(f"Resultados para o modelo: {name}")
    print("="*40)
    print(f"Acurácia: {accuracy:.2%}")
    print("\nRelatório de Classificação:")
    print(classification_report(y_test, y_pred, target_names=['Indicado', 'Vencedor'], zero_division=0))

    # Plota a Matriz de Confusão
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Previsto: Indicado', 'Previsto: Vencedor'],
                yticklabels=['Real: Indicado', 'Real: Vencedor'])
    plt.title(f'Matriz de Confusão - {name}', fontsize=16)
    plt.show()

"""7. Seleção do Melhor Modelo"""

print("\n>>> Passo 7: Selecionando o melhor modelo...")

# Encontra o nome do modelo com a maior acurácia
best_model_name = max(results, key=lambda name: results[name]['accuracy'])
best_model_info = results[best_model_name]
best_model = best_model_info['model']

print(f"O melhor modelo foi '{best_model_name}' com uma acurácia de {best_model_info['accuracy']:.2%}.\n")

"""8. Salvando o melhor Modelo e o Scaler"""

print(">>> Passo 8: Salvando o melhor modelo e o scaler...")
joblib.dump(best_model, 'models/melhor_modelo_goty.pkl')
joblib.dump(scaler, 'models/scaler_goty.pkl')
print("Modelo salvo como 'melhor_modelo_goty.pkl'")
print("Scaler salvo como 'scaler_goty.pkl'")

"""9. Usando o melhor Modelo para analisar toda a planilha"""

print("\n>>> Passo 9: Usando o melhor modelo salvo para prever em TODA a planilha...")

loaded_model = joblib.load('melhor_modelo_goty.pkl')
loaded_scaler = joblib.load('scaler_goty.pkl')

X_full = df[features_for_model]
X_full_scaled = loaded_scaler.transform(X_full)

full_probabilities = loaded_model.predict_proba(X_full_scaled)[:, 1] * 100

full_results_df = df.copy()
full_results_df['Probabilidade de Vencer (%)'] = full_probabilities

full_results_df = full_results_df[['Jogo', 'Ano', 'Vencedor', 'Probabilidade de Vencer (%)']]
full_results_df.rename(columns={'Vencedor': 'Vencedor Real'}, inplace=True)
full_results_df = full_results_df.sort_values(by='Probabilidade de Vencer (%)', ascending=False)

full_results_df_display = full_results_df.copy()
full_results_df_display['Probabilidade de Vencer (%)'] = full_results_df_display['Probabilidade de Vencer (%)'].map('{:.2f}%'.format)

print("\n" + "="*70)
print(f"Probabilidades de Vitória (usando {best_model_name})")
print("="*70)
print(full_results_df_display.to_string())

"""10. Salvando os Resultados Finais em um novo arquivo"""

print("\n>>> Passo 10: Salvando os resultados em um arquivo CSV...")

output_csv_path = 'results/resultados_previsao_goty.csv'
full_results_df.to_csv(output_csv_path, index=False, decimal='.', float_format='%.2f')

print(f"Resultados salvos com sucesso no arquivo: '{output_csv_path}'")

print("\nAnálise concluída.")
