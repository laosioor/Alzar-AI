# Alzar AI 🎮🏆
<p>This project is for my Artificial Intelligence class. It’s a GOTY (Game of the Year) predictor designed to guess the next award-winning game using AI.</p>
<p>"Alzar AI" is named after Alzara, the fortune-teller character from the 2025 game Blue Prince.</p>
<br>

## 📁 Project Structure
```bash
├── Alzar-AI/
│   ├── data/
│   │   └── Alzar_AI_Base_de_Dados.csv
│   ├── models/
│   │   ├── melhor_modelo_goty.pkl
│   │   └── scaler_goty.pkl
│   ├── scripts/
│   │   └── alzar_ai.py
│   ├── results/
│   │   └── resultados_previsao_goty.csv
│   ├── .gitignore
│   ├── README.md (this one that you are reading right now)
│   ├── requirements.txt
```

## 🚀 Usage & Installation
Follow the steps to configure and execute the project on your own computer.

### Pre-requisites
Certify to have Python 3.8+ installed.

### 1. Clone the repo
```bash
git clone https://github.com/laosioor/Alzar-AI.git
cd Alzar-AI
```

### 2. Install dependencies
Install all the libraries required that are listed on `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 3. Preparing data
Make sure `Alzar_AI_Base_de_Dados.csv` is located on `data/`.

### 4. Executing...
Run `alzar_ai.py` as usual:
```bash
python scripts/alzar_ai.py
```

## 📊 Analysis and Results
The `alzar_ai.py` script executes the following steps:
* **Data preps**: Reads the CSV, handles null values, and prepares the columns.
* **Exploratory Data Analysis (EDA)**: Generates visualizations to understand the impact of 'Nominations', 'Metacritic Score', 'Total Composite Review', and 'Hype' on winning GOTY. The graphs will be displayed during execution.
* **Correlation Analysis**: Displays a correlation matrix between the input variables and the 'Winner' variable.
* **Model Comparison**: Trains and evaluates Logistic Regression, Random Forest, and Gradient Boosting. For each model, the following are displayed:
   * Accuracy
   * Classification Report (Precision, Recall, F1-Score)
   * Confusion Matrix (visualization)
* **Best Model Selection and Saving**: The model with the highest accuracy on the test set is selected and saved as `models/melhor_modelo_goty.pkl`, along with `models/scaler_goty.pkl`.
* **Prediction on the Entire Dataset**: The saved model is loaded and used to predict the probability of winning for each game in the complete dataset. The results are displayed in the console and saved in `results/resultados_previsao_goty.csv`.

## 🤝 Contributions
Feel free to open issues or pull requests if you have suggestions, improvements, or encounter any problems.

<p>Made by: <a href="https://github.com/laosioor">Aloísio</a> & <a href="https://github.com/RezeScarlet">Clarisse</a>.</p>
