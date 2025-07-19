# Alzar AI 🎮🏆
<p>This project is for my Artificial Intelligence class. It’s a GOTY (Game of the Year) predictor designed to guess the next award-winning game using AI.</p>
<p>"Alzar AI" is named after Alzara, the fortune-teller character from the 2025 game Blue Prince.</p>
<br>

## 📁 Project Structure
```bash
├── Alzar-AI/
│   ├── data/
│   │   ├── Alzar_AI_Base_de_Dados.csv
│   │   └── Alzar_AI_Base_de_Dados.xlsx
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
Run `alzar_ai.py` to train the models, evaluate their perfomance, select the best one, and save the generated artifacts (trained model, scaler, and prediction results CSV).
```bash
python scripts/alzar_ai.py
```
This script will print progress and classification reports to the console. Upon completion, `melhor_modelo_goty.pkl` and `scaler_goty.pkl` will be saved in the `models/` folder, and `resultados_previsao_goty.csv` in the `results/` folder.

## 📊 Analysis and Results
The `alzar_ai.py` executes the following steps:
* **Data preps**: Reads the CSV, handles null values, and prepares the columns.
* **Model Comparison**: Trains and evaluates Logistic Regression, Random Forest, and Gradient Boosting. For each model, the following are displayed:
  * Accuracy
  * Classification Report (Precision, Recall, F1-Score)
  * Confusion Matrix (visualization)
* **Best Model Selection and Saving**: The model with the highest accuracy on the test set is selected and saved as `models/melhor_modelo_goty.pkl`, along with `models/scaler_goty.pkl`.
* **Prediction on the Entire Dataset**: The saved model is loaded and used to predict the probability of winning for each game in the complete dataset. The results are displayed in the console and saved in `results/resultados_previsao_goty.csv`.

## 🤝 Contributions
Feel free to open issues or pull requests if you have suggestions, improvements, or encounter any problems.

<p>Made by: <a href="https://github.com/laosioor">Aloísio</a> & <a href="https://github.com/RezeScarlet">Clarisse</a>.</p>
