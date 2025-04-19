import requests
from bs4 import BeautifulSoup
import pandas as pd

url = "https://en.wikipedia.org/wiki/List_of_Game_of_the_Year_awards"
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
tables = soup.find_all('table', {'class': 'wikitable'})[0:]

goty_history = []

for table in tables:
    rows = table.find_all('tr')[1:]  # Pula o cabeçalho

    for row in rows:
        cols = row.find_all('td')
        if len(cols) > 3:
            ano = cols[0].text.strip()
            jogo = cols[1].text.strip()
            genero = cols[2].text.strip()
            estudio = cols[3].text.strip()
            goty_history.append(
                {'Ano': ano,
                 'Jogo': jogo,
                 'Gênero': genero,
                 'Estúdio': estudio})

df = pd.DataFrame(goty_history)
df.to_csv('goty_history_auto.csv', index=False)
