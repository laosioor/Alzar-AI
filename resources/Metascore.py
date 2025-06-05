import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
import unicodedata

# Jogos onde não foi possível a automatização de coleta de notas pelo metacritic
EXCECOES = {
    "Hearthstone": "hearthstone-heroes-of-warcraft",
    "Ratchet & Clank: Rift Apart": "ratchet-and-clank-rift-apart",
    "God of War Ragnarök": "god-of-war-ragnarok",
    "Alan Wake 2": "alan-wake-ii"
}

def formatar_nome(jogo):
    if jogo in EXCECOES:
        return EXCECOES[jogo]
    jogo = jogo.lower()
    jogo = jogo.replace("&", "and")  # Corrige & para and
    jogo = remover_acentos(jogo)
    jogo = re.sub(r"[^\w\s-]", "", jogo)  # Remove pontuação
    jogo = jogo.replace(" ", "-")
    return jogo


def remover_acentos(txt):
    return ''.join(
        c for c in unicodedata.normalize('NFD', txt)
        if unicodedata.category(c) != 'Mn'
    )

# === Função para buscar nota no Metacritic ===
def pegar_nota_metacritic(jogo):
    jogo_url = formatar_nome(jogo)
    url = f"https://www.metacritic.com/game/{jogo_url}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
    }

    try:
        resp = requests.get(url, headers=headers)
        if resp.status_code != 200:
            print(f"[{resp.status_code}] Não encontrado: {jogo}")
            return None

        soup = BeautifulSoup(resp.text, "html.parser")

        nota_div = soup.find("div", class_=re.compile(r"c-siteReviewScore.*"))
        if nota_div:
            nota_span = nota_div.find("span")
            if nota_span:
                return nota_span.text.strip()

        print(f"Nota não encontrada: {jogo}")
        return None

    except Exception as e:
        print(f"Erro ao buscar {jogo}: {e}")
        return None

# === Lê o CSV com os jogos GOTY ===
df = pd.read_csv("the_game_awards_goty.csv")

# === Cria nova coluna com notas ===
notas = []
for jogo in df["Jogo"]:
    print(f"🔎 Buscando nota para: {jogo}")
    nota = pegar_nota_metacritic(jogo)
    notas.append(nota)
    time.sleep(1.5)  # Delay pra evitar bloqueio

df["Nota Metacritic"] = notas

# === Salva novo CSV ===
df.to_csv("the_game_awards_goty.csv", index=False, encoding="utf-8")
print("✅ CSV salvo como 'the_game_awards_goty.csv'")

