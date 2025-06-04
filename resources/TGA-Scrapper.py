import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

# Dicionário dos anos e links

urls = {ano: f"https://en.wikipedia.org/wiki/The_Game_Awards_{ano}" for ano in range(2014, 2025)}


resultados = []

def limpar_texto(txt):
    if not txt:
        return ""
    txt = re.split(r'‡', txt)[0]
    txt = txt.strip()
    return txt

for ano, url in urls.items():
    print(f"Processando {ano}...")
    resp = requests.get(url)
    if resp.status_code != 200:
        print(f"Erro ao acessar {url}")
        continue

    soup = BeautifulSoup(resp.content, "html.parser")
    tabelas = soup.find_all("table", {"class": "wikitable"})

    for tabela in tabelas:
        header = tabela.find("th")
        if header and "Game of the Year" in header.text:
            celula = tabela.find("td")
            if not celula:
                continue

            itens = celula.find_all("li")

            for item in itens:
                texto_completo = item.get_text(separator=" ").strip()

                # Verifica vencedor: se tem <b> ou se tem o caractere ‡ no texto, caso tenha significa que o jogo ganhou a premiação
                jogo_tag = item.find("i")
                if not jogo_tag:
                    continue

                jogo_b_tag = jogo_tag.find("b")
                vencedor = bool(jogo_b_tag) or ("‡" in texto_completo)

                # Nome do jogo
                nome_jogo = jogo_b_tag.text.strip() if jogo_b_tag else jogo_tag.text.strip()

                # Desenvolvedor
                partes = texto_completo.split("–")
                if len(partes) > 1:
                    desenvolvedor = partes[1].strip()
                else:
                    desenvolvedor = "Desenvolvedor não encontrado"

                resultados.append({
                    "Ano": ano,
                    "Jogo": limpar_texto(nome_jogo),
                    "Desenvolvedor": limpar_texto(desenvolvedor),
                    "Vencedor": vencedor
                })
            print(\nf"✅ Ano {ano} processado com sucesso!")
            break

# Criar DataFrame
df = pd.DataFrame(resultados)

# Transforma as células na coluna Desenvolvedor em arrays
df["Desenvolvedor"] = df["Desenvolvedor"].apply(lambda x: [d.strip() for d in x.split("/")])

# Exporta para CSV
df.to_csv("the_game_awards_goty.csv", index=False, encoding="utf-8")

print("\n✅ CSV gerado com sucesso: the_game_awards_goty.csv")

