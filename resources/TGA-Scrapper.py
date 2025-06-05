import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

# ✅ Gera automaticamente os anos de 2014 a 2024
urls = {ano: f"https://en.wikipedia.org/wiki/The_Game_Awards_{ano}" for ano in range(2014, 2025)}


# 🔧 Função de limpeza de textos
def limpar_texto(txt):
    if not txt:
        return ""
    txt = re.split(r'‡', txt)[0]
    return txt.strip()


# 🎯 Função para extrair dados do GOTY (nome, desenvolvedor, vencedor)
def extrair_goty(ano, url):
    resultados = []

    print(f"\n🔎 Processando GOTY {ano}...")

    resp = requests.get(url)
    if resp.status_code != 200:
        print(f"❌ Erro ao acessar {url}")
        return resultados

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

                jogo_tag = item.find("i")
                if not jogo_tag:
                    continue

                jogo_b_tag = jogo_tag.find("b")
                vencedor = bool(jogo_b_tag) or ("‡" in texto_completo)

                nome_jogo = jogo_b_tag.text.strip() if jogo_b_tag else jogo_tag.text.strip()

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

            print(f"✅ Ano {ano} processado com sucesso!")
            break

    return resultados


# 🔥 Função para contar indicações e premiações no ano
def contabilizar_indicacoes_premiacoes(df, ano, url):
    print(f"📊 Contabilizando indicações e premiações {ano}...")

    resp = requests.get(url)
    if resp.status_code != 200:
        print(f"❌ Erro ao acessar {url}")
        return df

    soup = BeautifulSoup(resp.content, "html.parser")
    tabela = soup.find("table", {"class": "wikitable"})
    if not tabela:
        print(f"❌ Tabela não encontrada para {ano}")
        return df

    # 🎯 Jogos indicados (em <i>)
    titulos_jogos = [i.get_text(separator=" ").strip().lower() for i in tabela.find_all("i")]

    # 🏆 Jogos vencedores (em <b>)
    vencedores_tags = tabela.find_all("b")
    vencedores_texto = [v.get_text(separator=" ").strip().lower() for v in vencedores_tags]

    for idx, row in df[df["Ano"] == ano].iterrows():
        nome_jogo = row["Jogo"].lower()

        # 📈 Conta indicações apenas pelo nome em <i>
        indicacoes = titulos_jogos.count(nome_jogo)

        # 🥇 Conta premiações (aparece em <b>)
        premiacoes = sum(1 for v in vencedores_texto if nome_jogo in v)

        df.at[idx, "Indicacoes"] = indicacoes
        df.at[idx, "Premiacoes"] = premiacoes

    return df


# 🚀 Pipeline principal
todos_resultados = []

for ano, url in urls.items():
    resultados_ano = extrair_goty(ano, url)
    todos_resultados.extend(resultados_ano)

# 🧠 Cria DataFrame inicial
df = pd.DataFrame(todos_resultados)

# 🔗 Transforma desenvolvedores em arrays (quando tem múltiplos separados por "/")
df["Desenvolvedor"] = df["Desenvolvedor"].apply(lambda x: [d.strip() for d in x.split("/")])

# 🏆 Adiciona colunas de Indicações e Premiações
df["Indicacoes"] = 0
df["Premiacoes"] = 0

for ano, url in urls.items():
    df = contabilizar_indicacoes_premiacoes(df, ano, url)

# 💾 Exporta CSV final
df.to_csv("the_game_awards_goty.csv", index=False, encoding="utf-8")

print("\n✅ CSV final gerado com sucesso: the_game_awards_goty.csv")

