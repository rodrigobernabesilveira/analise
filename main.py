import os
import json
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from collections import Counter
import urllib.parse
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

HISTORICO_FILE = 'historico.json'
GRAFICO_FILE = 'grafico_tendencias.png'

# --- STEP 1: Coleta de Dados do arXiv ---
CATEGORIES = "cat:cs.AI OR cat:cs.LG OR cat:math.NT"
MAX_RESULTS = 100

# Codifica os espaços e caracteres especiais para formato seguro de URL
search_query_encoded = urllib.parse.quote(CATEGORIES)

url = f'https://export.arxiv.org/api/query?search_query={search_query_encoded}&sortBy=submittedDate&sortOrder=descending&max_results={MAX_RESULTS}'

# User-Agent no formato exigido pela política de uso da API do arXiv:
# "NomeDoApp/Versao (mailto:seu_email@dominio.com)"
headers = {
    'User-Agent': 'ArXivKeywordTracker/1.0 (https://github.com/rodrigobernabesilveira/analise)'
}

req_obj = urllib.request.Request(url, headers=headers)

try:
    with urllib.request.urlopen(req_obj) as response:
        xml_data = response.read()
        root = ET.fromstring(xml_data)
except urllib.error.HTTPError as e:
    print(f"Erro HTTP ao acessar arXiv: {e.code} - {e.reason}")
    raise
	
ns = {'atom': 'http://www.w3.org/2005/Atom'}

corpus = []
for entry in root.findall('atom:entry', ns):
    title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
    summary = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')
    corpus.append(f"{title} {summary}")

# Extração de bigramas e trigramas
vectorizer = CountVectorizer(
    stop_words='english',
    ngram_range=(2, 3),
    min_df=2
)

X = vectorizer.fit_transform(corpus)
words = vectorizer.get_feature_names_out()
counts = X.sum(axis=0).A1

freq_distribution = Counter(dict(zip(words, counts)))
top_terms = dict(freq_distribution.most_common(15))

# --- STEP 2: Atualização Acumulativa do historico.json ---
hoje = datetime.now().strftime('%Y-%m-%d')

historico = []
if os.path.exists(HISTORICO_FILE):
    with open(HISTORICO_FILE, 'r', encoding='utf-8') as f:
        try:
            historico = json.load(f)
        except json.JSONDecodeError:
            historico = []

# Atualiza ou adiciona o registro do dia
registro_hoje = next((item for item in historico if item["data"] == hoje), None)
if registro_hoje:
    registro_hoje["ranking"] = top_terms
else:
    historico.append({
        "data": hoje,
        "ranking": top_terms
    })

with open(HISTORICO_FILE, 'w', encoding='utf-8') as f:
    json.dump(historico, f, indent=2, ensure_ascii=False)

print(f"Histórico atualizado com sucesso para {hoje}.")

# --- STEP 3: Geração do Gráfico de Tendências (Matplotlib) ---
if len(historico) > 0:
    datas = [entry["data"] for entry in historico]
    
    # Pega os 5 termos mais frequentes do dia mais recente para destacar no gráfico
    ultimos_termos = historico[-1]["ranking"]
    top_5_destaque = list(ultimos_termos.keys())[:5]

    plt.figure(figsize=(10, 6))

    for termo in top_5_destaque:
        valores = [entry["ranking"].get(termo, 0) for entry in historico]
        plt.plot(datas, valores, marker='o', linewidth=2, label=termo)

    plt.title('Evolução das Tendências de Termos no arXiv', fontsize=14)
    plt.xlabel('Data', fontsize=12)
    plt.ylabel('Frequência de Citações', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title='Termos em Destaque', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    plt.savefig(GRAFICO_FILE, dpi=300)
    print(f"Gráfico atualizado e salvo como '{GRAFICO_FILE}'.")