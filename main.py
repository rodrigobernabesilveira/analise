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

# 1. CORREÇÃO: Usar a URL base oficial da API
base_url = 'http://arxiv.org'

# 2. CORREÇÃO: Montar os parâmetros de forma que os dois-pontos (:) não quebrem
# Passamos safe=':' para que o urllib não converta o ':' em '%3A'
query_params = {
    'search_query': CATEGORIES,
    'sortBy': 'submittedDate',
    'sortOrder': 'descending',
    'max_results': MAX_RESULTS
}
encoded_params = urllib.parse.urlencode(query_params, safe=':')
url = f"{base_url}?{encoded_params}"

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'application/atom+xml, application/xml, text/xml, */*',
    'Accept-Language': 'en-US,en;q=0.9'
}

req_obj = urllib.request.Request(url, headers=headers)# --- STEP 2: Extração dos Resumos/Títulos do XML ---

NAMESPACE = {'atom': 'http://www.w3.org/2005/Atom'}

corpus = []
for entry in root.findall('atom:entry', NAMESPACE):
    summary = entry.find('atom:summary', NAMESPACE)
    title = entry.find('atom:title', NAMESPACE)
    
    texto = ""
    if title is not None and title.text:
        texto += title.text + " "
    if summary is not None and summary.text:
        texto += summary.text
        
    if texto.strip():
        # Limpa quebras de linha e espaços extras
        texto_limpo = " ".join(texto.split())
        corpus.append(texto_limpo)

print(f"Total de documentos coletados para o corpus: {len(corpus)}")

# --- STEP 3: Validação de Segurança e Extração de N-grams ---
if not corpus:
    print("Aviso: Nenhum documento retornado pelo arXiv para esta busca. Encerrando execução sem atualizar histórico.")
    exit(0)

# Configura o vectorizer ignorando palavras vazias/curtas
vectorizer = CountVectorizer(
    stop_words='english',
    ngram_range=(2, 3),  # Bigramas e Trigramas
    min_df=1             # Aceita termos que apareçam ao menos 1 vez
)

try:
    X = vectorizer.fit_transform(corpus)
except ValueError as e:
    print(f"Erro ao extrair vocabulário: {e}")
    print("Tentando fallback sem limite de n-gramas rigoroso...")
    # Fallback para unigramas + bigramas caso os textos sejam muito curtos
    vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 2), min_df=1)
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