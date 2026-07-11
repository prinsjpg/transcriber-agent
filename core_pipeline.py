"""
core_pipeline.py — Motore condiviso della pipeline di trascrizione.

Contiene TUTTA la logica comune ai tre agenti (chunking, RAG, grafo LangGraph,
cane da guardia, rendering HTML, revisione finale). I singoli agenti
(`agente.py`, `agente_codice.py`, `agente_esercizi.py`) si limitano a definire
una `PipelineConfig` e a chiamare `run(config)`.
"""

import os
import re
import time
import glob
import json
import difflib
import hashlib
import argparse
from dataclasses import dataclass
from typing import TypedDict

import markdown
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage
from markitdown import MarkItDown
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from bs4 import BeautifulSoup

load_dotenv()

# Prompt di correzione: identico per tutti gli agenti (raw string per il LaTeX).
SYSTEM_PROMPT_CORREZIONE = r"""Sei un revisore editoriale. Il tuo compito è correggere la trascrizione fonetica di un singolo blocco di una lezione.
    Usa il testo delle slide fornito per capire e correggere i termini tecnici storpiati.
    REGOLE: Correggi la punteggiatura ma NON tagliare assolutamente nulla, NON riassumere per nessun motivo e mantieni il 100% del parlato originale."""


class GraphState(TypedDict):
    testo_slide: str
    trascrizione_grezza: str
    trascrizione_pulita: str
    memoria_precedente: str
    documento_finale: str


@dataclass
class PipelineConfig:
    # --- Provider / modello ---
    base_url: str
    api_key_env: str
    model: str
    # --- Dati di input/output ---
    cartella_slide: str
    cartella_trascrizioni: str
    nome_output: str
    # --- Prompt di generazione (specifico per ciascun agente) ---
    system_prompt_generazione: str
    user_prompt_suffix: str = ""          # blocco recency-bias finale (opzionale)
    # --- Comportamento della pipeline ---
    include_code_files: bool = False      # il RAG legge anche i file sorgente di codice
    has_esercizio: bool = False           # estrai e gestisci il tag <esercizio>
    separatori_teoria: bool = False       # inserisci "---" tra i blocchi di teoria
    render_teoria_markdown: bool = False  # rendi la teoria con la libreria markdown
    markdown_extensions: tuple = ("fenced_code",)
    enable_code_highlight: bool = False   # includi Highlight.js nel template
    enable_exercise_css: bool = False     # includi lo stile dei box esercizio
    enable_table_css: bool = False        # includi lo stile delle tabelle
    proteggi_variabili_dollaro: bool = True  # racchiude le pseudo-variabili Yacc ($$/$1) in <code> per non collidere con MathJax
    allega_slide_immagini: bool = False   # [prototipo] incastona la slide PDF piu' rilevante come immagine sotto ogni esercizio
    dpi_slide: int = 110                  # risoluzione di rendering delle pagine PDF in PNG
    print_rag_sources: bool = False       # stampa a terminale le slide consultate
    usa_cache: bool = True                # riusa le conversioni PDF->Markdown già fatte
    cache_dir: str = ".cache/markitdown"  # cartella su disco per la cache delle slide
    # --- Retriever ---
    usa_retriever_ibrido: bool = False    # affianca a BM25 un retriever semantico (embedding)
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    peso_bm25: float = 0.5                # peso del ramo lessicale nell'ensemble
    peso_semantico: float = 0.5           # peso del ramo semantico nell'ensemble
    retriever_k: int = 3                  # numero di slide recuperate per blocco
    # --- Parametri numerici ---
    max_parole: int = 1500
    overlap_parole: int = 150
    memoria_caratteri: int = 4000
    soglia_antiripetizione: float = 0.85  # scarta un paragrafo di teoria se simile oltre questa soglia a uno già inserito (1.0 = disattiva)
    pausa_secondi: int = 15
    pausa_retry: int = 30       # attesa base del backoff esponenziale (primo fallimento)
    backoff_max: int = 240      # tetto massimo dell'attesa tra i retry
    temperature: float = 0.3
    max_tokens: int = 8192


# ==========================================================================
# INGESTION E PREPROCESSING
# ==========================================================================
def _percorso_cache(cache_dir: str, percorso_file: str) -> str:
    """Nome univoco del file di cache, derivato dal percorso assoluto della slide."""
    chiave = hashlib.md5(os.path.abspath(percorso_file).encode('utf-8')).hexdigest()
    return os.path.join(cache_dir, f"{chiave}.json")


def _leggi_cache(cache_dir: str, percorso_file: str):
    """Restituisce il Markdown in cache se il file sorgente non è cambiato (mtime + size)."""
    percorso_cache = _percorso_cache(cache_dir, percorso_file)
    if not os.path.exists(percorso_cache):
        return None
    try:
        stat = os.stat(percorso_file)
        with open(percorso_cache, 'r', encoding='utf-8') as f:
            dati = json.load(f)
        if dati.get('mtime') == stat.st_mtime and dati.get('size') == stat.st_size:
            return dati.get('text')
    except Exception:
        return None
    return None


def _scrivi_cache(cache_dir: str, percorso_file: str, testo: str):
    """Salva su disco il Markdown estratto insieme a mtime + size del sorgente."""
    try:
        os.makedirs(cache_dir, exist_ok=True)
        stat = os.stat(percorso_file)
        with open(_percorso_cache(cache_dir, percorso_file), 'w', encoding='utf-8') as f:
            json.dump({'mtime': stat.st_mtime, 'size': stat.st_size, 'text': testo}, f)
    except Exception:
        pass


def estrai_materiale_didattico(cartella: str, include_code: bool = False,
                               usa_cache: bool = True, cache_dir: str = ".cache/markitdown") -> list[Document]:
    """
    Converte slide PDF/PPTX/DOCX/XLSX in Markdown tramite MarkItDown e,
    opzionalmente, allega i file di codice sorgente per il RAG. Le conversioni
    vengono memorizzate su disco e riusate se il file sorgente non è cambiato.
    """
    documenti = []
    md = MarkItDown()

    estensioni_doc = ['*.pdf', '*.pptx', '*.docx', '*.xlsx']
    file_doc = []
    for est in estensioni_doc:
        file_doc.extend(glob.glob(f"{cartella}/{est}"))

    if not file_doc and not include_code:
        print(f"[!] Nessun file di supporto trovato nella cartella '{cartella}'.")

    for percorso in file_doc:
        nome_file = os.path.basename(percorso)

        testo_cache = _leggi_cache(cache_dir, percorso) if usa_cache else None
        if testo_cache is not None:
            print(f"    - Slide da cache: {nome_file}")
            testo_cache = _normalizza_simboli(testo_cache)
            documenti.append(Document(page_content=f"--- FONTE: {nome_file} ---\n{testo_cache}"))
            continue

        print(f"    - Conversione slide in Markdown: {nome_file}...")
        try:
            risultato = md.convert(percorso)
            if risultato.text_content:
                if usa_cache:
                    _scrivi_cache(cache_dir, percorso, risultato.text_content)
                testo_slide = _normalizza_simboli(risultato.text_content)
                documenti.append(Document(page_content=f"--- FONTE: {nome_file} ---\n{testo_slide}"))
        except Exception as e:
            print(f"    [!] Errore conversione {nome_file}: {e}")

    if include_code:
        documenti.extend(_estrai_file_codice(cartella))

    return documenti


def _estrai_file_codice(cartella: str) -> list[Document]:
    """Allega al RAG i file di codice sorgente presenti nella cartella delle slide."""
    documenti = []
    estensioni_codice = ['*.py', '*.js', '*.html', '*.java', '*.cpp', '*.c', '*.txt', '*.md']
    file_codice = []
    for est in estensioni_codice:
        file_codice.extend(glob.glob(f"{cartella}/{est}"))
    for percorso in file_codice:
        nome_file = os.path.basename(percorso)
        try:
            with open(percorso, 'r', encoding='utf-8', errors='ignore') as f:
                contenuto = f.read()
                documenti.append(Document(page_content=f"--- SORGENTE CODICE: {nome_file} ---\n{contenuto}"))
        except Exception:
            pass
    return documenti


def estrai_slide_per_pagina(cartella: str, include_code: bool = False) -> list[Document]:
    """
    Variante di ingestione a granularità di PAGINA per i soli PDF: produce un
    Document per ogni pagina, con il testo estratto (via PyMuPDF) e i metadati
    (percorso, numero di pagina, nome file) che servono poi a rendere quella
    esatta pagina come immagine. Le pagine senza testo non sono indicizzabili dal
    RAG lessicale/semantico e vengono saltate (limite noto del prototipo).
    """
    import fitz

    documenti: list[Document] = []
    file_pdf = sorted(glob.glob(f"{cartella}/*.pdf"))
    if not file_pdf:
        print(f"[!] Nessun PDF trovato in '{cartella}' per l'estrazione a pagina.")

    for percorso in file_pdf:
        nome_file = os.path.basename(percorso)
        try:
            doc = fitz.open(percorso)
        except Exception as e:
            print(f"    [!] Impossibile aprire {nome_file}: {e}")
            continue
        pagine_con_testo = 0
        for n in range(doc.page_count):
            testo = _normalizza_simboli(doc.load_page(n).get_text())
            if not testo.strip():
                continue
            pagine_con_testo += 1
            documenti.append(Document(
                page_content=f"--- FONTE: {nome_file} (pag. {n + 1}) ---\n{testo}",
                metadata={"percorso": percorso, "pagina": n, "fonte": nome_file},
            ))
        doc.close()
        print(f"    - Slide indicizzate per pagina: {nome_file} ({pagine_con_testo} pagine con testo)")

    if include_code:
        documenti.extend(_estrai_file_codice(cartella))

    return documenti


def _rendi_pagina_pdf_base64(percorso: str, pagina: int, dpi: int = 110) -> str:
    """Rende una singola pagina PDF in PNG e la restituisce come stringa base64."""
    import base64
    import fitz

    doc = fitz.open(percorso)
    try:
        pix = doc.load_page(pagina).get_pixmap(dpi=dpi)
        png = pix.tobytes("png")
    finally:
        doc.close()
    return base64.b64encode(png).decode("ascii")


def dividi_trascrizione_in_blocchi(testo: str, max_parole: int = 1500, overlap_parole: int = 150) -> list[str]:
    parole = testo.split()
    blocchi = []
    passo = max_parole - overlap_parole
    i = 0
    while i < len(parole):
        blocco = " ".join(parole[i : i + max_parole])
        blocchi.append(blocco)
        # Se questo blocco arriva già alla fine del testo, fermati qui:
        # evita di generare un blocco finale orfano composto solo dall'overlap.
        if i + max_parole >= len(parole):
            break
        i += passo

    print(f"[Info] Trascrizione divisa in {len(blocchi)} blocchi.")
    print(f"[Info] Impostato overlap di sicurezza di {overlap_parole} parole tra i blocchi.")
    return blocchi


def estrai_testo_da_cartella_txt(cartella: str) -> str:
    """
    Legge tutti i file .txt e .md di una cartella, li ordina alfabeticamente
    e li unisce in un unico grande testo.
    """
    file_txt = glob.glob(f"{cartella}/*.txt")
    file_md = glob.glob(f"{cartella}/*.md")

    file_trovati = sorted(file_txt + file_md)
    testo_totale = ""

    if not file_trovati:
        print(f"[!] Nessun file TXT o MD trovato nella cartella '{cartella}'.")
        return ""

    for percorso_file in file_trovati:
        nome_file = os.path.basename(percorso_file)
        print(f"    - Aggiungo trascrizione: {nome_file}")

        with open(percorso_file, 'r', encoding='utf-8') as file:
            testo_totale += f"\n\n--- INIZIO {nome_file} ---\n\n"
            testo_totale += file.read() + "\n\n"

    return testo_totale


# ==========================================================================
# GRAFO LANGGRAPH
# ==========================================================================
def crea_nodo_correzione(llm):
    def nodo_correzione(state: GraphState) -> dict:
        user_prompt = f"-- SLIDE --\n{state['testo_slide']}\n-- TRASCRIZIONE --\n{state['trascrizione_grezza']}"
        risposta = llm.invoke([SystemMessage(content=SYSTEM_PROMPT_CORREZIONE), HumanMessage(content=user_prompt)])
        return {"trascrizione_pulita": risposta.content}
    return nodo_correzione


def crea_nodo_generazione(llm, system_prompt: str, user_prompt_suffix: str = ""):
    def nodo_generazione(state: GraphState) -> dict:
        user_prompt = f"""
    --- CONTESTO PRECEDENTE (Cosa hai già scritto nel blocco precedente. NON RISPIEGARE QUESTE COSE) ---
    {state['memoria_precedente']}

    --- SLIDE DI RIFERIMENTO ---
    {state['testo_slide']}

    --- BLOCCO TRASCRIZIONE PULITA DA ELABORARE ORA ---
    {state['trascrizione_pulita']}
    {user_prompt_suffix}"""
        risposta = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
        return {"documento_finale": risposta.content}
    return nodo_generazione


class RetrieverIbrido:
    """
    Fonde un retriever lessicale (BM25) e uno semantico (embedding) con la
    Reciprocal Rank Fusion pesata: ogni documento riceve un punteggio pari alla
    somma, su entrambe le classifiche, di peso / (c + rango). È la stessa logica
    dell'EnsembleRetriever di LangChain, reimplementata per non dipendere dal
    meta-pacchetto `langchain` (i cui namespace cambiano tra le versioni).
    """
    def __init__(self, bm25, semantico, peso_bm25: float, peso_semantico: float, k: int, c: int = 60):
        self.bm25 = bm25
        self.semantico = semantico
        self.peso_bm25 = peso_bm25
        self.peso_semantico = peso_semantico
        self.k = k
        self.c = c

    def invoke(self, query: str) -> list[Document]:
        classifiche = [
            (self.bm25.invoke(query), self.peso_bm25),
            (self.semantico.invoke(query), self.peso_semantico),
        ]
        punteggi: dict[str, float] = {}
        doc_per_chiave: dict[str, Document] = {}
        for documenti, peso in classifiche:
            for rango, doc in enumerate(documenti, start=1):
                chiave = doc.page_content
                doc_per_chiave[chiave] = doc
                punteggi[chiave] = punteggi.get(chiave, 0.0) + peso * (1.0 / (self.c + rango))
        ordinate = sorted(punteggi, key=punteggi.get, reverse=True)
        return [doc_per_chiave[chiave] for chiave in ordinate[:self.k]]


def costruisci_retriever(documenti_slide: list[Document], config: PipelineConfig):
    """
    Costruisce il motore di ricerca RAG. Di default usa il solo BM25 (lessicale).
    Se `usa_retriever_ibrido` è attivo, lo fonde con un retriever semantico
    basato su embedding (import "pigri": le dipendenze pesanti servono solo qui).
    """
    bm25 = BM25Retriever.from_documents(documenti_slide)
    bm25.k = config.retriever_k

    if not config.usa_retriever_ibrido:
        return bm25

    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.vectorstores import InMemoryVectorStore

    print(f"[RAG] Inizializzo il retriever ibrido (embedding: {config.embedding_model})...")
    embeddings = HuggingFaceEmbeddings(model_name=config.embedding_model)
    vector_store = InMemoryVectorStore.from_documents(documenti_slide, embeddings)
    semantico = vector_store.as_retriever(search_kwargs={"k": config.retriever_k})

    return RetrieverIbrido(bm25, semantico, config.peso_bm25, config.peso_semantico, config.retriever_k)


def costruisci_grafo(llm, config: PipelineConfig):
    workflow = StateGraph(GraphState)
    workflow.add_node("correzione", crea_nodo_correzione(llm))
    workflow.add_node("generazione", crea_nodo_generazione(llm, config.system_prompt_generazione, config.user_prompt_suffix))
    workflow.add_edge(START, "correzione")
    workflow.add_edge("correzione", "generazione")
    workflow.add_edge("generazione", END)
    return workflow.compile()


# ==========================================================================
# PARSING E RENDERING HTML
# ==========================================================================
def pulisci_meta_commenti(testo_html: str) -> str:
    """Cancella via Regex le tipiche frasi introduttive robotiche dell'IA."""
    pattern_logorrea = r"(?i)(?:In questo|Questo|Il|Il presente|Proseguendo da dove)[^\.]*?(?:blocco|segmento|frammento|paragrafo)[^\.]*?(?:si concentra|approfondisce|analizza|analizzeremo|parleremo di|si focalizza|siamo interrotti|segue)[^\.]*\.\s*"
    return re.sub(pattern_logorrea, "", testo_html)


def genera_indice(testo_html: str) -> str:
    """Trova i titoli, assegna un ID univoco e inietta un indice cliccabile sotto l'H1."""
    soup = BeautifulSoup(testo_html, 'html.parser')

    indice_html = "<div class='indice'><h2>Indice dei Contenuti</h2><ul>"
    titoli = soup.find_all(['h2', 'h3'])

    if not titoli:
        return testo_html

    for i, tag in enumerate(titoli):
        id_titolo = f"sezione-{i}"
        tag['id'] = id_titolo
        indice_html += f"<li><a href='#{id_titolo}'>{tag.text.strip()}</a></li>"

    indice_html += "</ul></div>"

    container = soup.find('div', class_='container')
    if container and container.h1:
        container.h1.insert_after(BeautifulSoup(indice_html, 'html.parser'))
    elif soup.body:
        soup.body.insert(0, BeautifulSoup(indice_html, 'html.parser'))

    return str(soup)


# Tabella di codifica del font Adobe "Symbol": codice carattere -> Unicode reale.
# Le slide PDF che usano il font Symbol per i simboli matematici, una volta
# estratte, escono come codepoint della Private Use Area (0xF000 + codice) che
# nessun font sa disegnare (compaiono come "quadratini"/tofu). Questa mappa li
# riporta ai caratteri Unicode corretti (frecce, operatori insiemistici, greche).
_SYMBOL_A_UNICODE = {
    0x22: "∀", 0x24: "∃", 0x27: "∋", 0x40: "≅",
    0x41: "Α", 0x42: "Β", 0x43: "Χ", 0x44: "Δ", 0x45: "Ε", 0x46: "Φ",
    0x47: "Γ", 0x48: "Η", 0x49: "Ι", 0x4A: "ϑ", 0x4B: "Κ", 0x4C: "Λ",
    0x4D: "Μ", 0x4E: "Ν", 0x4F: "Ο", 0x50: "Π", 0x51: "Θ", 0x52: "Ρ",
    0x53: "Σ", 0x54: "Τ", 0x55: "Υ", 0x56: "ς", 0x57: "Ω", 0x58: "Ξ",
    0x59: "Ψ", 0x5A: "Ζ",
    0x61: "α", 0x62: "β", 0x63: "χ", 0x64: "δ", 0x65: "ε", 0x66: "φ",
    0x67: "γ", 0x68: "η", 0x69: "ι", 0x6A: "ϕ", 0x6B: "κ", 0x6C: "λ",
    0x6D: "μ", 0x6E: "ν", 0x6F: "ο", 0x70: "π", 0x71: "θ", 0x72: "ρ",
    0x73: "σ", 0x74: "τ", 0x75: "υ", 0x76: "ϖ", 0x77: "ω", 0x78: "ξ",
    0x79: "ψ", 0x7A: "ζ",
    0xA2: "′", 0xA3: "≤", 0xA5: "∞", 0xAB: "↔", 0xAC: "←", 0xAD: "↑",
    0xAE: "→", 0xAF: "↓", 0xB0: "°", 0xB1: "±", 0xB2: "″", 0xB3: "≥",
    0xB4: "×", 0xB5: "∝", 0xB6: "∂", 0xB7: "•", 0xB8: "÷", 0xB9: "≠",
    0xBA: "≡", 0xBB: "≈", 0xC6: "∅", 0xC7: "∩", 0xC8: "∪", 0xC9: "⊃",
    0xCA: "⊇", 0xCB: "⊄", 0xCC: "⊂", 0xCD: "⊆", 0xCE: "∈", 0xCF: "∉",
    0xD1: "∇", 0xD5: "∏", 0xD6: "√", 0xD7: "⋅", 0xD8: "¬", 0xD9: "∧",
    0xDA: "∨", 0xDB: "⇔", 0xDC: "⇐", 0xDD: "⇑", 0xDE: "⇒", 0xDF: "⇓",
    0xE5: "∑", 0xF2: "∫",
}
_SIMBOLI_PUA = {chr(0xF000 + codice): ch for codice, ch in _SYMBOL_A_UNICODE.items()}
_RE_SIMBOLI_PUA = re.compile("|".join(map(re.escape, _SIMBOLI_PUA)))


def _normalizza_simboli(testo: str) -> str:
    """
    Riporta a Unicode i simboli del font Symbol estratti dai PDF come codepoint
    della Private Use Area (i "quadratini"). Se non c'è nulla da rimappare il
    testo torna invariato.
    """
    if not testo:
        return testo
    return _RE_SIMBOLI_PUA.sub(lambda m: _SIMBOLI_PUA[m.group(0)], testo)


def _neutralizza_variabili_dollaro(testo: str) -> str:
    """
    Rete di sicurezza contro la collisione tra le pseudo-variabili di Yacc/Bison
    ($$, $1, $2, ...) e i delimitatori matematici di MathJax ($ e $$). Nel testo
    discorsivo queste variabili verrebbero interpretate come formule, sballando
    l'impaginazione. Le racchiude in <code> (che MathJax ignora di default),
    preservando invece il codice già formattato e le vere formule LaTeX.
    """
    segnaposto: list[str] = []

    def _maschera(testo_da_proteggere: str) -> str:
        segnaposto.append(testo_da_proteggere)
        return f"\x00{len(segnaposto) - 1}\x00"

    # 1. Proteggi i fence di codice ```...```: restano grezzi (li converte markdown).
    testo = re.sub(r"```.*?```", lambda m: _maschera(m.group(0)), testo, flags=re.DOTALL)
    # 2. Converti gli span inline `...` in <code>...</code> e proteggili subito.
    #    <code> è universale (MathJax lo salta e rende come codice in OGNI sezione,
    #    anche in Concetti/digressioni/box che non passano dal renderer markdown),
    #    quindi neutralizza i $-riferimenti anche quando il modello li ha già messi
    #    tra backtick seguendo la regola 9 del prompt.
    testo = re.sub(r"`([^`\n]+)`", lambda m: _maschera(f"<code>{m.group(1)}</code>"), testo)
    # 3. Proteggi la matematica LaTeX autentica: i blocchi display "$$...$$" che
    #    contengono un comando LaTeX (\...) e l'inline "$ ... $" scritto con gli
    #    spazi previsti dalla regola del prompt (es. "$ L(G) $").
    testo = re.sub(r"\$\$[^$]*?\\[^$]*?\$\$", lambda m: _maschera(m.group(0)), testo, flags=re.DOTALL)
    # Il lookbehind (?<!\$) evita di partire dal secondo "$" di un "$$" Yacc, e il
    # lookahead (?!\d) evita che il "$" di chiusura sia in realtà l'inizio di un "$N":
    # così "$$ = $1" non viene scambiato per la formula inline "$ = $".
    testo = re.sub(r"(?<!\$)\$ [^$\n]+? \$(?!\d)", lambda m: _maschera(m.group(0)), testo)
    # 4. Racchiudi in <code> le pseudo-variabili Yacc nude rimaste ($$, $$P1, $1, ...).
    testo = re.sub(r"\$\$[A-Za-z0-9_]*|\$\d+", lambda m: f"<code>{m.group(0)}</code>", testo)
    # 5. Ripristina le porzioni protette.
    testo = re.sub(r"\x00(\d+)\x00", lambda m: segnaposto[int(m.group(1))], testo)
    return testo


def salva_dispensa_html(config: PipelineConfig, s1: str, s2: str, s3: str):
    # Rimappa i simboli PUA (tofu) del font Symbol prima di ogni altra cosa.
    s1, s2, s3 = _normalizza_simboli(s1), _normalizza_simboli(s2), _normalizza_simboli(s3)

    if config.proteggi_variabili_dollaro:
        s1 = _neutralizza_variabili_dollaro(s1)
        s2 = _neutralizza_variabili_dollaro(s2)
        s3 = _neutralizza_variabili_dollaro(s3)

    def formatta_esercizi(testo_markdown):
        testo = re.sub(r"<box_esercizio>\s*", '<div class="exercise-box"><div class="exercise-title">📝 Esercizio Guidato / Procedura</div>\n\n', testo_markdown)
        testo = re.sub(r"\s*</box_esercizio>", '\n</div>\n', testo)
        return testo

    def formatta_paragrafi(testo, classe_css=""):
        class_attr = f' class="{classe_css}"' if classe_css else ""
        paragrafi = testo.strip().split('\n\n')
        html = ""
        for p in paragrafi:
            if p.strip() and p.strip() != "None" and p.strip() != "null":
                html += f"<p{class_attr}>{p.strip()}</p>\n"
        return html

    def formatta_aneddoti(testo):
        paragrafi = testo.strip().split('\n\n')
        html = ""
        frasi_vuote = ["non sono presenti digressioni", "non emergono nel frammento", "non sono presenti aneddoti", "nessun aneddoto", "nessuna digressione"]
        for p in paragrafi:
            testo_p = p.strip()
            if testo_p and not any(frase in testo_p.lower() for frase in frasi_vuote):
                html += f"""
                <div class="anecdote-box">
                    <div class="anecdote-title">💡 Spunto di Riflessione / Digressione</div>
                    <p class="anecdote-content">{testo_p}</p>
                </div>
                """
        return html

    html_concetti = formatta_paragrafi(s1, "concept-text")

    if config.render_teoria_markdown:
        s2_render = formatta_esercizi(s2) if config.has_esercizio else s2
        html_teoria = markdown.markdown(s2_render, extensions=list(config.markdown_extensions))
        teoria_block = f'<div class="theory-text">\n                {html_teoria}\n            </div>'
    else:
        teoria_block = formatta_paragrafi(s2, "theory-text")

    html_aneddoti = formatta_aneddoti(s3)

    # --- Blocchi opzionali dell'head (Highlight.js) ---
    if config.enable_code_highlight:
        highlight_head = """
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
        <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
        <script>hljs.highlightAll();</script>
        <script>
            // Compatta i blocchi di codice piu' lunghi cosi' restano leggibili e
            // spezzano meno le pagine in stampa (soglie a numero di righe).
            document.addEventListener('DOMContentLoaded', function () {
                document.querySelectorAll('pre').forEach(function (pre) {
                    var righe = (pre.innerText.match(/\\n/g) || []).length + 1;
                    if (righe > 32) { pre.classList.add('code-lunghissimo'); }
                    else if (righe > 18) { pre.classList.add('code-lungo'); }
                });
            });
        </script>
"""
    else:
        highlight_head = ""

    # --- CSS opzionali ---
    css_extra = ""
    if config.enable_code_highlight:
        css_extra += """
        pre {
            break-inside: avoid;
            page-break-inside: avoid;
            overflow-x: auto;
        }
        pre code {
            border-radius: 8px;
            font-family: 'Consolas', 'Courier New', Courier, monospace;
            font-size: 9.5pt;
            line-height: 1.4;
            padding: 14px 15px;
            margin-top: 15px;
            margin-bottom: 15px;
        }
        /* Blocchi lunghi: carattere e interlinea ridotti per stare in pagina. */
        pre.code-lungo code { font-size: 8pt; line-height: 1.3; }
        pre.code-lunghissimo code { font-size: 7pt; line-height: 1.25; }"""
    if config.enable_exercise_css:
        css_extra += """
        .exercise-box {
            background-color: #f0fdf4;
            border-left: 5px solid #22c55e;
            padding: 20px;
            margin: 25px 0;
            border-radius: 0 8px 8px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .exercise-title {
            font-weight: 700;
            color: #166534;
            font-size: 11pt;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 12px;
            border-bottom: 1px solid #bbf7d0;
            padding-bottom: 5px;
        }"""
    if config.allega_slide_immagini:
        css_extra += """
        details.slide-originale { margin-top: 14px; }
        details.slide-originale summary {
            cursor: pointer;
            font-weight: 600;
            color: #166534;
            font-size: 9.5pt;
            list-style: none;
        }
        details.slide-originale summary::before { content: "▸ "; }
        details.slide-originale[open] summary::before { content: "▾ "; }
        details.slide-originale img {
            max-width: 100%;
            height: auto;
            margin-top: 10px;
            border: 1px solid #e2e8f0;
            border-radius: 6px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        }"""
    if config.enable_table_css:
        css_extra += """
        table { border-collapse: collapse; width: 100%; margin: 20px 0; font-size: 10.5pt; }
        th { background-color: #2b6cb0; color: white; padding: 10px; text-align: left; }
        td { border: 1px solid #e2e8f0; padding: 10px; }
        tr:nth-child(even) { background-color: #f8fafc; }"""

    css_extra_block = f"        <style>{css_extra}\n        </style>\n" if css_extra else ""

    template_html = f"""<!DOCTYPE html>
    <html lang="it">
    <head>
        <meta charset="UTF-8">
        <title>Dispensa Universitaria Autonoma</title>

        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">

        <script>
            MathJax = {{
                tex: {{ inlineMath: [['$', '$'], ['\\\\(', '\\\\)']] }}
            }};
        </script>
        <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
{highlight_head}
{css_extra_block}
        <style>
            @page {{ size: A4; margin: 20mm 18mm; }}
            body {{
                max-width: 800px;
                margin: 40px auto;
                font-family: 'Inter', sans-serif;
                line-height: 1.7;
                color: #2c3e50;
                padding: 0 20px;
            }}
            .indice {{
                background-color: #f8f9fa;
                border-left: 4px solid #3498db;
                padding: 20px;
                margin-bottom: 40px;
                border-radius: 4px;
            }}
            .indice a {{ text-decoration: none; color: #2980b9; }}
            .indice a:hover {{ text-decoration: underline; }}

            h1 {{ color: #1a365d; font-size: 22pt; border-bottom: 2px solid #2b6cb0; padding-bottom: 5px; margin-top: 40px; text-transform: uppercase; letter-spacing: 0.5px; }}
            h2 {{ color: #2b6cb0; font-size: 15pt; margin-top: 35px; margin-bottom: 15px; border-left: 5px solid #2b6cb0; padding-left: 10px; }}
            p {{ text-align: justify; text-justify: inter-word; margin-bottom: 14px; font-size: 11pt; }}
            .concept-text {{ font-weight: 500; color: #2c5282; }}
            .theory-text {{ color: #2d3748; }}
            .anecdote-box {{ background-color: #fffaf0; border-left: 4px solid #dd6b20; padding: 15px 18px; margin: 20px 0; border-radius: 0 6px 6px 0; page-break-inside: avoid; }}
            .anecdote-title {{ font-weight: bold; color: #dd6b20; font-size: 9.5pt; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }}
            .anecdote-content {{ font-style: italic; color: #4a5568; margin: 0; font-size: 10.5pt; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Dispensa Ufficiale del Corso</h1>
            <h2>1. Concetti Chiave e Nozioni Fondamentali</h2>
            {html_concetti}
            <h2>2. Spiegazione Dettagliata e Sviluppo</h2>
            {teoria_block}
            <h2>3. Ulteriori Spunti e Contenuti di Supporto</h2>
            {html_aneddoti}
        </div>
    </body>
    </html>
    """

    html_pulito = pulisci_meta_commenti(template_html)
    html_finale = genera_indice(html_pulito)

    with open(config.nome_output, 'w', encoding='utf-8') as f:
        f.write(html_finale)
    print(f"\n[✓] Layout grafico generato con successo in: {config.nome_output}")


# ==========================================================================
# ORCHESTRAZIONE
# ==========================================================================
def _applica_override_cli(config: PipelineConfig):
    """Permette di sovrascrivere cartelle/modello/output da riga di comando."""
    parser = argparse.ArgumentParser(description="Generatore di dispense universitarie autonome.")
    parser.add_argument("--slide", dest="cartella_slide", help="Cartella con le slide (PDF/PPTX/DOCX).")
    parser.add_argument("--testi", dest="cartella_trascrizioni", help="Cartella con le trascrizioni (TXT/MD).")
    parser.add_argument("--output", dest="nome_output", help="Nome del file HTML di output.")
    parser.add_argument("--model", dest="model", help="Nome del modello LLM da usare.")
    args = parser.parse_args()

    if args.cartella_slide:
        config.cartella_slide = args.cartella_slide
    if args.cartella_trascrizioni:
        config.cartella_trascrizioni = args.cartella_trascrizioni
    if args.nome_output:
        config.nome_output = args.nome_output
    if args.model:
        config.model = args.model
    return config


def run(config: PipelineConfig):
    config = _applica_override_cli(config)

    llm = ChatOpenAI(
        base_url=config.base_url,
        api_key=os.environ.get(config.api_key_env),
        model=config.model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        max_retries=5,
        timeout=60,
    )

    app = costruisci_grafo(llm, config)

    print(f"\n[FUSIONE] Lettura e unione delle trascrizioni nella cartella '{config.cartella_trascrizioni}'...")
    trascrizione_completa = estrai_testo_da_cartella_txt(config.cartella_trascrizioni)

    print(f"\n[RAG] Lettura di tutti i documenti nella cartella '{config.cartella_slide}'...")
    if config.allega_slide_immagini:
        print("[RAG] Modalità immagini attiva: indicizzazione delle slide PDF a granularità di pagina.")
        documenti_slide = estrai_slide_per_pagina(config.cartella_slide, include_code=config.include_code_files)
    else:
        documenti_slide = estrai_materiale_didattico(
            config.cartella_slide,
            include_code=config.include_code_files,
            usa_cache=config.usa_cache,
            cache_dir=config.cache_dir,
        )

    if not documenti_slide:
        print("[!] ERRORE GRAVE: Nessun documento valido trovato. Impossibile creare il motore di ricerca.")
        print("Assicurati di aver inserito le slide e di aver installato 'markitdown[all]'. Uscita in corso...")
        return

    motore_ricerca = costruisci_retriever(documenti_slide, config)

    blocchi_trascrizione = dividi_trascrizione_in_blocchi(trascrizione_completa, config.max_parole, config.overlap_parole)
    sezione_1, sezione_2, sezione_3 = "", "", ""
    paragrafi_teoria: list[str] = []    # storico per la guardia anti-ripetizione
    paragrafi_concetti: list[str] = []  # storico dei Concetti, per il cross-check con la teoria

    memoria_storica = "Questo è il primo blocco, inizia l'introduzione."

    print("\n--- AVVIO ELABORAZIONE SEQUENZIALE (TOTALMENTE BLINDATA) ---")
    for indice, blocco in enumerate(blocchi_trascrizione, start=1):
        print(f"\n---> Avvio Elaborazione Blocco {indice} di {len(blocchi_trascrizione)}...")

        slide_recuperate = motore_ricerca.invoke(blocco)

        if config.print_rag_sources:
            fonti_usate = set()
            for doc in slide_recuperate:
                match = re.search(r"--- FONTE: (.*?) ---", doc.page_content)
                if match:
                    fonti_usate.add(match.group(1))
            testo_fonti = ", ".join(fonti_usate) if fonti_usate else "Nessun riferimento specifico"
            print(f"    [RAG] Consultando le slide: {testo_fonti}")

        slide_rilevanti_per_blocco = "\n\n".join([doc.page_content for doc in slide_recuperate])

        # [prototipo] Rende la slide più rilevante come immagine, da allegare
        # all'eventuale esercizio di questo blocco (una volta per blocco).
        slide_immagine_html = ""
        if config.allega_slide_immagini and slide_recuperate:
            meta = slide_recuperate[0].metadata or {}
            if "percorso" in meta:
                try:
                    b64 = _rendi_pagina_pdf_base64(meta["percorso"], meta["pagina"], config.dpi_slide)
                    slide_immagine_html = (
                        '\n\n<details class="slide-originale">'
                        f'<summary>📄 Slide originale — {meta.get("fonte", "")} (pag. {meta["pagina"] + 1})</summary>\n'
                        f'<img src="data:image/png;base64,{b64}" alt="Slide originale"/>'
                        '</details>\n\n'
                    )
                except Exception as e:
                    print(f"    [!] Rendering della slide fallito: {e}")

        input_stato = {
            "testo_slide": slide_rilevanti_per_blocco,
            "trascrizione_grezza": blocco,
            "trascrizione_pulita": "",
            "memoria_precedente": memoria_storica,
            "documento_finale": "",
        }

        successo = False
        tentativi_falliti = 0
        while not successo:
            try:
                risultato = app.invoke(input_stato)
                tg = risultato["documento_finale"]

                m1 = re.search(r"<concetti>(.*?)</concetti>", tg, re.DOTALL | re.IGNORECASE)
                m2 = re.search(r"<spiegazione>(.*?)</spiegazione>", tg, re.DOTALL | re.IGNORECASE)
                m3 = re.search(r"<digressioni>(.*?)</digressioni>", tg, re.DOTALL | re.IGNORECASE)

                # --- CANE DA GUARDIA: se mancano i tag obbligatori, forza il retry del blocco ---
                if not m1 and not m2:
                    print(f"\n    [DEBUG] Il modello ha risposto questo invece dei tag XML:\n    >>> {tg[:500]}...\n")
                    raise ValueError("Il modello ha fallito la formattazione XML o ha restituito un errore.")

                if m1 and m1.group(1).strip():
                    testo_concetti = m1.group(1).strip()
                    paragrafi_concetti.append(testo_concetti)
                    sezione_1 += testo_concetti + "\n\n"

                if m2 and m2.group(1).strip():
                    testo_teoria = m2.group(1).strip()
                    # Guardia anti-ripetizione: scarta il paragrafo se quasi-identico
                    # a uno già inserito nella teoria O nei Concetti (capita che il
                    # modello "eco-i" la memoria o riproponga il riassunto come teoria).
                    duplicato = any(
                        difflib.SequenceMatcher(None, testo_teoria, precedente).ratio() >= config.soglia_antiripetizione
                        for precedente in paragrafi_teoria + paragrafi_concetti
                    )
                    if duplicato:
                        print("    [Anti-ripetizione] Paragrafo di teoria quasi-identico a teoria/Concetti già inseriti: scartato.")
                    else:
                        paragrafi_teoria.append(testo_teoria)
                        if config.separatori_teoria:
                            separatore = "\n\n---\n\n" if len(sezione_2) > 0 else ""
                            sezione_2 += separatore + testo_teoria + "\n\n"
                        else:
                            sezione_2 += testo_teoria + "\n\n"

                if config.has_esercizio:
                    m_ex = re.search(r"<esercizio>(.*?)</esercizio>", tg, re.DOTALL | re.IGNORECASE)
                    testo_ex = m_ex.group(1).strip() if m_ex else ""
                    frasi_vuote_ex = ["non presente", "nessun esercizio", "non viene risolto", "non ci sono esercizi", "nessun frammento"]
                    if testo_ex and not any(frase in testo_ex.lower() for frase in frasi_vuote_ex):
                        sezione_2 += "\n\n<box_esercizio>\n" + testo_ex + slide_immagine_html + "\n</box_esercizio>\n\n"

                if m3 and m3.group(1).strip():
                    sezione_3 += m3.group(1).strip() + "\n\n"

                if m2 and m2.group(1).strip():
                    memoria_storica = m2.group(1).strip()[-config.memoria_caratteri:]

                print(f"[✓] Blocco {indice} completato con successo!")
                successo = True

                if indice < len(blocchi_trascrizione):
                    print(f"    [Pausa] Attesa di {config.pausa_secondi} secondi per non sovraccaricare il provider...")
                    time.sleep(config.pausa_secondi)

            except KeyboardInterrupt:
                print("\n[STOP] Hai interrotto manualmente il programma.")
                raise SystemExit()

            except Exception as e:
                tentativi_falliti += 1
                # Backoff esponenziale: 30, 60, 120, 240... con tetto massimo (backoff_max).
                attesa = min(config.pausa_retry * (2 ** (tentativi_falliti - 1)), config.backoff_max)
                print(f"    [!] Il provider ha avuto un mancamento (tentativo {tentativi_falliti}): {e}")
                print(f"    [!] Niente panico. Backoff esponenziale: pausa di {attesa} secondi e poi riprovo il Blocco {indice}...")
                time.sleep(attesa)

    # --- FASE 3: REVISIONE FINALE E DEDUPLICAZIONE ---
    print("\n[REVISIONE FINALE] Lettura incrociata per eliminare i doppioni dalla Sezione 3...")

    prompt_revisione = f"""Sei un revisore editoriale spietato.
    Qui sotto troverai due testi estratti da una lezione: la SEZIONE 2 (teoria e narrazione principale) e la SEZIONE 3 (digressioni extra).
    Il tuo UNICO compito è leggere la SEZIONE 3 e CANCELLARE qualsiasi aneddoto, storia o concetto che è già stato raccontato nella SEZIONE 2.
    Se un aneddoto nella SEZIONE 3 è un doppione (anche se raccontato con parole leggermente diverse), eliminalo del tutto. Se invece è una storia o una battuta nuova, mantienila.

    RESTITUISCI SOLO ED ESCLUSIVAMENTE IL TESTO PULITO DELLA SEZIONE 3. Non aggiungere nessun meta-commento, titolo o introduzione.

    --- SEZIONE 2 (Testo di riferimento - GIA' PERFETTO) ---
    {sezione_2}

    --- SEZIONE 3 (Testo da pulire e filtrare) ---
    {sezione_3}
    """

    try:
        risposta_revisore = llm.invoke([HumanMessage(content=prompt_revisione)])
        sezione_3_pulita = risposta_revisore.content.strip()
        print("[✓] Revisione completata! Doppioni eliminati con successo.")
    except Exception as e:
        print(f"[!] Errore durante la revisione finale ({e}). Uso la Sezione 3 originale.")
        sezione_3_pulita = sezione_3

    # --- FASE 4: ESPORTAZIONE IN HTML/PDF ---
    salva_dispensa_html(config, sezione_1, sezione_2, sezione_3_pulita)

    print(f"\n[SUCCESSO TOTALE] Pipeline completata. Apri '{config.nome_output}' nel browser e premi Ctrl+P per stampare in PDF!")
