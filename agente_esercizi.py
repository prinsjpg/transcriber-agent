import os
import re
import time
import glob
import markdown
from dotenv import load_dotenv
from typing import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage
from markitdown import MarkItDown
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from bs4 import BeautifulSoup

load_dotenv()

# Configurazione universale tramite OpenRouter
llm = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY_OWL_ALPHA"),  # Usa la chiave specifica per il modello scelto
    model="openrouter/owl-alpha",  # Qui scrivi il nome del modello che vuoi usare
    temperature=0.3,
    max_tokens=8192, 
    max_retries=5,
    timeout=60
)

class GraphState(TypedDict):
    testo_slide: str
    trascrizione_grezza: str
    trascrizione_pulita: str
    memoria_precedente: str
    documento_finale: str

def estrai_materiale_didattico(cartella: str) -> list[Document]:
    """
    Legge PDF/PPTX tramite MarkItDown e allega i file di codice sorgente per il RAG.
    """
    documenti = []
    md = MarkItDown()
    
    # 1. Estrazione da Slide e Documenti (NUOVO MOTORE)
    estensioni_doc = ['*.pdf', '*.pptx', '*.docx']
    file_doc = []
    for est in estensioni_doc:
        file_doc.extend(glob.glob(f"{cartella}/{est}"))
        
    for percorso in file_doc:
        nome_file = os.path.basename(percorso)
        print(f"    - Conversione slide in Markdown: {nome_file}...")
        try:
            risultato = md.convert(percorso)
            if risultato.text_content:
                documenti.append(Document(page_content=f"--- FONTE: {nome_file} ---\n{risultato.text_content}"))
        except Exception as e:
            print(f"    [!] Errore conversione {nome_file}: {e}")
                
    # 2. Estrazione dai file di Codice Sorgente (Migliorata)
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

def dividi_trascrizione_in_blocchi(testo: str, max_parole: int = 1500, overlap_parole: int = 150) -> list[str]:
    parole = testo.split()
    blocchi = []
    passo = max_parole - overlap_parole
    for i in range(0, len(parole), passo):
        blocco = " ".join(parole[i : i + max_parole])
        blocchi.append(blocco)
        
    print(f"[Info] Trascrizione divisa in {len(blocchi)} blocchi.")
    print(f"[Info] Impostato overlap di sicurezza di {overlap_parole} parole tra i blocchi.")
    return blocchi

def pulisci_meta_commenti(testo_html: str) -> str:
    """
    Usa le espressioni regolari (Regex) per trovare e cancellare le tipiche 
    frasi introduttive generate dall'IA.
    """
    # Regex ampliata per intercettare anche le forme "Il frammento approfondisce..." o "Il blocco che segue..."
    pattern_logorrea = r"(?i)(?:In questo|Questo|Il|Il presente|Proseguendo da dove)[^\.]*?(?:blocco|segmento|frammento|paragrafo)[^\.]*?(?:si concentra|approfondisce|analizza|analizzeremo|parleremo di|si focalizza|siamo interrotti|segue)[^\.]*\.\s*"
    testo_pulito = re.sub(pattern_logorrea, "", testo_html)
    return testo_pulito

def genera_indice(testo_html: str) -> str:
    """
    Analizza l'HTML, trova i titoli, assegna loro un ID univoco 
    e inietta un indice cliccabile subito dopo il titolo principale.
    """
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
    
    # Cerca il contenitore principale per inserire l'indice in modo elegante sotto l'H1
    container = soup.find('div', class_='container')
    if container and container.h1:
        container.h1.insert_after(BeautifulSoup(indice_html, 'html.parser'))
    elif soup.body:
        soup.body.insert(0, BeautifulSoup(indice_html, 'html.parser'))
        
    return str(soup)

# --- NUOVA FUNZIONE DI IMPAGINAZIONE HTML/PDF ---
def salva_dispensa_html(s1: str, s2: str, s3: str, nome_file: str = "dispensa_perfetta.html"):
    def formatta_esercizi(testo_markdown):
        # Trasforma i tag <box_esercizio> in div HTML con la classe 'exercise-box'
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
        # Lista delle frasi tipiche di quando non ci sono aneddoti
        frasi_vuote = ["non sono presenti digressioni", "non emergono nel frammento", "non sono presenti aneddoti", "nessun aneddoto"]
        
        for p in paragrafi:
            testo_p = p.strip()
            # Inserisce il box SOLO se il testo non contiene nessuna delle frasi vuote
            if testo_p and not any(frase in testo_p.lower() for frase in frasi_vuote):
                html += f"""
                <div class="anecdote-box">
                    <div class="anecdote-title">💡 Spunto di Riflessione / Digressione</div>
                    <p class="anecdote-content">{testo_p}</p>
                </div>
                """
        return html

    html_concetti = formatta_paragrafi(s1, "concept-text")
    
    s2_con_esercizi = formatta_esercizi(s2)
    # Usiamo la libreria markdown per convertire magicamente i blocchi di codice (fenced_code) 
    # in tag <pre><code> perfetti per Highlight.js
    html_teoria = markdown.markdown(s2_con_esercizi, extensions=['fenced_code'])
    
    html_aneddoti = formatta_aneddoti(s3)

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
        
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css">
        
        <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
        
        <script>hljs.highlightAll();</script>

        <style>
        pre code {{
            border-radius: 8px;
            font-family: 'Courier New', Courier, monospace;
            font-size: 10.5pt;
            padding: 15px;
            margin-top: 15px;
            margin-bottom: 15px;
        }}
        .exercise-box {{ 
            background-color: #f0fdf4; 
            border-left: 5px solid #22c55e; 
            padding: 20px; 
            margin: 25px 0; 
            border-radius: 0 8px 8px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}
        .exercise-title {{ 
            font-weight: 700; 
            color: #166534; 
            font-size: 11pt; 
            text-transform: uppercase; 
            letter-spacing: 0.5px; 
            margin-bottom: 12px;
            border-bottom: 1px solid #bbf7d0;
            padding-bottom: 5px;
        }}
        </style>

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
            <div class="theory-text">
                {html_teoria}
            </div>
            <h2>3. Ulteriori Spunti e Contenuti di Supporto</h2>
            {html_aneddoti}
        </div>
    </body>
    </html>
    """

    # --- APPLICAZIONE DEI FILTRI FINALI ---
    # 1. Pulisce l'HTML generato dalle frasi robotiche
    html_pulito = pulisci_meta_commenti(template_html)
    
    # 2. Genera e inietta l'indice di navigazione
    html_finale = genera_indice(html_pulito)

    with open(nome_file, 'w', encoding='utf-8') as f:
        f.write(html_finale)
    print(f"\n[✓] Layout grafico generato con successo in: {nome_file}")

def nodo_correzione(state: GraphState) -> dict:
    system_prompt = """Sei un revisore editoriale. Il tuo compito è correggere la trascrizione fonetica di un singolo blocco di una lezione.
    Usa il testo delle slide fornito per capire e correggere i termini tecnici storpiati.
    REGOLE: Correggi la punteggiatura ma NON tagliare assolutamente nulla, NON riassumere per nessun motivo e mantieni il 100% del parlato originale."""
    
    user_prompt = f"-- SLIDE --\n{state['testo_slide']}\n-- TRASCRIZIONE --\n{state['trascrizione_grezza']}"
    
    risposta = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
    return {"trascrizione_pulita": risposta.content}

def nodo_generazione(state: GraphState) -> dict:
    system_prompt = """Sei un Tutor Universitario e uno Scrittore Tecnico iper-dettagliato. 
    Stai analizzando un SINGOLO frammento di una lezione molto più ampia.
    
    REGOLE STILISTICHE TASSATIVE: 
    1. DIVIETO DI SINTESI E META-COMMENTI: Espandi il testo in modo discorsivo. È tassativamente vietato usare frasi come "In questo frammento", "Il professore spiega", "Passiamo a". Spiega direttamente i concetti.
    2. TRADUZIONE DEI RIFERIMENTI VISIVI E CROMATICI: Il professore farà spesso riferimento a colori (es. "il nodo rosso", "la freccia blu") o a posizioni ("qui in alto", "questo albero"). TU NON PUOI VEDERE I COLORI NE' LE IMMAGINI. Devi usare la logica: incrocia la spiegazione del professore con il contenuto testuale delle slide (automi, grammatiche) per dedurre a quale elemento tecnico si riferisce. Sostituisci sempre il riferimento al colore con il nome tecnico corretto. (Sbagliato: "Seguendo la freccia rossa...", Corretto: "Seguendo la transizione dal nodo q1 al nodo q0...").
    3. GRAMMATICHE E CODICE: Usa sempre i blocchi di codice Markdown (```) per scrivere le produzioni grammaticali (BNF), il codice sorgente, o per disegnare semplici alberi sintattici in formato testuale.
    4. VINCOLO LINGUISTICO: Scrivi ESCLUSIVAMENTE in lingua Italiana. È tassativamente vietato l'uso di caratteri cinesi, ideogrammi asiatici o parole in altre lingue.
    5. PROTEZIONE NOMI TECNICI: Racchiudi i nomi di token, variabili, nodi e classi tra i backtick (es. `TokenScanner`, `q0`).
    
    Estrai le informazioni e classificale usando ESATTAMENTE questi quattro tag XML:
    
    <concetti>
    (Riassumi i temi chiave in 4 o 5 righe in stile impersonale da libro di testo. Non annunciare mai cosa stai per spiegare).
    </concetti>
    
    <spiegazione>
    (Spiegazione teorica dettagliata a paragrafi continui. Nessun elenco puntato qui).
    </spiegazione>
    
    <esercizio>
    (SE IL PROFESSORE STA RISOLVENDO UN ESERCIZIO: Scrivi qui i passaggi per la risoluzione. Qui PUOI usare elenchi numerati per descrivere i "Passo 1, Passo 2". Mostra il codice o la grammatica passo-passo. Se non c'è nessun esercizio nel frammento, lascia vuoto questo tag).
    </esercizio>
    
    <digressioni>
    (Raccogli qui tutti gli aneddoti e le storie non strettamente tecniche).
    </digressioni>"""
    
    user_prompt = f"""
    --- CONTESTO PRECEDENTE (Cosa hai già scritto nel blocco precedente. NON RISPIEGARE QUESTE COSE) ---
    {state['memoria_precedente']}
    
    --- SLIDE DI RIFERIMENTO ---
    {state['testo_slide']}
    
    --- BLOCCO TRASCRIZIONE PULITA DA ELABORARE ORA ---
    {state['trascrizione_pulita']}
    """
    
    risposta = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
    return {"documento_finale": risposta.content}

def costruisci_grafo():
    workflow = StateGraph(GraphState)
    workflow.add_node("correzione", nodo_correzione)
    workflow.add_node("generazione", nodo_generazione)
    workflow.add_edge(START, "correzione")
    workflow.add_edge("correzione", "generazione")
    workflow.add_edge("generazione", END)
    return workflow.compile()

def estrai_testo_da_cartella_txt(cartella: str) -> str:
    """
    Legge tutti i file .txt e .md all'interno di una cartella, li ordina alfabeticamente 
    e li unisce in un unico grande testo.
    """
    # Cerca sia i file .txt che i file .md
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

if __name__ == "__main__":
    app = costruisci_grafo()
    
    cartella_slide = "slide_compilatori" 
    cartella_trascrizioni = "testi_compilatori"
    nome_output = "dispensa_compilatori.html"
    
    print(f"\n[FUSIONE] Lettura e unione delle trascrizioni nella cartella '{cartella_trascrizioni}'...")
    trascrizione_completa = estrai_testo_da_cartella_txt(cartella_trascrizioni)

    print(f"\n[RAG] Lettura di tutti i documenti nella cartella '{cartella_slide}'...")
    documenti_slide = estrai_materiale_didattico(cartella_slide)
    
    if not documenti_slide:
        print("[!] ERRORE GRAVE: Nessun documento valido trovato. Impossibile creare il motore di ricerca.")
        print("Assicurati di aver inserito le slide e di aver installato 'markitdown[all]'. Uscita in corso...")
        exit()

    motore_ricerca = BM25Retriever.from_documents(documenti_slide)
    motore_ricerca.k = 3 
    
    blocchi_trascrizione = dividi_trascrizione_in_blocchi(trascrizione_completa, max_parole=1500, overlap_parole=150)
    sezione_1, sezione_2, sezione_3 = "", "", ""
    
    memoria_storica = "Questo è il primo blocco, inizia l'introduzione."

    print("\n--- AVVIO ELABORAZIONE SEQUENZIALE (TOTALMENTE BLINDATA) ---")
    for indice, blocco in enumerate(blocchi_trascrizione, start=1):
        print(f"\n---> Avvio Elaborazione Blocco {indice} di {len(blocchi_trascrizione)}...")
        
        slide_recuperate = motore_ricerca.invoke(blocco)
        slide_rilevanti_per_blocco = "\n\n".join([doc.page_content for doc in slide_recuperate])
        
        input_stato = {
            "testo_slide": slide_rilevanti_per_blocco,
            "trascrizione_grezza": blocco, 
            "trascrizione_pulita": "", 
            "memoria_precedente": memoria_storica,
            "documento_finale": ""
        }
        
        successo = False
        while not successo:
            try:
                risultato = app.invoke(input_stato)
                tg = risultato["documento_finale"]
                
                m1 = re.search(r"<concetti>(.*?)</concetti>", tg, re.DOTALL | re.IGNORECASE)
                m2 = re.search(r"<spiegazione>(.*?)</spiegazione>", tg, re.DOTALL | re.IGNORECASE)
                m_ex = re.search(r"<esercizio>(.*?)</esercizio>", tg, re.DOTALL | re.IGNORECASE)
                m3 = re.search(r"<digressioni>(.*?)</digressioni>", tg, re.DOTALL | re.IGNORECASE)
                
                if m1 and m1.group(1).strip(): sezione_1 += m1.group(1).strip() + "\n\n"
                if m2 and m2.group(1).strip(): sezione_2 += m2.group(1).strip() + "\n\n"
                # --- FILTRO ANTI-BOX VUOTI PER GLI ESERCIZI ---
                testo_ex = m_ex.group(1).strip() if m_ex else ""
                # Lista delle frasi che il modello usa per dire che non ci sono esercizi
                frasi_vuote_ex = ["non presente", "nessun esercizio", "non viene risolto", "non ci sono esercizi", "nessun frammento"]
                
                # Aggiungiamo il box solo se c'è testo VERO e non contiene le scuse
                if testo_ex and not any(frase in testo_ex.lower() for frase in frasi_vuote_ex):
                    sezione_2 += "\n\n<box_esercizio>\n" + testo_ex + "\n</box_esercizio>\n\n"
                if m3 and m3.group(1).strip(): sezione_3 += m3.group(1).strip() + "\n\n"
                if m2 and m2.group(1).strip(): 
                    testo_spiegazione = m2.group(1).strip()
                    memoria_storica = testo_spiegazione[-4000:]
                
                print(f"[✓] Blocco {indice} completato con successo!")
                successo = True 
                
                if indice < len(blocchi_trascrizione):
                    print("    [Pausa] Attesa di 15 secondi per non sovraccaricare il provider...")
                    time.sleep(15)
                    
            except KeyboardInterrupt:
                print("\n[STOP] Hai interrotto manualmente il programma.")
                exit() 
                
            except Exception as e:
                print(f"    [!] Il provider ha avuto un mancamento: {e}")
                print(f"    [!] Niente panico. Pausa di 30 secondi e poi riprovo il Blocco {indice}...")
                time.sleep(30)

    # ==========================================
    # FASE 3: REVISIONE FINALE E DEDUPLICAZIONE
    # ==========================================
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

    # ==========================================
    # FASE 4: ESPORTAZIONE IN HTML/PDF
    # ==========================================
    salva_dispensa_html(sezione_1, sezione_2, sezione_3_pulita, nome_output)
    
    print("\n[SUCCESSO TOTALE] Pipeline completata. Apri 'dispensa_compilatori.html' nel browser e premi Ctrl+P per stampare in PDF!")