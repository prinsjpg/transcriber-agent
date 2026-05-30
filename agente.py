import os
import re
import time
import glob
from dotenv import load_dotenv
from typing import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage
from pypdf import PdfReader
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from bs4 import BeautifulSoup

load_dotenv()

# Configurazione universale tramite OpenRouter
llm = ChatOpenAI(
    base_url="https://api.xiaomimimo.com/v1",
    api_key=os.environ.get("MIMO_API_KEY"),  # Usa la chiave specifica per il modello scelto
    model="mimo-v2.5",  # Qui scrivi il nome del modello che vuoi usare
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

def estrai_documenti_da_cartella_pdf(cartella: str) -> list[Document]:
    """
    Legge TUTTI i file PDF all'interno di una cartella e li trasforma 
    in documenti per il RAG, etichettando ogni pagina col nome del file originale.
    """
    documenti = []
    # Cerca tutti i file .pdf nella cartella specificata
    file_trovati = glob.glob(f"{cartella}/*.pdf")
    
    if not file_trovati:
        print(f"[!] Nessun file PDF trovato nella cartella '{cartella}'.")
        return []
        
    for percorso_file in file_trovati:
        nome_file = os.path.basename(percorso_file)
        reader = PdfReader(percorso_file)
        
        for numero_pagina, pagina in enumerate(reader.pages):
            testo_pagina = pagina.extract_text()
            if testo_pagina and testo_pagina.strip():
                # Il RAG ora saprà esattamente da quale pacco di slide arriva la pagina!
                contenuto = f"--- Fonte: {nome_file} - Pagina {numero_pagina + 1} ---\n{testo_pagina}"
                documenti.append(Document(page_content=contenuto))
                
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
    html_teoria = formatta_paragrafi(s2, "theory-text")
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
            {html_teoria}
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
    
    REGOLE STILISTICHE E ANTI-SINTESI TASSATIVE: 
    1. DIVIETO DI SINTESI: Espandi il testo in modo discorsivo, fluido ed ESTREMAMENTE LUNGO. Non usare mai uno stile telegrafico.
    2. DIVIETO DI META-COMMENTI: Non usare MAI espressioni introduttive come "Questo blocco si apre con" o "Il frammento analizza". Tuffati immediatamente nella spiegazione come se stessi continuando un discorso già iniziato.
    3. DIVIETO DI ELENCHI PUNTATI: Scrivi in forma discorsiva a paragrafi continui.
    4. DIVIETO DI LOOP E SINONIMI (ANTI-ALLUCINAZIONE): Non creare MAI liste infinite di termini, sinonimi o parole chiave ripetitive. Sii analitico, razionale e discorsivo.
    5. DIVIETO DI RIPETIZIONE (MEMORIA A STAFFETTA): Leggi attentamente il 'CONTESTO PRECEDENTE'. Se un concetto, un acronimo o una spiegazione è già presente lì, È SEVERAMENTE VIETATO rispiegarlo in questo blocco. Dai per scontato che il lettore lo sappia già e prosegui in avanti con il discorso.
    6. FORMULE MATEMATICHE: Se il professore spiega una formula matematica, un'equazione o un teorema, DEVI obbligatoriamente ricostruire la formula esatta e scriverla nel testo utilizzando la sintassi LaTeX. Usa $ per le formule in linea e $$ per le formule centrate su una nuova riga. Non limitarti a raccontarla a parole.

    Estrai le informazioni da questo frammento e classificale usando ESATTAMENTE questi tre tag XML. Non usare titoli markdown, restituisci solo i tag compilati:
    
    <concetti>
    (Riassumi i temi chiave in 4 o 5 righe. REGOLA TASSATIVA: Scrivi in stile impersonale, come un libro di testo universitario. È SEVERAMENTE VIETATO usare parole come "blocco", "frammento", "lezione", "professore", "studente", o fare la telecronaca di cosa succede nel testo. Spiega direttamente la teoria senza mai annunciare cosa stai per spiegare).
    </concetti>
    
    <spiegazione>
    (Spiegazione dettagliata. Inizia esattamente da dove si era interrotto il contesto precedente, senza ripetere).
    </spiegazione>
    
    <digressioni>
    (Raccogli qui tutti gli aneddoti e le storie).
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
    Legge tutti i file .txt all'interno di una cartella, li ordina alfabeticamente 
    (es. parte1.txt, parte2.txt) e li unisce in un unico grande testo.
    """
    # Cerca tutti i file .txt e li mette in ordine alfabetico
    file_trovati = sorted(glob.glob(f"{cartella}/*.txt"))
    testo_totale = ""
    
    if not file_trovati:
        print(f"[!] Nessun file TXT trovato nella cartella '{cartella}'.")
        return ""
        
    for percorso_file in file_trovati:
        nome_file = os.path.basename(percorso_file)
        print(f"    - Aggiungo trascrizione: {nome_file}")
        
        with open(percorso_file, 'r', encoding='utf-8') as file:
            # Aggiunge il testo e un paio di a capo per separare nettamente le parti
            testo_totale += f"\n\n--- INIZIO {nome_file} ---\n\n"
            testo_totale += file.read() + "\n\n"
            
    return testo_totale

if __name__ == "__main__":
    app = costruisci_grafo()
    
    cartella_slide = "slide_lezione"  # <--- NUOVA CARTELLA
    cartella_trascrizioni = "testi_lezione" # <--- NUOVA CARTELLA PER I TESTI
    
    print(f"\n[FUSIONE] Lettura e unione delle trascrizioni nella cartella '{cartella_trascrizioni}'...")
    trascrizione_completa = estrai_testo_da_cartella_txt(cartella_trascrizioni)

    print(f"\n[RAG] Lettura di tutti i PDF nella cartella '{cartella_slide}'...")
    documenti_slide = estrai_documenti_da_cartella_pdf(cartella_slide)
    
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
                m3 = re.search(r"<digressioni>(.*?)</digressioni>", tg, re.DOTALL | re.IGNORECASE)
                
                if m1 and m1.group(1).strip(): sezione_1 += m1.group(1).strip() + "\n\n"
                if m2 and m2.group(1).strip(): sezione_2 += m2.group(1).strip() + "\n\n"
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
    salva_dispensa_html(sezione_1, sezione_2, sezione_3_pulita, "dispensa_perfetta.html")
    
    print("\n[SUCCESSO TOTALE] Pipeline completata. Apri 'dispensa_perfetta.html' nel browser e premi Ctrl+P per stampare in PDF!")