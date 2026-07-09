# AI University Transcriber & Formatter

Pipeline autonoma basata su **LangGraph** e **LangChain** che trasforma le trascrizioni grezze delle lezioni universitarie in dispense di altissima qualità, impaginate in HTML pronto per la stampa in PDF. Il sistema incrocia il parlato del docente con le slide (RAG) per ricostruire terminologia tecnica, formule LaTeX, codice sorgente e tabelle.

## Architettura

Ogni agente segue lo stesso grafo sequenziale (`StateGraph`): **correzione** della trascrizione fonetica guidata dalle slide → **generazione** della dispensa strutturata in tag XML. Il testo viene diviso in blocchi con overlap di sicurezza e processato in sequenza mantenendo una memoria di continuità tra un blocco e il successivo.

## I tre agenti

Il progetto offre tre script pensati per livelli crescenti di complessità. Ognuno legge da una propria coppia di cartelle (slide + trascrizioni) e produce un file HTML dedicato.

| Script | Uso tipico | Tag XML | Extra |
|---|---|---|---|
| `agente.py` | Lezioni discorsive (teoria pura) | `concetti`, `spiegazione`, `digressioni` | Base |
| `agente_codice.py` | Lezioni con codice sorgente | idem | Blocchi di codice con Highlight.js |
| `agente_esercizi.py` | Ammiraglia: esercizi + compilatori | + `esercizio` | Codice, tabelle, box esercizi guidati |

## Caratteristiche

* **Chunking intelligente**: divide il testo senza perdere il contesto grazie a un overlap di 150 parole; l'ultimo blocco è sempre pulito, senza frammenti orfani.
* **Motore RAG locale (BM25)**: converte le slide PDF/PPTX in Markdown (MarkItDown) e recupera le sezioni pertinenti a ogni blocco. Le fonti consultate vengono stampate a terminale. Le conversioni sono messe in cache su disco (`.cache/`) e riusate se il file non cambia.
* **Retriever ibrido opzionale**: se attivato (`usa_retriever_ibrido`, già on in `agente_esercizi.py`), affianca a BM25 un retriever semantico basato su embedding multilingua locali, fondendo i risultati con Reciprocal Rank Fusion — recupera le slide giuste anche con sinonimi/parafrasi. Richiede le dipendenze opzionali di `requirements-hybrid.txt`.
* **Memoria a staffetta**: trasporta gli ultimi 4000 caratteri del blocco precedente per garantire continuità narrativa ed evitare ripetizioni locali.
* **Cane da guardia XML**: se il modello non produce i tag richiesti (o l'API restituisce un errore/rate-limit), lo stesso blocco viene rielaborato automaticamente dopo una pausa, finché il formato non è corretto.
* **Prompt blindati (raw string)**: i system prompt sono dichiarati come raw string per preservare la sintassi LaTeX destinata a MathJax.
* **Rendering ricco**: MathJax per le formule, Highlight.js per il codice, tabelle e box grafici per esercizi e digressioni.
* **Revisore editoriale finale**: passata di deduplicazione che rimuove dalle digressioni gli aneddoti già presenti nella teoria.

## Come usarlo

1. Installa le dipendenze:
   ```bash
   pip install -r requirements.txt
   ```
   Solo per il retriever ibrido (usato da `agente_esercizi.py`), installa anche le dipendenze opzionali — al primo avvio scaricano un modello di embedding (~450MB):
   ```bash
   pip install -r requirements-hybrid.txt
   ```
2. Copia `.env.example` in `.env` e inserisci le tue API key personali (una per ciascun provider/modello che intendi usare).
3. Inserisci le slide (PDF/PPTX) e le trascrizioni (`.txt`/`.md`) nelle cartelle attese dall'agente scelto:
   * `agente.py` → `slide_lezione/` e `testi_lezione/`
   * `agente_codice.py` → `slide_codice_info/` e `testi_info/`
   * `agente_esercizi.py` → `slide_compilatori/` e `testi_compilatori/`
4. Avvia lo script desiderato, ad esempio:
   ```bash
   python agente_esercizi.py
   ```
5. Apri il file HTML generato nel browser e premi **Ctrl+P** per esportarlo in PDF.
