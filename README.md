# AI University Transcriber & Formatter

Agente autonomo basato su LangGraph e LangChain che trasforma le trascrizioni grezze delle lezioni universitarie in dispense di altissima qualità, impaginate in HTML/PDF.

## Caratteristiche
* **Chunking Intelligente**: Divide il testo senza perdere il contesto (overlap).
* **Motore RAG Locale**: Analizza le slide (PDF) per correggere i termini tecnici.
* **Memoria a Staffetta**: Evita la ripetizione degli argomenti.
* **Revisore Editoriale**: Rimuove i doppioni e gli aneddoti ridondanti.
* **Cane da Guardia**: Resiste ai crash del server (502, 429, Timeout) riprovando in automatico.

## Come usarlo
1. Rinomina il file `.env.example` in `.env` e inserisci la tua API Key.
2. Inserisci le tue slide come `slide.pdf` e la trascrizione come `trascrizione.txt`.
3. Avvia lo script.