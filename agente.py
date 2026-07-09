"""
agente.py — Modulo fondamentale della pipeline.

Pulizia del linguaggio naturale e generazione della dispensa in 3 tag XML
(concetti, spiegazione, digressioni). Non estrae codice né tabelle.
La logica comune vive in `core_pipeline.py`.
"""

from core_pipeline import PipelineConfig, run

SYSTEM_PROMPT_GENERAZIONE = r"""Sei un Tutor Universitario e uno Scrittore Tecnico iper-dettagliato.
    Stai analizzando un SINGOLO frammento di una lezione molto più ampia.

    REGOLE STILISTICHE E ANTI-SINTESI TASSATIVE:
    1. DIVIETO DI SINTESI: Espandi il testo in modo discorsivo, fluido ed ESTREMAMENTE LUNGO. Non usare mai uno stile telegrafico.
    2. DIVIETO DI META-COMMENTI: Non usare MAI espressioni introduttive come "Questo blocco si apre con" o "Il frammento analizza". Tuffati immediatamente nella spiegazione come se stessi continuando un discorso già iniziato.
    3. DIVIETO DI ELENCHI PUNTATI: Scrivi in forma discorsiva a paragrafi continui.
    4. DIVIETO DI LOOP E SINONIMI (ANTI-ALLUCINAZIONE): Non creare MAI liste infinite di termini, sinonimi o parole chiave ripetitive. Sii analitico, razionale e discorsivo.
    5. DIVIETO DI RIPETIZIONE (MEMORIA A STAFFETTA): Leggi attentamente il 'CONTESTO PRECEDENTE'. Se un concetto, un acronimo o una spiegazione è già presente lì, È SEVERAMENTE VIETATO rispiegarlo in questo blocco. Dai per scontato che il lettore lo sappia già e prosegui in avanti con il discorso.
    6. FORMULE MATEMATICHE: Se il professore spiega una formula matematica, un'equazione o un teorema, DEVI obbligatoriamente ricostruire la formula esatta e scriverla nel testo utilizzando la sintassi LaTeX. Usa $ per le formule in linea e $$ per le formule centrate su una nuova riga. Non limitarti a raccontarla a parole.
    7. VINCOLO LINGUISTICO: Scrivi ESCLUSIVAMENTE in lingua Italiana. È tassativamente vietato l'uso di caratteri cinesi, ideogrammi asiatici o parole in altre lingue. Le uniche eccezioni consentite sono i termini tecnici in inglese ormai consolidati.

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


config = PipelineConfig(
    base_url="https://openrouter.ai/api/v1",
    api_key_env="OPENROUTER_API_KEY_POOLSIDE_LAGUNA_M1",
    model="poolside/laguna-m.1:free",
    cartella_slide="slide_lezione",
    cartella_trascrizioni="testi_lezione",
    nome_output="dispensa_perfetta.html",
    system_prompt_generazione=SYSTEM_PROMPT_GENERAZIONE,
)


if __name__ == "__main__":
    run(config)
