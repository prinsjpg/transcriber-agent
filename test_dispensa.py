"""
test_dispensa.py — Test di regressione sulla dispensa HTML generata.

Automatizza i controlli che altrimenti si farebbero a mano dopo ogni rigenerazione:
niente simboli "tofu" (PUA), niente pseudo-variabili Yacc nude nel testo, niente
slide-immagine duplicate, indice coerente con i titoli e numerazione progressiva,
niente diagrammi mermaid non convertiti, un solo <h1>.

Uso:
    python test_dispensa.py                      # controlla dispensa_compilatori.html
    python test_dispensa.py altra_dispensa.html  # controlla un file specifico

Esce con codice 0 se tutti gli invarianti sono rispettati, 1 altrimenti.
È anche compatibile con pytest: `pytest test_dispensa.py`.
"""

import sys
import os

from core_pipeline import verifica_invarianti_html

# File di default: l'output dell'agente esercizi.
DISPENSA_DEFAULT = "dispensa_compilatori.html"


def _percorso_dispensa() -> str:
    """Percorso della dispensa da controllare (argomento CLI o default)."""
    if len(sys.argv) > 1:
        return sys.argv[1]
    return DISPENSA_DEFAULT


def carica_html(percorso: str) -> str:
    with open(percorso, "r", encoding="utf-8") as f:
        return f.read()


def controlla(percorso: str) -> list[str]:
    """Ritorna la lista dei problemi trovati nella dispensa (vuota se tutto ok)."""
    return verifica_invarianti_html(carica_html(percorso))


def test_invarianti_dispensa():
    """Entry point per pytest: fallisce se la dispensa viola un invariante."""
    percorso = _percorso_dispensa()
    assert os.path.exists(percorso), f"Dispensa non trovata: {percorso}"
    problemi = controlla(percorso)
    assert not problemi, "Invarianti violati:\n" + "\n".join(f"  - {p}" for p in problemi)


def main() -> int:
    percorso = _percorso_dispensa()
    if not os.path.exists(percorso):
        print(f"[!] Dispensa non trovata: {percorso}")
        return 1

    print(f"[TEST] Controllo invarianti su: {percorso}")
    problemi = controlla(percorso)

    if not problemi:
        print("[✓] Tutti gli invarianti sono rispettati.")
        return 0

    print(f"[✗] {len(problemi)} invariante/i violato/i:")
    for p in problemi:
        print(f"    - {p}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
