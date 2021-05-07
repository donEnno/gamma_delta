# _Methodenentwicklung zur Klassifikation des Patienten-Outcomes aufgrund von TCR Sequenzen_ (BA)

**Vergangene Meetings:** 
23.04.2021, 10:00
24.30.04.2021, 11:00
Siehe MSA_Surevey.pptx

**Nächstes Meeting: 14.05.2021, 14:00**

## Outline
- Multiple Sequence Alignment in verschiedenen Modalitäten
- Sequenzen nach diesem MSA sortiereren/clustern = Spätere Features für das ML Modell
- ML Modell (lineare Methoden)
- Klassifikator zum Outcome eine Patienten
- I/O: TCR Sequenz/Patient gesund oder krank bzw. Prognose über Medikation

## TODO

- MSA mit Standardmatrizen/physiochemischen Matrizen, Distanzmatrizen :warning: :heavy_check_mark:
- Daten "von Hand" kurieren und dokumentieren, um mögliche "Querschlägersequenzen" ausfindig zu machen -> Qualität des MSA
- T-Coffee: Verschiedene MSAs für alle Patienten generieren und diese dann mergen / evtl. paarweise betrachten
- gd-TCR Struktur in PDB betrachten und mit gegebenen Sequenzen mappen :heavy_check_mark:
- T-Coffe vs. MAFFT vs. ClustalO "Warentest", auch Evaluationsmethoden :heavy_check_mark:
- Jalview Dokumentation: Farbcodes, Formatierungsmöglichkeiten, ... :heavy_check_mark:
- Referenzstruktur fixieren für Bindestellen --> Aligniere jede Sequenz auf Referenzsequenz :warning:
-	MSA mit verschiendene Matrizen --> Für MSAs paarweise Distanzen der Sequenzen berechnen. Abstände als Input für      tSNE/UMAP/force-directed graph verwenden :warning: :heavy_check_mark:
-	Änhlichkeitsmaße für MSAs recherchieren
- Visualisierungen/Slides zur Präsentation generieren! :heavy_check_mark:


## Anstehend

- Mit dem besten Alignment Gruppen von Sequenzen definieren und als Features nutzen
- Projektionen der Sequenzen visualisieren
- Pro Patient Häufigkeiten von Sequenzen in Gruppen als Features
- Diskussion: MSA über alle Patienten oder jeder einzeln
