# _Methodenentwicklung zur Klassifikation des Patienten-Outcomes aufgrund von TCR Sequenzen_ (BA)

**Vergangene Meetings: 23.04.2021, 10:00**
Siehe MSA_Surevey.pptx

**Nächstes Meeting: 30.04.2021, 11:00**

## Outline
- Multiple Sequence Alignment in verschiedenen Modalitäten
- Sequenzen nach diesem MSA sortiereren/clustern = Spätere Features für das ML Modell
- ML Modell (lineare Methoden)
- Klassifikator zum Outcome eine Patienten
- I/O: TCR Sequenz/Patient gesund oder krank bzw. Prognose über Medikation

## TODO

- MSA mit Standardmatrizen/physiochemischen Matrizen
- Daten "von Hand" kurieren und dokumentieren, um mögliche "Querschlägersequenzen" ausfindig zu machen -> Qualität des MSA
- T-Coffee: Verschiedene MSAs für alle Patienten generieren und diese dann mergen / evtl. paarweise betrachten
- gd-TCR Struktur in PDB betrachten und mit gegebenen Sequenzen mappen :heavy_check_mark:
- T-Coffe vs. MAFFT vs. ClustalO "Warentest", auch Evaluationsmethoden
- Jalview Dokumentation: Farbcodes, Formatierungsmöglichkeiten, ...
- Visualisierungen/Slides zur Präsentation generieren!


## Anstehend

- Mit dem besten Alignment Gruppen von Sequenzen definieren und als Features nutzen
- Projektionen der Sequenzen visualisieren
- Pro Patient Häufigkeiten von Sequenzen in Gruppen als Features
- Diskussion: MSA über alle Patienten oder jeder einzeln
