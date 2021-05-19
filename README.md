# _Methodenentwicklung zur Klassifikation des Patienten-Outcomes aufgrund von TCR Sequenzen_ (BA)

**Vergangene Meetings:** 
23.04.2021, 10:00
24.30.04.2021, 11:00
Siehe MSA_Survey.pptx
07.05.2021, 14:00
Siehe Graph_Survey.pptx

**Nächstes Meeting: 21.05.2021, 14:00**

## Outline
- Multiple Sequence Alignment in verschiedenen Modalitäten
- Sequenzen nach diesem MSA sortiereren/clustern = Spätere Features für das ML Modell
- ML Modell (lineare/logistische Methoden)
- Klassifikator zum Outcome eine Patienten
- I/O: TCR Sequenz/Patient gesund oder krank bzw. Prognose über Medikation

## TODO

- MSA mit Standardmatrizen/physiochemischen Matrizen, Distanzmatrizen :warning: :heavy_check_mark:
- Daten "von Hand" kurieren und dokumentieren, um mögliche "Querschlägersequenzen" ausfindig zu machen -> Qualität des MSA
- T-Coffee: Verschiedene MSAs für alle Patienten generieren und diese dann mergen / evtl. paarweise betrachten
- gd-TCR Struktur in PDB betrachten und mit gegebenen Sequenzen mappen :heavy_check_mark:
-	MSA mit verschiendene Matrizen --> Für MSAs paarweise Distanzen der Sequenzen berechnen. Abstände als Input für      tSNE/UMAP/force-directed graph verwenden :warning: :heavy_check_mark:
-	Änhlichkeitsmaße für MSAs recherchieren
-	Pipeline bauen (Prototyping): MSA, Distanzmatrizen, Klassifizierer
- Visualisierungen/Slides zur Präsentation generieren! :heavy_check_mark:


## "Warentest"
-	Kriterien in "Stiftung Warentest" genauer recherchieren, welche davon sind für uns wichtig?
-	TCoffe Resultat (extended library) als Distanzen für Clustering verwenden
-	Genaue Methodik von ClustaIO nachvollziehen (word2vec interessant?) --> HMM Profile?


## Done
- Jalview Dokumentation: Farbcodes, Formatierungsmöglichkeiten, ... 
- Referenzstruktur fixieren für Bindestellen --> Aligniere jede Sequenz auf Referenzsequenz :warning:
- Clustering (Louvain) in UMAP einfügen :heavy_check_mark:

