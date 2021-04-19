# _Methodenentwicklung zur Klassifikation des Patienten-Outcomes aufgrund von TCR Sequenzen_ (BA)

**Nächstes Meeting: 23.04.2021, 10:00**


## Outline
- Multiple Sequence Alignment in verschiedenen Modalitäten

- Sequenzen nach diesem MSA sortiereren/clustern = Spätere Features für das ML Modell
- ML Modell (lineare Methoden)
- Klassifikator zum Outcome eine Patienten
- I/O: TCR Sequenz/Patient gesund oder krank bzw. Prognose über Medikation

## TODO

- Einarbeitung in die BLAST API
- ~~Debugging von Needleman-Wunsch-Implementierung~~
- Finden eines sinnvollen Ähnlichkeitsmaßes
- MSA mit Standardmatrizen
- MSA mit angepassten (phsyikochemisch motivierten) Matrizen
- Visualisierungen/Slides zur Präsentation generieren!
- Mit dem besten Alignment Gruppen von Sequenzen definieren und als Features nutzen
- Projektionen der Sequenzen visualisieren
- Pro Patient Häufigkeiten von Sequenzen in Gruppen als Features
