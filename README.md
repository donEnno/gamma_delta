# _gdTCR-MOCCA_

** 21.01. kommt Regierungspräsidium **

### Anstehend
- ~~Ähnlichkeit der doppelt gesorteten Patienten bestimmen~~
- ~~Mapping der Patienten und eCRF Nummern~~
- ~~Modelle für alle vier Kombinationen der doppelt gesorteten (ds) Patienten generieren~~
  - ~~absolute Häufigkeiten~~
  - ~~relative Häufigkeiten~~
  - ~~_freq_ als Zählmaß~~
  - ~~Wie verteilen sich die ds Patienten auf die Cluster?~~
  - Literaturrecherche: Gewichteter Similaritätsscore/MHI, Readcounts
  - Antwort aus Hannover abwarten ▶️ Entscheidung über ds Patienten
  
  
### Testweise kann ausprobiert werden
- Klassifikation der Response: BL vs HD
- Regression der PFS/OS
- Kern der Kohorten bestimmen (MHI / Overlap / Shannon (Ravens et al))
- Readcounts als Zählmaß


Wichtigkeit: OS -> PFS -> Response

#### Wichtige Infos über die signifikanten Cluster
- Unique Sequenzen
- Patienten-Verteilung
- Delta-Ketten-Verteilung


## Misc
- BL vs FU: Matching gemeinsamer Sequenzen
- MART-1 Sequenzen 
- 

## Fragen
@Matthias:
- Wald Test und Coefficient Sign: LogisticRegressionCV(cv=5) vs. LogisticRegression auf 5 Fold und dann mitteln?
- Thresholding on the graph construction from the DM

@Kilian & Nicola:
- Welche Patienten?
- Min. Size von Clustern -> Filtern?
