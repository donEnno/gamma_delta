# _gdTCR-MOCCA_

### Anstehend
- ~~Ähnlichkeit der doppelt gesorteten Patienten bestimmen~~
- ~~Mapping der Patienten und eCRF Nummern~~
- ~~Modelle für alle vier Kombinationen der doppelt gesorteten (ds) Patienten generieren~~
  - ~~absolute Häufigkeiten~~
  - ~~relative Häufigkeiten~~
  - ~~_freq_ als Zählmaß~~
  - Wie verteilen sich die ds Patienten auf die Cluster?
  
**Auswahl der ds Patienten in Rücksprache mit Nicola und Kilian**

Für die Modelle BLOSUM45, BLOSUM62, PAM70:
- Klassifikation der Response: BL vs HD
- Regression der PFS/OS

Wichtigkeit: OS -> PFS -> Response

#### Wichtige Infos über die signifikanten Cluster
- Unique Sequenzen
- Patienten-Verteilung
- Delta-Ketten-Verteilung


## Misc
- BL vs FU: Matching gemeinsamer Sequenzen
- MART-1 Sequenzen 
- Kern der Kohorten bestimmen (MHI / Overlap / Shannon (Ravens et al))
- Readcounts als Zählmaß
- Gewichteter Similaritätsscore/MHI, Readcounts


## Fragen
@Matthias:
- Wald Test und Coefficient Sign: LogisticRegressionCV(cv=5) vs. LogisticRegression auf 5 Fold und dann mitteln?
- Thresholding on the graph construction from the DM

@Kilian & Nicola:
- Welche Patienten?
- Min. Size von Clustern -> Filtern?
