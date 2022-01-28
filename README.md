# _gdTCR-MOCCA_

<img src="https://user-images.githubusercontent.com/73899443/151515795-2c45bcef-43d8-4958-9fa8-4adccffc2eb4.jpg" width="650">

### Anstehend
- ~~Ähnlichkeit der doppelt gesorteten Patienten bestimmen~~
- ~~Mapping der Patienten und eCRF Nummern~~
- ~~Modelle für alle vier Kombinationen der doppelt gesorteten (ds) Patienten generieren~~
  - ~~absolute Häufigkeiten~~
  - ~~relative Häufigkeiten~~
  - ~~_freq_ als Zählmaß~~
  - ~~Wie verteilen sich die ds Patienten auf die Cluster?~~
  - ~~Literaturrecherche: Gewichteter Similaritätsscore/MHI, Readcounts~~
  - Antwort aus Hannover abwarten ▶️ Entscheidung über ds Patienten
 

### Pipeline 2.0
- ~~Test-Train Split~~
- Distanzen-Verteilung
- kNN Graph / ~~KDTree~~
- Clustering
  - ~~Louvain~~
  - Spectral
- ~~Training-Cluster-Profile~~
- kNN Classifier
  - ~~Mode of neighbors~~
  - ...
- ~~Logistic Regression~~
  - Grid Search für gamma
  - Klassenbestimmung (BL vs HD, BL vs FU, ...)
- Cluster Analysen

### Literatur
[Graph Adjacency Matrix](https://ieeexplore.ieee.org/document/8659769)

[MART-1 Sequenzen Paper](https://github.com/donEnno/gamma_delta/files/7957292/Sci.Immunol.Benveniste_eaav4036.full.pdf)

[MART-1 Supplementary](https://github.com/donEnno/gamma_delta/files/7957293/SM_Generation.and.molecular.recognition.of.mel.pdf)

### Spectral Clustering
[Spectral Clusterin I](https://towardsdatascience.com/spectral-clustering-aba2640c0d5b), [Spectral Clustering II](https://towardsdatascience.com/spectral-graph-clustering-and-optimal-number-of-clusters-estimation-32704189afbe), [Luxburg - Spectral Clustering Tutorial](http://www.tml.cs.uni-tuebingen.de/team/luxburg/publications/Luxburg07_tutorial.pdf), [Spectral Clusterin III](http://web.cs.ucla.edu/~yzsun/classes/2017Winter_CS249/Slides/Clustering2.pdf)




Wichtigkeit: OS -> PFS -> Response

#### Wichtige Infos über die signifikanten Cluster
- Unique Sequenzen
- Patienten-Verteilung
- Delta-Ketten-Verteilung


## Misc
- BL vs FU: Matching gemeinsamer Sequenzen
- MART-1 Sequenzen 
- Kern der Kohorten bestimmen (MHI / Overlap / Shannon (Ravens et al))
