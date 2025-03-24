# _gdTCR-MOCCA_

For anyone wondering: This is the repository to my bachelor thesis!


<img src="https://user-images.githubusercontent.com/73899443/151515795-2c45bcef-43d8-4958-9fa8-4adccffc2eb4.jpg" width="650">

### Distance approach
- Gridsearch mit Penalisierung (elasticnet)
- Cox-Regression
- Klassifikation von OS, PFS und responder vs. non-responder
- Consensus Sequenzen in UMAP anzeigen


### k-mer approach
- Pipeline fertigstellen
- Shannon vs. Simpson index
- Ähnliche k-mers zusammenfassen: Reduziertes AS Alphabet, Partitoinierung im Graph, ...
- Konkatenieren von unterschiedlichen k-mer Längen
- Differenz der k-mer counts in BL-FU Paaren als Feature für Cox-Regression

# Insgesamt die Analysen auch für delta2-negativen Datensatz ausprobieren.

### Literatur
[Graph Adjacency Matrix](https://ieeexplore.ieee.org/document/8659769)

[MART-1 Sequenzen Paper](https://github.com/donEnno/gamma_delta/files/7957292/Sci.Immunol.Benveniste_eaav4036.full.pdf)

[MART-1 Supplementary](https://github.com/donEnno/gamma_delta/files/7957293/SM_Generation.and.molecular.recognition.of.mel.pdf)

[Spectral Clusterin I](https://towardsdatascience.com/spectral-clustering-aba2640c0d5b), [Spectral Clustering II](https://towardsdatascience.com/spectral-graph-clustering-and-optimal-number-of-clusters-estimation-32704189afbe), [Luxburg - Spectral Clustering Tutorial](http://www.tml.cs.uni-tuebingen.de/team/luxburg/publications/Luxburg07_tutorial.pdf), [Spectral Clusterin III](http://web.cs.ucla.edu/~yzsun/classes/2017Winter_CS249/Slides/Clustering2.pdf)

[gdT coeliac disease k-mer paper](https://onlinelibrary.wiley.com/doi/10.1002/path.5592)

[Simpson vs. Shannon](https://www.sciencedirect.com/science/article/pii/S0958166920301051))

