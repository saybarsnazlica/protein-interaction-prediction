# Predicting Protein-Protein Interactions from Atomic Structural Features

## Purpose

Protein-protein interactions (PPIs) have an essential role in many
biological processes. The gold standard of experimental methods for identifying
PPIs is direct experimental observation by X-ray crystallography or cryo-
electron microscopy. However such experiments are time-consuming. There
is also a large gap between the number of proteins discovered so far and the
number of crystallized protein complexes. Therefore, developing computational
methods such as protein docking to predict PPIs better can fill this gap.
Although currently there are many protein-protein docking methods, our purpose
is to create a machine learning approach that emphasizes features that can be
extracted from proteins in a public database to predict PPIs.

## Methods

We used a histogram of atom contacts as protein structure
features. We created docked models from the Protein Data Bank (PDB) with the
PIPER docking software. We created a features matrix with the ratio of the
number of observed over expected atom contacts based on the interface atoms
and the accessible surface area of the protein (ASA). Then, we trained a
machine learning classifier using the native and docked conformations of protein
complexes. We used singular value decomposition (SVD) to reduce the high
dimensional features to lower dimensions. We ranked the models according to the
target score and calculated the receiver operating characteristic (ROC) curve
and area under the curve (AUC).

## Results

The classifier achieved a ROC AUC of 0.87 on the test set.

## Conclusion

We showed that by only using atom-based structural features,
it is possible to create a machine learning pipeline that reaches a high accuracy
of model discrimination.
