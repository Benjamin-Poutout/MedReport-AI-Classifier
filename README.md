# MedReport-AI-Classifier

## Introduction
In this project, we present the MedReport AI Classifier, a set of tools and models designed to extract and generate medical case reports, and then classify them using fine-tuning between humans and AI. This project aims to evaluate how language models can understand and classify medical information based on its context.

## Install
Follow these instructions to install the necessary dependencies for the project.

```bash
git clone https://github.com/Benjamin-Poutout/MedReport-AI-Classifier
cd MedReport-AI-Classifier
pip install -r requirements.txt
```

Reconstruct MedReport

To reproduce the construction of MedReport, you need to perform the following steps:

1. **Extracting Case Reports** : In the [PubMed](https://github.com/Benjamin-Poutout/MedReport-AI-Classifier/tree/main/1.PubMed) directory, detailed steps are provided to extract medical cas reports needed for the MedReport dataset development.
2. **Using MedReport dataset** : You can use the already extracted data from the [Data](https://github.com/Benjamin-Poutout/MedReport-AI-Classifier/tree/main/2.Data) directory or create your own Dataset with the [PubMed](https://github.com/Benjamin-Poutout/MedReport-AI-Classifier/tree/main/1.PubMed) directory.
4. **Generate new entities / case reports** : The [Generating](https://github.com/Benjamin-Poutout/MedReport-AI-Classifier/tree/main/3.Generating) folder contains scripts that enable you to generate using various large language models using the MedReport dataset.
5. **Train and evaluate classification models**:  The [Classification](https://github.com/Benjamin-Poutout/MedReport-AI-Classifier/tree/main/4.Classification) folder contains scripts that enable you to train and evaluate LongFormer on the MedReport generations.

## Citation

```bibtex
@article{poutout2024medreport,
author={Poutout, Benjamin and Geogakopoulos, Iassonas and others},
title={MedReport AI Classifier: Un Outil pour l'Extraction et la Classification des Rapports Médicaux},
year={2024},
}
```
