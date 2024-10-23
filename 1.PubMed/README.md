# Extract French Medical Case Reports

This directory contains the folder [Extracting-Case-Reports](https://github.com/Benjamin-Poutout/MedReport-AI-Classifier/tree/main/1.PubMed/Extracting-Case-Reports) used to extract all case reports and to create the dataset.

To download and execute all the code present in [Extracting-Case-Reports](https://github.com/Benjamin-Poutout/MedReport-AI-Classifier/tree/main/1.PubMed/Extracting-Case-Reports), you will need to have the file '**pubmed_search.txt**' containing all the PMCIDs from PubMed you want to extract on. This file is already available in the Git inside the [Data](https://github.com/Benjamin-Poutout/MedReport-AI-Classifier/edit/main/2.Data) but you can create your own depending on the data you want to extract (the texte has to be written in french).

**Important Note** : 

Note that when running the '**extract_reports.py**' file, several informations will be asked : 

The name of the xml_folder you will have to create, after doing manipulations, the name of the final folder containing the remaining case reports, the desired size of the files you want (you can chose to exclude files too big for example), the title_pattern which is a re.compile of the possible ways our title can be write and where unwanted_title is the general name of the title. (e.g You don't want to include the sections or sub-sections of an .xml file with the name Discussion).

**Running the code extract_reports.py** : 

Several folders will be created during the execution of the code '**extract_reports.py**', the main file relevant will be the '**cleaned_data_structure.json**' file.

```bash
python extract_reports.py
```
