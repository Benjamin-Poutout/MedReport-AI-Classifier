# MedReport Generating

**Important Note** : You will have to edit the files to add your **HuggingFace Token**. Without it, you won't be able to run the code. Ask permissions for the different models on [HuggingFace](https://huggingface.co/). You can then also change the model_id in the code to chose any model you want (make sure you are in the Llama file for models based on Llama and on the Cohere file if you want a model based on Cohere).

You will find for each strategy of generation a folder [Plots and stats](https://github.com/Benjamin-Poutout/MedReport-AI-Classifier/tree/main/3.Generating/Chain%20of%20Thought%20Generation/Plots%20and%20Stats) that you can use to create a set of different statistics on the generations aswell as some visualizations.

**Requirements** : You need to copy the files 'inference_data.json' and 'Entities_to_Generate.csv' from the folder [Data](https://github.com/Benjamin-Poutout/MedReport-AI-Classifier/tree/main/2.Data).

## Extracting Entities ##

To generate the extraction through LLMs of entities, you can simply run :

```bash
cd /3.Generating
python entities_gen.py
```

Make sure first to copy the data in [Data_Extracting_Entities](https://github.com/Benjamin-Poutout/MedReport-AI-Classifier/blob/main/2.Data/Data_Extracting_Entities/inference_data.json) inside of [Generating](https://github.com/Benjamin-Poutout/MedReport-AI-Classifier/tree/main/3.Generating) before running the code.

## Generating through Chain Of Thought ##

To generate new medical case reports using the entities created previously, you can simply run :

```bash
python chain-of-thought-cohere.py
```

If you want to generate through a Cohere model, or :

```bash
python chain-of-thought-llama.py
```

If you want to generate through a Llama model.

## Generating through One Shot ##

To generate new medical case reports using the entities created previously, you can simply run :

```bash
python one-shot-cohere.py
```

If you want to generate through a Cohere model, or :

```bash
python one-shot-llama-3.py
```

If you want to generate through a Llama model.

## Generating through One Shot ##

To generate new medical case reports using the entities created previously, you can simply run :

```bash
python few-shot-cohere.py
```

If you want to generate through a Cohere model, or :

```bash
python few-shot-llama.py
```

If you want to generate through a Llama model.

## Generating through a Fine-Tuning of Llama-3.1-8B ##

To generate new medical case reports using a fine-tuning of Llama-3.1-8B, you have to either create your own train and test files with [Data_Fine_Tune](https://github.com/Benjamin-Poutout/MedReport-AI-Classifier/tree/main/2.Data/Data_Fine_Tune):

```bash
python Data_Format.py
```
Will create your train and test dataset based on the file containing the entities extracted and the human text corresponding.

To fine-tune the Llama-3.1-8B model, you can then just run :

```bash
python Llama-3.1-8B_fine_tuning.py
```

After the fine-tuning, you can use the model on inference mode with the code :

```bash
python Llama_3.1_8B_Inference_After_Fine_Tuning.py
```



