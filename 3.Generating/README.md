# MedReport Generating

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

**Important Note** You will have to edit the files to add your **HuggingFace Token**. Without it, you won't be able to run the code. Ask permissions for the different models on [HuggingFace](https://huggingface.co/). You can then also change the model_id in the code to chose any model you want (make sure you are in the Llama file for models based on Llama and on the Cohere file if you want a model based on Cohere).
