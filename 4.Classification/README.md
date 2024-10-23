#MedReport Classification

For the final task, you can either run '**Dataset_Train_Test.py**' to create the dataset you will use to fine-tune and test LongFormer or you can use the file '**classification_data.json**' directly from [Data Classification](https://github.com/Benjamin-Poutout/MedReport-AI-Classifier/tree/main/2.Data/Data_Classification).

To create the Dataset, you will need the CSV file created when generating with any of the strategies/model, change the path to the dataset inside the run '**Dataset_Train_Test.py**', then run :

```bash
python Dataset_Train_Test.py
```

To fine-tune LongFormer, you can simply run :

```bash
python Fine_Tuning_LongFormer.py
```

If you want to fine-tune with your own data, use the dataset you will create before with '**Dataset_Train_Test.py**'.

Finally, to run inference with your fine-tuned model, simply run :

```bash
python LongFormer_Inference_Prediction.py
```
