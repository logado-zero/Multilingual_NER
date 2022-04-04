# Multilingual NER
## **1. Reference**
* The solution in thís repository based on the architecture published in the paper: 

  [Multilingual Named Entity Recognition Using Pretrained Embeddings, Attention Mechanism and NCRF](https://arxiv.org/abs/1906.09978)

* This repository contains solution of NER task based on PyTorch [reimplementation](https://github.com/huggingface/transformers) of 
[Google's TensorFlow repository for the BERT model](https://github.com/google-research/bert)
that was released together with the paper [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
by Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova.

* This code is referenced from github [sberbank-ai/ner-bert](https://github.com/sberbank-ai/ner-bert)
## **2.Usage**
### **2.1. Prepare Dataset**
Dataset must be formatted in *csv* file with ```sep="\t"``` as follow:
| labels | text |
| :---   |:---     |
| O O B-LOC O O O	   | Cả nước Thái bàng hoàng .    |

Dataset folder must have 3 files : ```train.csv``` ```dev.csv``` ```test.csv```
### **2.2. Train model**
```python run_train.py --data_path /home/longdvg/Intern_Project/NER/data --bert_embedding False --embedder_name roberta-base```
### **2.3.Finetune**
```
python finetune.py --data_path /home/longdvg/Intern_Project/NER/data \
--checkpoint /home/longdvg/Intern_Project/NER/Multilingual_NER/check_point/RoBERTaBiLSTMAttnCRF.cpt \
--bert_embedding False --embedder_name roberta-base
```
### **2.4. Predict**
```
python run_predict.py --data_path /home/longdvg/Intern_Project/NER/data/test.csv 
--idx2labels_path /home/longdvg/Intern_Project/NER/data/idx2labels.txt \
--model /home/longdvg/Intern_Project/NER/Multilingual_NER/check_point/RoBERTaBiLSTMAttnCRF.cpt \
--bert_embedding False --embedder_name roberta-base
``` 

