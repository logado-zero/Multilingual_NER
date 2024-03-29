import argparse
import os
import glob2
import torch
from modules.data import bert_data
from modules.models.ner_models import BERTBiLSTMAttnNCRF, AutoBiLSTMAttnNCRF
from modules.train.train import NerLearner
from modules.data.bert_data import get_data_loader_for_predict
from modules.analyze_utils.utils import bert_labels2tokens, voting_choicer
from modules.analyze_utils.plot_metrics import get_bert_span_report
from sklearn_crfsuite.metrics import flat_classification_report


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        default="/home/longdo/Documents/Intern_Project/data/NER/factRuEval-2016-master/test.csv",
        type=str,
        help="path to data file test.csv",
    )

    parser.add_argument(
        "--idx2labels_path",
        default="/home/longdo/Documents/Intern_Project/data/NER/factRuEval-2016-master/idx2labels.txt",
        type=str,
        help="path to file idx2labels.txt",
    )

    parser.add_argument(
        "--model",
        default="/home/longdo/Documents/Intern_Project/data/NER/factRuEval-2016-master/fre-BERTBiLSTMAttnCRF-fit_BERT-IO.cpt",
        type=str,
        help="model path",
    )

    parser.add_argument(
        "--bert_embedding",
        default="True",
        type=str,
        help="Check if use BERT Embedding or not",
    )

    parser.add_argument(
        "--embedder_name",
        default="xlm-roberta-base",
        type=str,
        help="Name embedder want to use ",
    )
    
    args = parser.parse_args()

    # seed_all(seed_value=args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('Model is running with',device)

    

    # Load data set
    if args.bert_embedding == "True":
        data = bert_data.LearnData.create(
            train_df_path=args.data_path,
            valid_df_path=None,
            idx2labels_path=args.idx2labels_path,
            clear_cache=False,
            batch_size=8,
            markup = "BIO",
            device=device
            )

        #Create model
        model = BERTBiLSTMAttnNCRF.create(len(data.train_ds.idx2label), crf_dropout=0.3, is_freeze=False, device=device)
    else :
        data = bert_data.LearnData.create(
            train_df_path=args.data_path,
            valid_df_path=None,
            idx2labels_path=args.idx2labels_path,
            clear_cache=False,
            batch_size=8,
            markup = "BIO",
            device=device,
            bert_embedding = False,
            model_name = args.embedder_name,
            )

        #Create model
        model = AutoBiLSTMAttnNCRF.create(len(data.train_ds.idx2label), crf_dropout=0.3, is_freeze=False, device=device,model_name=args.embedder_name)


    dl = get_data_loader_for_predict(data, df_path=data.train_ds.config["df_path"])
    #Build Learner
    learner = NerLearner(model, data, best_model_path= args.model, t_total=100 * len(data.train_dl), lr=0.00001)
    learner.load_model(device = device)

    #Predict
    preds = learner.predict(dl)

    #Display
    pred_tokens, pred_labels = bert_labels2tokens(dl, preds)
    true_tokens, true_labels = bert_labels2tokens(dl, [x.bert_labels for x in dl.dataset])

    tokens_report = flat_classification_report(true_labels, pred_labels, labels=["B-ORG","B-PER","B-LOC","B-MISC","I-ORG", "I-PER", "I-LOC","I-MISC"], digits=8)
    print(tokens_report)




    
