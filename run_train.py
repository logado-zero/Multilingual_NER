import argparse
import os
import glob2
import torch
from modules.data import bert_data
from modules.models.ner_models import BERTBiLSTMAttnNCRF, AutoBiLSTMAttnNCRF
from modules.train.train import NerLearner


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        default="/home/longdo/Documents/Intern_Project/data/NER/factRuEval-2016-master",
        type=str,
        help="path to folder has data train and valid (dev.csv and test.csv)",
    )

    parser.add_argument(
        "--idx2labels_path",
        default=None,
        type=str,
        help="path to file idx2labels.txt",
    )

    parser.add_argument(
        "--num_epochs",
        default=100,
        type=int,
        help="num epochs required for training",
    )

    parser.add_argument(
        "--checkpoint",
        default=None,
        type=str,
        help="file save path for continuing training",
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


    # seed_all(seed_value=args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('Model is running with',device)

    #Set path data train and test file .csv
    args = parser.parse_args()
    for file in glob2.glob(args.data_path+ '/*'):
        if 'train.csv' in file:
            train_df_path = file
        elif 'dev.csv' in file:
            valid_df_path = file

    #Check idx2labels.txt path
    if args.idx2labels_path is None:
        idx2labels_path = args.data_path + "/idx2labels.txt"
    else: idx2labels_path = args.idx2labels_path

    #Check file save model
    if args.checkpoint is None:
        if not os.path.exists('check_point'):
            os.mkdir('check_point')
        check_point_path = os.path.abspath(os.getcwd()) + '/check_point/fre-BERTBiLSTMAttnCRF-fit_BERT-IO.cpt'
    else:
        check_point_path = args.checkpoint


    # Load data set
    if args.bert_embedding == "True":
        data = bert_data.LearnData.create(
            train_df_path=train_df_path,
            valid_df_path=valid_df_path,
            idx2labels_path=idx2labels_path,
            clear_cache=True,
            batch_size=8,
            markup = "BIO",
            device=device
            )

        #Create model
        model = BERTBiLSTMAttnNCRF.create(len(data.train_ds.idx2label), crf_dropout=0.3, is_freeze=False, device=device)
    else :
        data = bert_data.LearnData.create(
            train_df_path=train_df_path,
            valid_df_path=valid_df_path,
            idx2labels_path=idx2labels_path,
            clear_cache=True,
            batch_size=8,
            markup = "BIO",
            device=device,
            bert_embedding = False,
            model_name = args.embedder_name,
            )

        #Create model
        model = AutoBiLSTMAttnNCRF.create(len(data.train_ds.idx2label), crf_dropout=0.3, is_freeze=False, device=device,model_name=args.embedder_name)

    #Create Learner Class for model
    learner = NerLearner(
        model, data, best_model_path = check_point_path,
        t_total=args.num_epochs * len(data.train_dl), lr=0.0001,sup_labels= ["B-MISC","I-MISC","B-PER","I-PER","B-LOC","I-LOC","B-ORG","I-ORG"])

    print('Number params of model :',model.get_n_trainable_params())
    #Start training......
    learner.fit(epochs=args.num_epochs)





    

