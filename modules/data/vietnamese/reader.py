import pandas as pd
import os

def get_file_names(path):
    res = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.src') :
                res.append([dirs, os.path.splitext(file)[0],os.path.join(root, os.path.splitext(file)[0])])
    return res

def to_data_frame(src_path,des_path):
    train_src=train_tgt=test_src=test_tgt = []
    for root, name, dir in get_file_names(src_path):
        with open(dir+'.src') as f:
            src = f.readlines()
        with open(dir+'.tgt') as f:
            tgt = f.readlines()
        if name == 'train':
            train_src =src
            train_tgt = tgt
        elif name == ' test':
            test_src = src
            test_tgt = tgt
    print(root)
    train = pd.DataFrame({"labels": map(lambda l: l.strip(),train_tgt), "text": map(lambda l: l.strip(), train_src)})
    train["clf"] = train["labels"].apply(lambda x: all([y.split("_")[0] == "O" for y in x.split()]))
    train.to_csv(des_path+"/train.csv", index=False, sep="\t")

    test = pd.DataFrame({"labels": map(lambda l: l.strip(),test_tgt), "text": map(lambda l: l.strip(), test_src)})
    test["clf"] = train["labels"].apply(lambda x: all([y.split("_")[0] == "O" for y in x.split()]))
    test.to_csv(des_path+"/test.csv", index=False, sep="\t")


        


