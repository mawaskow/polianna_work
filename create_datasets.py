'''
Creates the datasets. 
*****Currently only uses articles that are fewer than 512 tokens.
*****Currently only looking at Actor, InstrumentType, Objective, Resource, Time
Options are sghead and mhead
sghead: labels are B-class, I-class, O for all classes in one label set
mhead: labels are BIO format in a different label set for each class
'''
import json
import pandas as pd
import os
from datasets import Dataset, DatasetDict, load_from_disk

#############
# functions #
#############

def df_loading(pol_dir, col_sel=["Policy","Text","Tokens","Curation"]):
    '''
    Function that loads dataframe from POLIANNA pkl
    :param pol_dir: directory where preprocessed_dataframe.pkl is stored
    :param col_sel: columns of interest in df
    '''
    pol_df = pd.read_pickle(pol_dir+"/preprocessed_dataframe.pkl")[col_sel]
    return pol_df

def get_label_set(mode, method):
    if mode == "a": 
        labels = ["Actor", "InstrumentType", "Objective", "Resource", "Time"]
    elif mode == "b":
        labels = ["Policydesigncharacteristics", "Technologyandapplicationspecificity", "Instrumenttypes"]
    elif mode == "c": 
        labels = ["Actor", "InstrumentType", "TechnologySpecificity", "EnergySpecificity", "Compliance", "ApplicationSpecificity", "Reference", "Objective", "Time", "Resource"]
    elif mode == "d":
        labels = ["Actor", "InstrumentType", "Objective"]
    if method == "sghead":
        bio_labels = ["O"]
        for l in sorted(labels):
            bio_labels.append(f"B-{l}")
            bio_labels.append(f"I-{l}")
        return bio_labels
    elif method == "mhead":
        return labels

def span_to_sghead_lbls(mode, tokens, spans):
    '''
    Helper function for df_to_sghead_ds(df)
    For an article in the polianna dataframe, takes the list of token objects and the list of span objects
    and returns list of token labels (converted from BIO to integer)
    
    :param tokens: list of token objects for a single article
    :param spans: list of span objects for a single article
    :param label2id: maps bio labels to integer labels
    '''
    token_labels = ["O"] * len(tokens)
    for spn in spans:
        if mode in ["a", "c", "d"]:
            ftr_lst = get_label_set(mode, "mhead")
            ftr = spn.feature
            if ftr in ftr_lst:
                start_char = spn.start
                end_char = spn.stop
                inside_tokens = []
                for i, tok in enumerate(tokens):
                    tok_start = tok.start
                    tok_end = tok.stop
                    overlap = not (tok_end <= start_char or tok_start >= end_char)
                    if overlap:
                        inside_tokens.append(i)
                if inside_tokens:
                    token_labels[inside_tokens[0]] = f"B-{ftr}"
                    for i in inside_tokens[1:]:
                        token_labels[i] = f"I-{ftr}"
        elif mode == "b":
            lyr = spn.layer
            start_char = spn.start
            end_char = spn.stop
            inside_tokens = []
            for i, tok in enumerate(tokens):
                tok_start = tok.start
                tok_end = tok.stop
                overlap = not (tok_end <= start_char or tok_start >= end_char)
                if overlap:
                    inside_tokens.append(i)
            if inside_tokens:
                token_labels[inside_tokens[0]] = f"B-{lyr}"
                for i in inside_tokens[1:]:
                    token_labels[i] = f"I-{lyr}"
    return token_labels

def span_to_mhead_lbls(mode, name, tokens, spans):
    '''
    Helper function to df_to_mhead_dataset(df)
    For an article in the dataframe, for a specific feature type in the annotated spans,
    creates a list of token labels in bio format then converts to integers.
    
    :param feature_name: Name of feature whose BIO list is being created for the article
    :param tokens: list of token objects
    :param spans: list of span objects
    :param label2id: dictionary mapping labels to integers
    '''
    token_labels = ["O"] * len(tokens)
    for spn in spans:
        if mode in ["a", "c", "d"]:
            if spn.feature == name:
                start_char = spn.start
                end_char = spn.stop
                inside_tokens = []
                for i, tok in enumerate(tokens):
                    tok_start = tok.start
                    tok_end = tok.stop
                    overlap = not (tok_end <= start_char or tok_start >= end_char)
                    if overlap:
                        inside_tokens.append(i)
                if inside_tokens:
                    token_labels[inside_tokens[0]] = f"B"
                    for i in inside_tokens[1:]:
                        token_labels[i] = f"I"
        elif mode == "b":
            if spn.layer == name:
                start_char = spn.start
                end_char = spn.stop
                inside_tokens = []
                for i, tok in enumerate(tokens):
                    tok_start = tok.start
                    tok_end = tok.stop
                    overlap = not (tok_end <= start_char or tok_start >= end_char)
                    if overlap:
                        inside_tokens.append(i)
                if inside_tokens:
                    token_labels[inside_tokens[0]] = f"B"
                    for i in inside_tokens[1:]:
                        token_labels[i] = f"I"
    return token_labels

def df_to_ds(mode, htype, df):
    '''
    Converts pandas dataframe to huggingface dataset
    :param df: POLIANNA dataframe
    '''
    lbl_lst = get_label_set(mode, "mhead") # this variable is only used in mhead anyways
    datapoints = []
    for artid in df.index:
        tokens = df.loc[artid,"Tokens"]
        text = df.loc[artid,"Text"]
        spans = df.loc[artid,"Curation"]
        token_texts = [t.text for t in tokens]
        if htype == "sghead":
            token_level_labels = span_to_sghead_lbls(mode, tokens, spans)
            datapoints.append({
                "id": artid,
                "text": text,
                "tokens": token_texts,
                "ner_tags": token_level_labels
            })
        elif htype == "mhead":
            datapoint = {}
            datapoint['id'] = artid
            datapoint["text"] = text
            datapoint["tokens"] = token_texts
            for name in lbl_lst:
                token_level_labels = span_to_mhead_lbls(mode, name, tokens, spans)
                datapoint[f"labels_{name}"] = token_level_labels
            datapoints.append(datapoint)
    # return pd.DataFrame(datapoints)
    return Dataset.from_list(datapoints)

def create_ds(mode, htype, pol_dir, dir_addr):
    '''
    Creates dataset of polianna pkl, converts token and span lists to list of BIO labels converted to integers, 
    and saves new dataset and the list of labels in integer order in provided address.
    
    :param pol_dir: address of polianna pkl
    :param dir_addr: directory where to save dataset
    '''
    pol_df = df_loading(pol_dir)
    ds = df_to_ds(mode, htype, pol_df)
    ds.save_to_disk(dir_addr)
    print(f"Created dataset in {dir_addr}")

def create_dsdcts(dataset, dsdct_dir, r_list=[0]):
    for r in r_list:
        td_test = dataset.train_test_split(test_size=0.2, seed=r)
        train_dev = td_test['train'].train_test_split(test_size=0.25, seed=r)
        ds_dct = DatasetDict({"train":train_dev['train'], "dev":train_dev['test'], "test":td_test['test']})
        ds_dct.save_to_disk(f"{dsdct_dir}/dsdct_r{r}")
    print(f"Created {len(r_list)} dataset(s) in {dsdct_dir}")

########
# main #
########

def main():
    cwd = os.getcwd()
    pol_dir = cwd+"/src/d01_data"
    ### creates whole sghead and mhead datasets from original POLIANNA database
    for mode in ["a", "b", "c", "d"]:
        print(mode)
        create_ds(mode,"sghead", pol_dir, cwd+f"/inputs/{mode}/sghead_ds/")
        create_ds(mode, "mhead", pol_dir, cwd+f"/inputs/{mode}/mhead_ds")
        print(f"Made {mode} datasets")
    print("Made datasets")
    ### creates the dataset dictionaries for each r split from the sghead and mhead datasets
    for mode in ["a", "b", "c", "d"]:
        print(mode)
        sghead_ds = load_from_disk(cwd+f"/inputs/{mode}/sghead_ds")
        create_dsdcts(sghead_ds, cwd+f"/inputs/{mode}/sghead_dsdcts", list(range(5)))
        mhead_ds = load_from_disk(cwd+f"/inputs/{mode}/mhead_ds")
        create_dsdcts(mhead_ds, cwd+f"/inputs/{mode}/mhead_dsdcts", list(range(5)))
        print(f"Made {mode} dsdcts")
    print("Made datasetdcts")

if __name__=="__main__":
    main()