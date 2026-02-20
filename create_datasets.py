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

ORIG_SPAN_WEIGHTS = {'Instrumenttypes': 0.16719409282700423,
 'Policydesigncharacteristics': 0.5456463367855773,
 'Technologyandapplicationspecificity': 0.28715957038741846,
 'InstrumentType': 0.16244725738396623,
 'Actor': 0.28869390103567316,
 'Time': 0.038262370540851555,
 'Compliance': 0.08769658611430764,
 'Reversibility': 0.0019179133103183737,
 'Reference': 0.05662639048714998,
 'Objective': 0.04756425009589567,
 'ApplicationSpecificity': 0.07431914077483698,
 'EnergySpecificity': 0.09148446490218642,
 'TechnologySpecificity': 0.1211162255466053,
 'InstrumentType_2': 0.0046988876102800154,
 'Resource': 0.02450134253931722,
 'end': 0.0006712696586114307,
 'Unspecified': 0.052742616033755275,
 'Authority_legislative': 0.015822784810126583,
 'Time_PolDuration': 0.0020138089758342925,
 'Time_InEffect': 0.005226313770617568,
 'Authority_monitoring': 0.03159762178749521,
 'Form_monitoring': 0.08539509014192559,
 'Time_Monitoring': 0.015487149980820868,
 'Reversibility_policy': 0.0019179133103183737,
 'Authority_default': 0.04502301495972382,
 'Ref_Strategy_Agreement': 0.01529535864978903,
 'Addressee_default': 0.09977943996931339,
 'Edu_Outreach': 0.015774836977368624,
 'RegulatoryInstr': 0.05902378212504795,
 'Ref_OtherPolicy': 0.03533755274261603,
 'Time_Compliance': 0.014863828154967396,
 'Addressee_sector': 0.06501726121979287,
 'Addressee_monitored': 0.023158803222094362,
 'Objective_QualIntention': 0.026658995013425394,
 'App_Other': 0.03768699654775604,
 'Energy_Other': 0.026371308016877638,
 'App_LowCarbon': 0.036632144227080936,
 'Tech_Other': 0.05072880705792098,
 'Energy_LowCarbon': 0.06511315688530879,
 'Tech_LowCarbon': 0.07038741848868431,
 'Subsidies_Incentives': 0.004890678941311853,
 'Addressee_resource': 0.006808592251630227,
 'Ref_PolicyAmended': 0.005993479094744917,
 'FrameworkPolicy': 0.01524741081703107,
 'TaxIncentives': 0.0037399309551208286,
 'VoluntaryAgrmt': 0.0034522439585730723,
 'Objective_QuantTarget': 0.0077675489067894135,
 'Resource_MonSpending': 0.005418105101649405,
 'Form_sanctioning': 0.0023014959723820483,
 'Resource_Other': 0.01735711545838128,
 'PublicInvt': 0.0046988876102800154,
 'Objective_QualIntention_noCCM': 0.012754123513617184,
 'RD_D': 0.0005753739930955121,
 'Objective_QuantTarget_noCCM': 0.0003835826620636747,
 'Resource_MonRevenues': 0.0017261219792865361,
 'TradablePermit': 0.007000383582662063,
 '': 0.0006712696586114307,
 'Time_Resources': 0.0006712696586114307,
 'Authority_established': 0.0014863828154967396}

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
        labels = ["Actor", 
            "InstrumentType", 
            "Objective", 
            "Resource", 
            "Time"
            ]
    elif mode == "b":
        labels = ['Addressee_default',
            'Addressee_monitored',
            'Addressee_resource',
            'Addressee_sector',
            'Authority_default',
            'Authority_established',
            'Authority_legislative',
            'Authority_monitoring',
            'Edu_Outreach',
            'Form_monitoring',
            'Form_sanctioning',
            'FrameworkPolicy',
            'Objective_QualIntention',
            'Objective_QualIntention_noCCM',
            'Objective_QuantTarget',
            'Objective_QuantTarget_noCCM',
            'PublicInvt',
            'RD_D',
            'Ref_OtherPolicy',
            'Ref_PolicyAmended',
            'Ref_Strategy_Agreement',
            'RegulatoryInstr',
            'Resource_MonRevenues',
            'Resource_MonSpending',
            'Resource_Other',
            'Reversibility_policy',
            'Subsidies_Incentives',
            'TaxIncentives',
            'Time_Compliance',
            'Time_InEffect',
            'Time_Monitoring',
            'Time_PolDuration',
            'Time_Resources',
            'TradablePermit',
            'Unspecified',
            'VoluntaryAgrmt'
        ] 
    elif mode == "c": 
        labels = ["Policydesigncharacteristics", 
            "Technologyandapplicationspecificity", 
            "Instrumenttypes"
            ]
    elif mode == "d":
        labels = ['Actor',
            'ApplicationSpecificity',
            'Compliance',
            'EnergySpecificity',
            'InstrumentType',
            'Objective',
            'Reference',
            'Resource',
            'Reversibility',
            'TechnologySpecificity',
            'Time'
        ]
    elif mode == "e":
        labels = ['Addressee_default',
            'Addressee_monitored',
            'Addressee_resource',
            'Addressee_sector',
            'App_LowCarbon',
            'App_Other',
            'Authority_default',
            'Authority_established',
            'Authority_legislative',
            'Authority_monitoring',
            'Edu_Outreach',
            'Energy_LowCarbon',
            'Energy_Other',
            'Form_monitoring',
            'Form_sanctioning',
            'FrameworkPolicy',
            'Objective_QualIntention',
            'Objective_QualIntention_noCCM',
            'Objective_QuantTarget',
            'Objective_QuantTarget_noCCM',
            'PublicInvt',
            'RD_D',
            'Ref_OtherPolicy',
            'Ref_PolicyAmended',
            'Ref_Strategy_Agreement',
            'RegulatoryInstr',
            'Resource_MonRevenues',
            'Resource_MonSpending',
            'Resource_Other',
            'Reversibility_policy',
            'Subsidies_Incentives',
            'TaxIncentives',
            'Tech_LowCarbon',
            'Tech_Other',
            'Time_Compliance',
            'Time_InEffect',
            'Time_Monitoring',
            'Time_PolDuration',
            'Time_Resources',
            'TradablePermit',
            'Unspecified',
            'VoluntaryAgrmt'
        ] 
    if method == "sghead":
        bio_labels = ["O"]
        for l in sorted(labels):
            bio_labels.append(f"B-{l}")
            bio_labels.append(f"I-{l}")
        return bio_labels
    elif method == "mhead":
        return labels

def identify_dup_spans(df, mode):
    '''
    Identifies the duplicate spans (at feature/layer level) in the dataset for each token
    Returns a set of the relevant span ids (of spans which are contained by other spans of the same feature/layer)
    
    :param df: Description
    :param mode: Description
    '''
    lbl_set = get_label_set(mode, "mhead")
    dup_ent_ids = []
    for art in df.index:
        for tokensp in df.loc[art, "Tokens"]:
            sc = tokensp.get_token_spans(annotators ='Curation')
            if sc:
                if len(sc)>1:
                    track = []
                    for s in sc:
                        if mode in ["c"]:
                            track.append((s.span_id, s.text, s.layer))
                        elif mode in ["a","d"]:
                            if s.feature in lbl_set:
                                track.append((s.span_id, s.text, s.feature))
                    # looping through the ids and texts of all the spans for that one token
                    track = sorted(track, key=lambda tup: len(tup[1]), reverse=True)
                    for idx, text, ent in track:
                        for idx2, text2, ent2 in track:
                            if idx!=idx2:
                                # if span includes another span at this token
                                if text2 in text:
                                    # and if they DO have the same layer
                                    if ent==ent2:
                                        # then add to dup span ids
                                        dup_ent_ids.append(idx2)
    return list(set(dup_ent_ids))

def deduped_df_fxn(df, mode):
    '''
    Returns an updated df with duplicate spans removed
    
    :param df: Description
    :param mode: Description
    '''
    df2 = df.copy()
    # remove irrelevant articles
    removal = [code for code in list(df2.index) if code.split("_")[-1] in ["front", "Whereas"]]  
    df2 = df2.drop(removal, axis=0)
    # remove duplicate spans
    id_set = identify_dup_spans(df2, mode)
    for art in df2.index:
        df2.loc[art, "Curation"] = list(filter(lambda span: span.span_id not in id_set, df2.loc[art, "Curation"]))
    return df2

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
        if mode in ["a","d"]:
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
        elif mode in ["c"]:
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
        elif mode in ["b","e"]:
            tag_lst = get_label_set(mode, "mhead")
            tag = spn.tag
            if tag in tag_lst:
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
                    token_labels[inside_tokens[0]] = f"B-{tag}"
                    for i in inside_tokens[1:]:
                        token_labels[i] = f"I-{tag}"
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
        if mode in ["a","d"]:
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
        elif mode in ["c"]:
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
        elif mode in ["b","e"]:
            if spn.tag == name:
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
    Converts the original polianna dataframe to huggingface dataset
    :param df: POLIANNA dataframe
    '''
    df = deduped_df_fxn(df, mode)
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

def create_hyptune_dsdct(dataset, dsdct_dir):
    train_dev = dataset.train_test_split(test_size=0.2, seed=9)
    ds_dct = DatasetDict({"train":train_dev['train'], "dev":train_dev['test']})
    ds_dct.save_to_disk(f"{dsdct_dir}/dsdct_hyptune")
    print(f"Created hyptune dataset in {dsdct_dir}")

########
# main #
########

def main():
    cwd = os.getcwd()
    pol_dir = cwd+"/src/d01_data"
    '''
    ### creates whole sghead and mhead datasets from original POLIANNA database
    for mode in ["a","b", "c", "d", "e"]:
        print(mode)
        create_ds(mode,"sghead", pol_dir, cwd+f"/inputs/{mode}/sghead_ds/")
        create_ds(mode, "mhead", pol_dir, cwd+f"/inputs/{mode}/mhead_ds")
        print(f"Made {mode} datasets")
    print("Made datasets")
    ### creates the dataset dictionaries for each r split from the sghead and mhead datasets
    for mode in ["a","b", "c", "d", "e"]:
        print(mode)
        sghead_ds = load_from_disk(cwd+f"/inputs/{mode}/sghead_ds")
        create_dsdcts(sghead_ds, cwd+f"/inputs/{mode}/sghead_dsdcts", list(range(5)))
        mhead_ds = load_from_disk(cwd+f"/inputs/{mode}/mhead_ds")
        create_dsdcts(mhead_ds, cwd+f"/inputs/{mode}/mhead_dsdcts", list(range(5)))
        print(f"Made {mode} dsdcts")
    print("Made datasetdcts")
    #### for hyperparameter tuning only
    '''
    
    for mode in ["a","b", "c", "d", "e"]:
        print(mode)
        sghead_ds = load_from_disk(cwd+f"/inputs/{mode}/sghead_ds")
        create_hyptune_dsdct(sghead_ds, cwd+f"/inputs/{mode}/sghead_dsdcts")
        mhead_ds = load_from_disk(cwd+f"/inputs/{mode}/mhead_ds")
        create_hyptune_dsdct(mhead_ds, cwd+f"/inputs/{mode}/mhead_dsdcts")
        print(f"Made {mode} dsdcts")
    print("Made datasetdcts")

if __name__=="__main__":
    main()