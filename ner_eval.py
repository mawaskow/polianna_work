'''
Evaluates the performance of the models.

Seqeval and Token scores
sghead and mhead
'''
import os
import evaluate
from datasets import DatasetDict
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
import json
from transformers import DataCollatorForTokenClassification
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, classification_report
from collections import Counter
from sghead_ner_ft import SgheadDataset, sghead_collate, sghead_evaluate_model
from mhead_ner_ft import MheadDataset, MheadTokenClassifier, mhead_collate, mhead_evaluate_model
from create_datasets import get_label_set
from auxil import bio_fixing, convert_numpy_torch_to_python

#############
# functions #
#############

def evaluate_model(mode, htype, model_name, model_save_addr, dsdct_dir, r, batch_size=16, dropout=0.1):
    label_list = get_label_set(mode,htype)
    if htype == "mhead":
        label_list = ["O","B","I"]
        head_lst = get_label_set(mode, htype)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_addr = f"{model_save_addr}/{model_name.split('/')[-1]}_{r}"
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for i, l in enumerate(label_list)}
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset_dict = DatasetDict.load_from_disk(f"{dsdct_dir}/dsdct_r{r}")
    if htype == "sghead":
        if model_name == "answerdotai/ModernBERT-base":
            test_dataset = SgheadDataset(dataset_dict["test"], tokenizer, label2id, max_length=2048)
        else:
            test_dataset = SgheadDataset(dataset_dict["test"], tokenizer, label2id)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda b: sghead_collate(b, tokenizer.pad_token_id)
        )
        model = AutoModelForTokenClassification.from_pretrained(model_addr).to(dev)
        metrics, preds, reals = sghead_evaluate_model(model, test_loader, dev, id2label, return_rnp=True)
    elif htype == "mhead":
        if model_name == "answerdotai/ModernBERT-base":
            test_dataset = MheadDataset(dataset_dict["test"], head_lst, tokenizer, label2id, max_length=2048)
        else:
            test_dataset = MheadDataset(dataset_dict["test"], head_lst, tokenizer, label2id)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda b: mhead_collate(b, tokenizer.pad_token_id)
        )
        model = MheadTokenClassifier(model_name, head_lst, dropout = dropout).to(dev)
        model.load_state_dict(torch.load(model_addr+"/model.pt", weights_only=True))
        model.eval()
        metrics, preds, reals = mhead_evaluate_model(model, test_loader, dev, id2label, return_rnp=True)
    return metrics, preds, reals

#all_preds, all_labels, all_inputids

################## sghead seqeval #############################

def visualize_run_sghead_seqeval_results(mode, eval, model_name, r, results_dir, idas=[0]):
    '''
    Currently just for sghead and seqeval
    For each ida (test set entity integer), for each token in the test set entity,
    prints the token next to the prediction next to the label
    For error evaluation
    
    :param mode: sghead or mhead
    :param eval: seqeval or token micro f1
    :param model_name: huggingface model name
    :param r: which model run to look at
    :param results_dir: directory where results files are saved
    :param idas: list of test set entity integers to examine
    '''
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    with open(f"{results_dir}/{mode}/{eval}_{model_name.split('/')[-1]}_{r}_pandr.json", "r", encoding="utf-8") as f:
        pandr = json.load(f)
    for ida in idas:
        tokens = tokenizer.convert_ids_to_tokens(pandr['input_ids'][ida])[1:-1]
        print(len(tokens),len(pandr['pred'][ida]), len(pandr['real'][ida]))
        for idx, tok in enumerate(tokens):
            print(f"{idx:03} | {tok:16} | "
                f"P:{pandr['pred'][ida][idx]:16}  "
                f"R:{pandr['real'][ida][idx]:16}  ")

def consol_sghead_seqeval_results(model_names=["microsoft/deberta-v3-base", "dslim/bert-base-NER-uncased"], r_vals=[0,1], results_dir="./"):
    '''
    Given the list of model names and r values, reads the results files and consolidates into a single results file
    
    :param model_names: list of model names whose results it will examine
    :param r_vals: list of r values for those model runs which it will examine
    :param results_dir: directory address where the results files are stored
    '''
    results_dict = {name:{} for name in model_names}
    for model_name in list(results_dict):
        results_dict[model_name]["Overall"] = {"precision":[], "recall":[], "f1":[], "accuracy":[]}
        for ftr in ["Actor", "InstrumentType", "Objective", "Resource", "Time"]:
            results_dict[model_name][ftr] = {"precision":[], "recall":[], "f1":[], "number":[]}
    for model_name in model_names:
        for r in r_vals:
            with open(f"{results_dir}/seqeval_{model_name.split('/')[-1]}_{r}_results.json","r", encoding="utf-8") as f:
                results = json.load(f)
            for k in list(results):
                if k[:4]=="over":
                    x, metric = k.split("_")
                    results_dict[model_name]['Overall'][metric].append(float(results[k][0]))
                else:
                    for mtr in list(results[k]):
                        results_dict[model_name][k][mtr].append(float(results[k][mtr][0]))
    with open(f"{results_dir}/seqeval_results.json","w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=4)
    return results_dict

############################# mhead tokf1

def tokf1_calc(htype, randp):
    if htype == "sghead":
        all_reals = []
        all_preds = []
        for art in randp:
            reals = [ent[0] for ent in art]
            preds = [ent[1] for ent in art]
            all_reals += reals
            all_preds += preds
        scores = {
            "micro_f1":f1_score(all_reals, all_preds, average='micro'),
            "macro_f1":f1_score(all_reals, all_preds, average="macro"),
            "weighted_f1":f1_score(all_reals, all_preds, average="weighted")
        }
        return scores
    elif htype == "mhead":
        label_lst = list(randp)
        scores_dct = {head:{} for head in label_lst}
        for head in label_lst:
            all_reals = []
            all_preds = []
            for art in randp[head]:
                reals = [ent[0] for ent in art]
                preds = [ent[1] for ent in art]
                all_reals += reals
                all_preds += preds
            scores_dct[head] = {
                "micro_f1":f1_score(all_reals, all_preds, average='micro'),
                "macro_f1":f1_score(all_reals, all_preds, average="macro"),
                "weighted_f1":f1_score(all_reals, all_preds, average="weighted")
            }
        return scores_dct

def tokf1_for_mhead(predictions_dct, labels_dct):
    '''
    Helper function for get_mhead_tokf1
    Generates results dictionary for provided preds and labels for tokf1 on mhead model
    
    :param predictions: dict of lists of lists of predictions of labels for each test entry for each featurename
    :param labels: dict of lists of lists of labels for each test entry for each featurename
    '''
    results = {}
    # for each head
    for head_name, preds_lists in predictions_dct.items():
        labels_lists = labels_dct[head_name] # size (batch, seq_len)
        labels_flat = [lbl for lbl_lst in labels_lists for lbl in lbl_lst]
        preds_flat = [pred for pred_lst in preds_lists for pred in pred_lst]
        # micro F1
        f1 = f1_score(labels_flat, preds_flat, average='micro') # pretty sure this is redundant
        clsf_rpt = classification_report(labels_flat, preds_flat, output_dict=True)
        results[f"{head_name}"] = clsf_rpt
        results[f"{head_name}"]["micro_f1"] = f1
    return results

def visualize_run_mhead_tokf1_results(mode, eval, model_name, r, results_dir, idas=[0]):
    '''
    Currently just for sghead and seqeval
    For each ida (test set entity integer), for each token in the test set entity,
    prints the token next to the prediction next to the label
    For error evaluation
    
    :param mode: sghead or mhead
    :param eval: seqeval or token micro f1
    :param model_name: huggingface model name
    :param r: which model run to look at
    :param results_dir: directory where results files are saved
    :param idas: list of test set entity integers to examine
    '''
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    with open(f"{results_dir}/{mode}/{eval}_{model_name.split('/')[-1]}_{r}_pandr.json", "r", encoding="utf-8") as f:
        pandr = json.load(f)
    for ida in idas:
        tokens = tokenizer.convert_ids_to_tokens(pandr['input_ids'][ida])[1:-1]
        #print(len(tokens),len(pandr['pred'][head][ida]), len(pandr['real'][head][ida]))
        for idx, tok in enumerate(tokens):
            try:
                print(f"{idx:03} | {tok:16} | "
                    f"| {pandr['real']['Actor'][ida][idx]:1}  "
                    f"{pandr['pred']['Actor'][ida][idx]:1}  "
                    f"| {pandr['real']['InstrumentType'][ida][idx]:1}  "
                    f"{pandr['pred']['InstrumentType'][ida][idx]:1}  "
                    f"| {pandr['real']['Objective'][ida][idx]:1}  "
                    f"{pandr['pred']['Objective'][ida][idx]:1}  "
                    f"| {pandr['real']['Resource'][ida][idx]:1}  "
                    f"{pandr['pred']['Resource'][ida][idx]:1}  "
                    f"| {pandr['real']['Time'][ida][idx]:1}  "
                    f"{pandr['pred']['Time'][ida][idx]:1}  ")
            except IndexError:
                pass

def shortestvis_tokf1(results_dict):
    '''
    Given *consolidated* results dictionary, prints only the rounded, averaged values across runs
    
    :param results_dict: results dictionary passed from consol_sghead_seqeval_results or loaded from file
    '''
    for m in list(results_dict):
        print(f"\n{m}")
        for res in list(results_dict[m]):
            print(f"\n{res}")
            df = pd.DataFrame(results_dict[m][res])
            df.loc['mean'] = df.mean()            
            df.loc['sd'] = df.std()
            print(round(df.loc['mean']*100,2))
            print(round(df.loc['sd']*100,2))

########################### other

def df_vis_consol_sghead_seqeval(results_dict):
    '''
    Given *consolidated* results dictionary, prints in dataframe format
    
    :param results_dict: results dictionary passed from consol_sghead_seqeval_results or loaded from file
    '''
    for m in list(results_dict):
        print(f"\n{m}")
        for res in list(results_dict[m]):
            print(f"\n{res}")
            df = pd.DataFrame(results_dict[m][res])
            df.loc['mean'] = df.mean()
            print(df)

def shortestvis(results_dict):
    '''
    Given *consolidated* results dictionary, prints only the rounded, averaged values across runs
    
    :param results_dict: results dictionary passed from consol_sghead_seqeval_results or loaded from file
    '''
    for m in list(results_dict):
        print(f"\n{m}")
        for res in list(results_dict[m]):
            print(f"\n{res}")
            df = pd.DataFrame(results_dict[m][res])
            df.loc['mean'] = df.mean()
            df.loc['sd'] = df.std()
            print(round(df.loc['mean']*100,2))
            print(round(df.loc['sd']*100,2))

########
# main #
########

def prettify_randp(htype, reals, preds):
    if htype == "sghead":
        randp = []
        for artind in range(len(reals)):
            arttrk = []
            for tokid in range(len(preds[artind])):
                arttrk.append((reals[artind][tokid], preds[artind][tokid]))
            randp.append(arttrk)
    elif htype == "mhead":
        heads = list(reals)
        randp = {head:[] for head in heads}
        for head in heads:
            for artind in range(len(reals[head][0])):
                arttrk = []
                for tokid in range(len(preds[head][0][artind])):
                    arttrk.append((reals[head][0][artind][tokid], preds[head][0][artind][tokid]))
                randp[head].append(arttrk)
    return randp

def extract_results(mode, metrics_dct, htype, eval_type):
    label_list = get_label_set(mode, "mhead")
    results_dict = {lbl: {} for lbl in label_list}
    if eval_type == "seqeval":
        if htype =="sghead":
            for lbl in label_list:
                try:
                    for mtr in list(metrics_dct[lbl]):
                        results_dict[lbl][mtr] = metrics_dct[lbl][mtr]
                except Exception as e:
                    print(e)
                    for mtr in ["precision","recall","f1","number"]:
                        results_dict[lbl][mtr] = 0

        elif htype == "mhead":
            for lbl in label_list:
                try:
                    for mtr in list(metrics_dct[lbl]["_"]):
                        results_dict[lbl][mtr] = metrics_dct[lbl]["_"][mtr]
                except:
                    for mtr in list(metrics_dct[lbl]):
                        results_dict[lbl][mtr] = metrics_dct[lbl][mtr]
    return results_dict

def count_labels(htype, randp):
    if htype == "sghead":
        all_reals = []
        all_preds = []
        for art in randp:
            reals = [ent[0] for ent in art]
            preds = [ent[1] for ent in art]
            all_reals += reals
            all_preds += preds
        count_dct = {
            "reals": dict(Counter(all_reals)),
            "preds": dict(Counter(all_preds))
        }
        return count_dct
    elif htype == "mhead":
        label_lst = list(randp)
        count_dct = {head:{} for head in label_lst}
        for head in label_lst:
            all_reals = []
            all_preds = []
            for art in randp[head]:
                reals = [ent[0] for ent in art]
                preds = [ent[1] for ent in art]
                all_reals += reals
                all_preds += preds
            count_dct[head] = {
                "reals": dict(Counter(all_reals)),
                "preds": dict(Counter(all_preds))
            }
        return count_dct

def make_run_report(mode, htype, metrics_dct, randp, eval_type):
    metric_out = extract_results(mode, metrics_dct, htype, eval_type)
    count_dct = count_labels(htype, randp)
    if htype == "sghead":
        results_dct = {}
        results_dct['metrics'] = metric_out
        results_dct['counts'] = count_dct
        return results_dct
    elif htype == "mhead":
        label_lst = list(metric_out)
        results_dct = {head:{} for head in label_lst}
        for head in label_lst:
            results_dct[head]['metrics'] = metric_out[head]
            results_dct[head]['counts'] = count_dct[head]
        return results_dct

def make_model_report_template(mode, htype):
    if htype == "sghead":
        tags = get_label_set(mode, htype)
        heads = get_label_set(mode, "mhead")
        results_dct = {
            "metrics": {head:{mtr:[] for mtr in ["precision", "recall", "f1", "number"]} for head in heads},
            "counts": {
                "reals": {tag:[] for tag in tags},
                "preds": {tag:[] for tag in tags}
            }
        }
    elif htype == "mhead":
        tags = ["O", "B", "I"]
        heads = get_label_set(mode, "mhead")
        results_dct = {head:
            {"metrics": {mtr:[] for mtr in ["precision", "recall", "f1", "number"]},
            "counts": {
                "reals": {tag:[] for tag in tags},
                "preds": {tag:[] for tag in tags}
            }
            } for head in heads
        }
    return results_dct

def make_model_report(mode, htype, results_file_prefix, r_lst):
    report = make_model_report_template(mode, htype)
    for r in r_lst:
        with open(f"{results_file_prefix}{r}.json", "r", encoding="utf-8") as f:
            results_dct = json.load(f)
            if htype == "sghead":
                for head in list(results_dct['metrics']):
                    for mtr in list(results_dct['metrics'][head]):
                        report['metrics'][head][mtr].append(results_dct['metrics'][head][mtr])
                for tag in list(report['counts']['reals']):
                    try:
                        report['counts']['reals'][tag].append(results_dct['counts']['reals'][tag])
                    except:
                        report['counts']['reals'][tag].append(0)
                    try:
                        report['counts']['preds'][tag].append(results_dct['counts']['preds'][tag])
                    except:
                        report['counts']['preds'][tag].append(0)
            elif htype == "mhead":
                for head in list(results_dct):
                    for mtr in list(results_dct[head]['metrics']):
                        try:
                            report[head]['metrics'][mtr].append(results_dct[head]['metrics'][mtr])
                        except:
                            metric = mtr.split("_")[-1]
                            if metric != "accuracy":
                                report[head]['metrics'][mtr.split("_")[-1]].append(results_dct[head]['metrics'][mtr])
                            print(f"Had to accomodate {mtr} metric type")
                    for tag in ["B", "I", "O"]:
                        try:
                            report[head]['counts']['reals'][tag].append(results_dct[head]['counts']['reals'][tag])
                        except:
                            report[head]['counts']['reals'][tag].append(0)
                        try:
                            report[head]['counts']['preds'][tag].append(results_dct[head]['counts']['preds'][tag])
                        except:
                            report[head]['counts']['preds'][tag].append(0)
    return report

def newmetrics_make_model_report(mode, htype, results_file_prefix, r_lst):
    heads = get_label_set(mode, "mhead")
    report = {}
    for r in r_lst:
        with open(f"{results_file_prefix}{r}.json", "r", encoding="utf-8") as f:
            results_dct = json.load(f)
            if htype == "sghead":
                for head in list(results_dct['metrics']):
                    for mtr in list(results_dct['metrics'][head]):
                        report['metrics'][head][mtr].append(results_dct['metrics'][head][mtr])
            elif htype == "mhead":
                for head in list(results_dct):
                    for mtr in list(results_dct[head]['metrics']):
                        try:
                            report[head]['metrics'][mtr].append(results_dct[head]['metrics'][mtr])
                        except:
                            metric = mtr.split("_")[-1]
                            if metric != "accuracy":
                                report[head]['metrics'][mtr.split("_")[-1]].append(results_dct[head]['metrics'][mtr])
                            print(f"Had to accomodate {mtr} metric type")
    return report

def display_model_report(mode, htype, model_report):
    if htype == "sghead":
        print("\nMetrics")
        for head in list(model_report['metrics']):
            print(head)
            for mtr in list(model_report['metrics'][head]):
                print(f"\t{mtr}")
                if mtr != "number":
                    print("\t\tAverage", round(np.mean(model_report['metrics'][head][mtr])*100,2))
                    print("\t\tSD", round(np.std(model_report['metrics'][head][mtr])*100,2))
                else:
                    print("\t\tAverage", round(np.mean(model_report['metrics'][head][mtr])))
                    print("\t\tSD", round(np.std(model_report['metrics'][head][mtr])))
        print("\nCounts")
        for tag in list(model_report['counts']['reals']):
            print(f"\t{tag}: real vs pred")
            print("\t\t",round(np.mean(model_report['counts']['reals'][tag])), round(np.mean(model_report['counts']['preds'][tag])))
    elif htype == "mhead":
        for head in list(model_report):
            print("\n",head)
            print("\tMetrics")
            for mtr in list(model_report[head]['metrics']):
                print(f"\t{mtr}")
                if mtr != "number":
                    print("\t\tAverage", round(np.mean(model_report[head]['metrics'][mtr])*100,2))
                    print("\t\tSD", round(np.std(model_report[head]['metrics'][mtr])*100,2))
                else:
                    print("\t\tAverage", round(np.mean(model_report[head]['metrics'][mtr])))
                    print("\t\tSD", round(np.std(model_report[head]['metrics'][mtr])))
            print("\tCounts")
            for tag in ["B", "I", "O"]:
                print(f"\t{tag}: real vs pred")
                print("\t\t",round(np.mean(model_report[head]['counts']['reals'][tag])), round(np.mean(model_report[head]['counts']['preds'][tag])))
                
def main():
    cwd = os.getcwd()
    results_dir = f"{cwd}/results"
    # get_results > prettify_results > consolidate_models > display_mdlrpt
    what_to_do = "display_mdlrpt"#"consolidate_models"#"prettify_results"#"get_results"#"consol_newmetrics"#"calc_newmetrics"#"test_seqeval"#"test_bifixing"#
    ######################################################
    if what_to_do == "get_results":
        for htype in ['mhead']:#, 'mhead']:
            for mode in ["a","b"]:#,"c","d"]:#,"e"]:
                dsdcts_dir = f"{cwd}/inputs/{mode}/{htype}_dsdcts"
                models_dir = f"{cwd}/models/{mode}/{htype}"
                for model_name in ["microsoft/deberta-v3-base","FacebookAI/xlm-roberta-base","dslim/bert-base-NER-uncased"]: #["answerdotai/ModernBERT-base"]
                    for r in list(range(5)):
                        print(f"\n-------- {htype} {mode} {model_name} {r} --------\n")
                        metrics, preds, reals = evaluate_model(mode, htype, model_name, models_dir, dsdcts_dir, r)
                        with open(f"{results_dir}/{mode}/{htype}/metrics_{model_name.split('/')[-1]}_{r}.json","w", encoding="utf-8") as f:
                            json.dump(convert_numpy_torch_to_python(metrics), f, indent=4)
                        print(metrics)
                        randp = prettify_randp(htype, reals, preds)
                        with open(f"{results_dir}/{mode}/{htype}/randp_{model_name.split('/')[-1]}_{r}.json","w", encoding="utf-8") as f:
                            json.dump(randp, f, indent=4)
    elif what_to_do == "prettify_results":
        for htype in ['mhead']:#, 'mhead']:
            for mode in ["a","b"]:#,"c","d","e"]:
                for model_name in ["microsoft/deberta-v3-base","FacebookAI/xlm-roberta-base","dslim/bert-base-NER-uncased"]:#, "answerdotai/ModernBERT-base"]:
                    for r in list(range(5)):
                        print(f"\n-------- {htype} {mode} {model_name} {r} --------\n")
                        with open(f"{results_dir}/{mode}/{htype}/metrics_{model_name.split('/')[-1]}_{r}.json", "r", encoding="utf-8") as f:
                            metrics = json.load(f)
                        with open(f"{results_dir}/{mode}/{htype}/randp_{model_name.split('/')[-1]}_{r}.json", "r", encoding="utf-8") as f:
                            randp = json.load(f)
                        try:
                            report = make_run_report(mode, htype, metrics, randp, "seqeval")
                            with open(f"{results_dir}/{mode}/{htype}/report_{model_name.split('/')[-1]}_{r}.json","w", encoding="utf-8") as f:
                                json.dump(report, f, indent=4)
                        except Exception as e:
                            print(f"{htype} {mode} {model_name} {r} FAILED")
                            print(e)
    elif what_to_do == "consolidate_models":
        for htype in ['mhead']:#, 'mhead']:
            for mode in ["a","b"]:#,"c","d","e"]:
                for model_name in ["microsoft/deberta-v3-base","FacebookAI/xlm-roberta-base","dslim/bert-base-NER-uncased"]:#, "answerdotai/ModernBERT-base"]:
                    print(f"\n-------- {htype} {mode} {model_name} --------\n")
                    results_file_prefix = f"{results_dir}/{mode}/{htype}/report_{model_name.split('/')[-1]}_"
                    r_lst = list(range(5))
                    report = make_model_report(mode, htype, results_file_prefix, r_lst)
                    with open(f"{results_dir}/{mode}/{htype}/model_report_{model_name.split('/')[-1]}.json","w", encoding="utf-8") as f:
                        json.dump(report, f, indent=4)
    elif what_to_do == "display_mdlrpt":
        for htype in ["mhead"]:#, 'mhead']:
            for mode in ["b"]:#,"b","c","d","e"]:
                for model_name in ["microsoft/deberta-v3-base"]:#,"FacebookAI/xlm-roberta-base","dslim/bert-base-NER-uncased"]:#, "answerdotai/ModernBERT-base"]:
                    print(f"\n-------- {htype} {mode} {model_name} --------\n")
                    with open(f"{results_dir}/{mode}/{htype}/model_report_{model_name.split('/')[-1]}.json","r", encoding="utf-8") as f:
                        model_report = json.load(f)
                    display_model_report(mode, htype, model_report)
    elif what_to_do == "calc_newmetrics":
        for htype in ['sghead', 'mhead']:
            for mode in ["a","b","c","d"]:
                for model_name in ["microsoft/deberta-v3-base","FacebookAI/xlm-roberta-base","dslim/bert-base-NER-uncased"]:
                    for r in list(range(5)):
                        print(f"\n-------- {htype} {mode} {model_name} {r} --------\n")
                        with open(f"{results_dir}/{mode}/{htype}/randp_{model_name.split('/')[-1]}_{r}.json", "r", encoding="utf-8") as f:
                            randp = json.load(f)
                        result = tokf1_calc(htype, randp)
                        with open(f"{results_dir}/{mode}/{htype}/newmetrics_{model_name.split('/')[-1]}_{r}.json","w", encoding="utf-8") as f:
                            json.dump(result, f, indent=4)
    elif what_to_do == "consol_newmetrics":
        for htype in ['sghead', 'mhead']:
            for mode in ["a","b","c","d"]:
                for model_name in ["microsoft/deberta-v3-base","FacebookAI/xlm-roberta-base","dslim/bert-base-NER-uncased"]:
                    print(f"\n-------- {htype} {mode} {model_name} --------\n")
                    results_file_prefix = f"{results_dir}/{mode}/{htype}/newmetrics_{model_name.split('/')[-1]}_"
                    r_lst = list(range(5))
                    report = newmetrics_make_model_report(mode, htype, results_file_prefix, r_lst)
                    with open(f"{results_dir}/{mode}/{htype}/mewmetrics_model_report_{model_name.split('/')[-1]}.json","w", encoding="utf-8") as f:
                        json.dump(report, f, indent=4)


if __name__=="__main__":
    main()
