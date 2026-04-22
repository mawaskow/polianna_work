import os
import evaluate
from datasets import DatasetDict
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
import bitsandbytes as bnb
import json
from transformers import DataCollatorForTokenClassification
from torch.utils.data import DataLoader
from safetensors.torch import load_file
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, classification_report
from collections import Counter
from sghead_ner_ft import SgheadDataset, sghead_collate, sghead_evaluate_model
from mhead_ner_ft import MheadDataset, MheadTokenClassifier, mhead_collate, mhead_evaluate_model
from create_datasets import get_label_set, convert_tokens_to_entities
from auxil import bio_fixing, convert_numpy_torch_to_python
import ast
from glob import glob

BNB_CONFIG = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": torch.bfloat16,
    "bnb_4bit_use_double_quant": True,
    "llm_int8_skip_modules": ["classifier"]
}
TARGET_MODULES_DICT={
    "microsoft/deberta-v3-base":["query_proj","key_proj","value_proj","dense"],
    "FacebookAI/xlm-roberta-base":["query","key","value","dense"],
    "dslim/bert-base-NER-uncased":["query","key","value","dense"],
    "answerdotai/ModernBERT-base":["attn.Wqkv","attn.Wo","mlp.Wi","mlp.Wo"]
}

def consol_tc_model_runs(model_run_dir, r_n):
    all_model_dirs = glob(f"{model_run_dir}/*")
    mdl_dct = {r:{i:"" for i in list(range(3))} for r in list(range(r_n))}
    for model_dir in all_model_dirs:
        randi = model_dir.split("_")[-1]
        mdl_dct[int(randi.split("-")[0])][int(randi.split("-")[1])] = model_dir
    for r in list(mdl_dct):
        # determine best i in each r split based on lowest average eval loss
        final_i = 0
        best_el = 100
        for i in list(mdl_dct[r]):
            dn = mdl_dct[r][i]
            with open(f"{dn}/metrics.json", "r", encoding="utf-8") as f:
                mtrcs = json.load(f)
            if mtrcs["avg_eval_loss"] < best_el:
                best_el = mtrcs["avg_eval_loss"]
                final_i = i
        for i in list(mdl_dct[r]):
            dn = mdl_dct[r][i]
            if i == final_i:
                os.rename(dn, dn[:-2])
            else:
                for filename in os.listdir(dn):
                    file_path = os.path.join(dn, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                os.rmdir(dn)

def get_tc_rpi(mode, htype, model_name, model_save_addr, dsdct_dir, r, batch_size=16, dropout=0.1, max_length=512, quant=True):
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
        test_dataset = SgheadDataset(dataset_dict["test"], tokenizer, label2id, max_length=max_length)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda b: sghead_collate(b, tokenizer.pad_token_id)
        )
        if quant:
            compute_dtype = torch.float16
            # incorporating QLoRA
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                llm_int8_skip_modules=["classifier"]
            )
            base_model = AutoModelForTokenClassification.from_pretrained(
                model_name,
                num_labels=len(label_list),
                #id2label=id2label,
                #label2id=label2id,
                quantization_config=bnb_config,
                ignore_mismatched_sizes=(model_name == "dslim/bert-base-NER-uncased"),
                device_map="auto"
            )
            base_model.classifier = torch.nn.Linear(base_model.config.hidden_size, len(label_list)) # fixes bert-base-NER-uncased errors
            if hasattr(base_model, "classifier"):
                base_model.classifier = base_model.classifier.to(dtype=compute_dtype)
            model = PeftModel.from_pretrained(base_model, model_addr)
            in_features = model.base_model.model.classifier.in_features
            model.base_model.model.classifier = torch.nn.Linear(in_features, len(label_list)).to(device=dev, dtype=compute_dtype)
            # inject weights from classifier head
            st_path = os.path.join(model_addr, "adapter_model.safetensors")
            if os.path.exists(st_path):
                tensors = load_file(st_path)
                new_state = {}
                for k, v in tensors.items():
                    if "classifier" in k:
                        clean_key = k.split(".")[-1] 
                        #new_state[clean_key] = v
                        new_state[clean_key] = v.to(compute_dtype)
                model.base_model.model.classifier.load_state_dict(new_state)
        else:
            model = AutoModelForTokenClassification.from_pretrained(model_addr).to(dev)
        model = model.eval()
        model.print_trainable_parameters()
        #model.base_model.model.classifier.to(dev)
        model.to(compute_dtype)
        metrics, preds, reals, input_ids = sghead_evaluate_model(model, test_loader, dev, id2label, return_rnp=True)
    elif htype == "mhead":
        test_dataset = MheadDataset(dataset_dict["test"], head_lst, tokenizer, label2id, max_length=max_length)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda b: mhead_collate(b, tokenizer.pad_token_id)
        )
        model = MheadTokenClassifier(model_name, head_lst, dropout=dropout, quant=quant).to(dev)
        model.load_state_dict(torch.load(model_addr+"/model.pt", weights_only=True), strict=False)
        model.eval()
        metrics, preds, reals, input_ids = mhead_evaluate_model(model, test_loader, dev, id2label, return_rnp=True)
    words = [tokenizer.convert_ids_to_tokens(sent) for sent in input_ids]
    return metrics, reals, preds, words

def prettify_tc_irp(htype, reals, preds, inputs):
    if htype == "sghead":
        irp = []
        for artind in range(len(reals)):
            arttrk = []
            for tokid in range(len(preds[artind])):
                arttrk.append((inputs[artind][tokid].replace("\u2581",""), reals[artind][tokid], preds[artind][tokid]))
            irp.append(arttrk)
    elif htype == "mhead":
        heads = list(reals)
        irp = {head:[] for head in heads}
        for head in heads:
            for artind in range(len(reals[head])):
                arttrk = []
                for tokid in range(len(preds[head][artind])):
                    arttrk.append((inputs[artind][tokid].replace("\u2581",""), reals[head][artind][tokid], preds[head][artind][tokid]))
                irp[head].append(arttrk)
    return irp

def calculate_all_results_tc(mode, htype, irp):
    all_results = {"token": {}, "entity": {}}
    labels = get_label_set(mode, "mhead")
    if htype == "sghead":
        s_reals = []
        s_preds = []
        s_inputs = []
        for art in irp:
            inputs = [ent[0] for ent in art]
            reals = [ent[1] for ent in art]
            preds = [ent[2] for ent in art]
            s_reals.append(reals)
            s_preds.append(preds)
            s_inputs.append(inputs)
    elif htype == "mhead":
        m_reals = {head:[] for head in labels}
        m_preds = {head:[] for head in labels}
        m_inputs = []
        for head in labels:
            for art in irp[head]:
                inputs = [ent[0] for ent in art]
                reals = [ent[1] for ent in art]
                preds = [ent[2] for ent in art]
                m_reals[head].append(reals)
                m_preds[head].append(preds)
                if head == labels[0]:
                    m_inputs.append(inputs)
    return all_results

def get_dspy_rpt(mode, appr, preds_f, reals_f):
    rpt = []
    times = []
    labels = get_label_set(mode, "mhead")
    if appr == "zero-shot":
        with open(preds_f, "r", encoding="utf-8") as f:
            preds_json = json.load(f)
        dataset_dict = DatasetDict.load_from_disk(reals_f)
        reals = convert_tokens_to_entities(dataset_dict["test"])
        for entry in reals:
            info = {"id": entry['id'], "text": entry['text']}
            info["real"] = {label: entry[label] for label in labels}
            for prede in preds_json:
                if prede["id"] == entry['id']:
                    try:
                        info["pred"] = ast.literal_eval(prede["output"])
                    except:
                        print(prede["output"])
                        info["pred"] = ast.literal_eval(prede["output"][1:-1])
                    times.append(prede["time_m"])
            rpt.append(info)
    return rpt

def get_irps(mode, htype, interest, model_name, cwd, r):
    model_run_dir = f"{cwd}/models/{mode}/{htype}/{interest}"
    print(f"\n --- INTEREST: {interest} ---")
    if "sent" in interest:
        print("\n --- SENTENCE LEVEL ---")
        dsdct_dir = f"{cwd}/inputs/{mode}/sent/{htype}_dsdcts"
    else:
        print("\n --- ARTICLE LEVEL ---")
        dsdct_dir = f"{cwd}/inputs/{mode}/{htype}_dsdcts"
    metrics, reals, preds, words = get_tc_rpi(mode, htype, model_name, model_run_dir, dsdct_dir, r)
    irp = prettify_tc_irp(htype, reals, preds, words)
    print(f"Successfully got predictions for {model_name} {r}")
    with open(f"{cwd}/results/{mode}/{htype}/{interest}/{model_name.split("/")[-1]}_irp_{r}.json", "w", encoding="utf-8") as f:
        json.dump(irp, f, indent=4)

def main():
    cwd = os.getcwd()
    mode = "a"
    appr = "zero-shot"
    r = 0
    preds_f = f"{cwd}/results/{mode}/dspy/zero-shot/art/Qwen3.5-9B_{r}.json"
    reals_f = f"{cwd}/inputs/{mode}/mhead_dsdcts/dsdct_r{r}"
    htype = "sghead"
    interest = "sent"#"og"#
    model_run_dir = f"{cwd}/models/{mode}/{htype}/{interest}"
    r_n = 5
    ###########################
    # zero shot getting results
    #rpt = get_dspy_rpt(mode, appr, preds_f, reals_f)
    #with open(f"{cwd}/results/{mode}/dspy/zero-shot/art/randp_{r}.json", "w", encoding="utf-8") as f:
    #    json.dump(rpt, f, indent=4)
    ############################
    # consolidating the various runs into only the best version for each r
    #consol_tc_model_runs(model_run_dir, r_n)
    ############################
    # tc getting results
    model_name = "microsoft/deberta-v3-base"
    #for r in range(r_n):
    #    get_irps(mode, htype, interest, model_name, cwd, r)
    ############################
    # getting r
    with open(f"{cwd}/results/{mode}/{htype}/{interest}/{model_name.split("/")[-1]}_irp_{r}.json", "r", encoding="utf-8") as f:
        irp = json.load(f)
    results_dct = calculate_all_results_tc(mode, htype, irp)
    with open(f"{cwd}/results/{mode}/{htype}/{interest}/{model_name.split("/")[-1]}_results_{r}.json", "w", encoding="utf-8") as f:
        json.dump(results_dct, f, indent=4)

if __name__ == "__main__":
    main()

