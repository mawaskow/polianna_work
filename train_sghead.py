'''
For use in subprocess to train the sghead models in loops without memory leakage between runs
'''
import os, json, sys, time, gc
import torch
from sghead_ner_ft import finetune_sghead_model
from create_datasets import get_label_set

params = {
    "microsoft/deberta-v3-base":{
            "num_epochs": 30,
            "lr": 7E-4,
            "weight_decay": 0.01,
            "batch_size":16,
            "num_warmup_steps":0,
            "patience": 5,
            "dropout": 0.1,
            "max_length": 512
    },
    "FacebookAI/xlm-roberta-base":{
            "num_epochs": 30,
            "lr": 7E-4,
            "weight_decay": 0.01,
            "batch_size":16,
            "num_warmup_steps":0,
            "patience": 5,
            "dropout": 0.1,
            "max_length": 512
    },
    "dslim/bert-base-NER-uncased":{
            "num_epochs": 30,
            "lr": 7E-4,
            "weight_decay": 0.01,
            "batch_size":16,
            "num_warmup_steps":0,
            "patience": 5,
            "dropout": 0.1,
            "max_length": 512
    },
    "answerdotai/ModernBERT-base":{
            "num_epochs": 30,
            "lr": 7E-4,
            "weight_decay": 0.01,
            "batch_size":16,
            "num_warmup_steps":0,
            "patience": 5,
            "dropout": 0.1,
            "max_length": 512
    }
}
extra = {
            "quant": True,
            "weight": False,
            "over": False,
            "sent": True
        }

if __name__ == '__main__':
    mode = sys.argv[1]
    model_name = sys.argv[2]
    r = int(sys.argv[3])
    model_save_addr = sys.argv[4]
    dsdct_dir = sys.argv[5]
    label_list = get_label_set(mode, "sghead")
    if extra['weight'] == True:
        params[model_name]["lr"] = params[model_name]["lr"]/2
    if extra['sent'] == True:
        params[model_name]["batch_size"] = params[model_name]["batch_size"]*2
    finetune_sghead_model(model_name, label_list, model_save_addr, dsdct_dir, r, params[model_name], extra)
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(3)