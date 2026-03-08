'''
For use in subprocess to train the mhead models in loops without memory leakage between runs
'''
import os, json, sys, time, gc
import torch
from mhead_ner_ft import finetune_mhead_model
from create_datasets import get_label_set, CLASS_WEIGHTS

params = {
    "microsoft/deberta-v3-base":{
            "num_epochs": 30,
            "lr": 1e-4,
            "weight_decay": 0.025,
            "batch_size":8,
            "num_warmup_steps":0,
            "patience": 5,
            "dropout": 0.1,
            "max_length": 512
    },
    "FacebookAI/xlm-roberta-base":{
            "num_epochs": 30,
            "lr": 1e-4,
            "weight_decay": 0.025,
            "batch_size":8,
            "num_warmup_steps":0,
            "patience": 5,
            "dropout": 0.1,
            "max_length": 512
    },
    "dslim/bert-base-NER-uncased":{
            "num_epochs": 30,
            "lr": 1e-4,
            "weight_decay": 0.025,
            "batch_size":8,
            "num_warmup_steps":0,
            "patience": 5,
            "dropout": 0.1,
            "max_length": 512
    },
    "answerdotai/ModernBERT-base":{
            "num_epochs": 30,
            "lr": 1e-4,
            "weight_decay": 0.025,
            "batch_size":8,
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
            "sent": False
        }

if __name__ == '__main__':
    mode = sys.argv[1]
    model_name = sys.argv[2]
    r = int(sys.argv[3])
    model_save_addr = sys.argv[4]
    dsdct_dir = sys.argv[5]
    label_list = get_label_set(mode, "mhead")
    #finetune_mhead_model(model_name, label_list, model_save_addr, dsdct_dir, r, params[model_name])
    finetune_mhead_model(model_name, label_list, model_save_addr, dsdct_dir, r, params[model_name], extra=extra)
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(3)