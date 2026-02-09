'''
For use in subprocess to train the mhead models in loops without memory leakage between runs
'''
import os, json, sys, time, gc
import torch
from mhead_ner_ft import finetune_mhead_model
from create_datasets import get_label_set

params = {
    "num_epochs": 25,
    "lr": 3e-5,#5e-5,#2e-5,#3e-5,
    "weight_decay": 0.01,
    "batch_size":16,
    "num_warmup_steps":0,
    "patience": 5,
    "dropout": 0.1,
    "max_length": 512
}

if __name__ == '__main__':
    mode = sys.argv[1]
    model_name = sys.argv[2]
    r = int(sys.argv[3])
    model_save_addr = sys.argv[4]
    dsdct_dir = sys.argv[5]
    label_list = get_label_set(mode, "mhead")
    finetune_mhead_model(model_name, label_list, model_save_addr, dsdct_dir, r, params)
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(3)