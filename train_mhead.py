'''
For use in subprocess to train the mhead models in loops without memory leakage between runs
'''
import os, json, sys, time, gc
import torch
from mhead_ner_ft import finetune_mhead_model
from create_datasets import get_label_set, CLASS_WEIGHTS
from auxil import HYPERPARAM_DCT

params = HYPERPARAM_DCT["mhead"]
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
    if extra['weight'] == True:
        extra['weight'] = CLASS_WEIGHTS[mode] # need to create in-function weight calculation 
        params[mode][model_name]['lr'] = params[mode][model_name]['lr']/2
    if extra['sent'] == True:
        params[mode][model_name]["batch_size"] = params[mode][model_name]["batch_size"]*2
    #finetune_mhead_model(model_name, label_list, model_save_addr, dsdct_dir, r, params[model_name])
    finetune_mhead_model(model_name, label_list, model_save_addr, dsdct_dir, r, params[mode][model_name], extra=extra)
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(3)