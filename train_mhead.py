'''
For use in subprocess to train the mhead models in loops without memory leakage between runs
'''
import os, json, sys, time, gc
import torch
from mhead_ner_ft import finetune_mhead_model
from create_datasets import get_label_set
from auxil import HYPERPARAM_DCT

params = HYPERPARAM_DCT["mhead"]
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
    label_list = get_label_set(mode, "mhead")
    if extra['over']:
        params[mode][model_name]["dropout"] = params[mode][model_name]["dropout"]*2
        params[mode][model_name]["weight_decay"] = params[mode][model_name]["weight_decay"]*10
        #params[mode][model_name]["lr"] = params[mode][model_name]["lr"]/2
    if extra['sent']:
        params[mode][model_name]["max_length"] = int(params[mode][model_name]["max_length"]/4)
    params[mode][model_name]["lr"]= params[mode][model_name]["lr"]/2
    for loop in [2,1,0]:
        finetune_mhead_model(model_name, label_list, model_save_addr, dsdct_dir, r, params[mode][model_name], extra=extra, loop=loop)
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(3)