'''
Normal token classification model training 
'''
import os
from transformers import AutoTokenizer, AutoModelForTokenClassification, get_linear_schedule_with_warmup, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from datasets import DatasetDict
import evaluate
import numpy as np
import json
import subprocess
import time
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import gc
import tqdm
from create_datasets import get_label_set
from auxil import bio_fixing, convert_numpy_torch_to_python

#############
# functions #
#############
class SgheadDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, label2id, max_length=512, chunk_overlap=128):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        self.chunk_overlap = chunk_overlap
        # let's prep these chunks
        # __getitem__ is gonna be sooo quick
        # also gotta bring tokenize_and_align_labels in here
        self.items = []
        for entry in hf_dataset:
            tokens = entry["tokens"]
            ner_tags = entry["ner_tags"]
            tokenized_inputs = self.tokenizer(tokens, truncation=True, is_split_into_words=True, max_length=self.max_length, return_tensors=None,
                                              stride=self.chunk_overlap, return_overflowing_tokens=True, return_special_tokens_mask=False, padding=False)
            for i in range(len(tokenized_inputs["input_ids"])):
                word_ids = tokenized_inputs.word_ids(batch_index=i)
                #enc = tokenized_inputs.encodings[i]
                #word_ids = enc.word_ids
                label_ids = []
                previous_word_idx = None
                for word_idx in word_ids:
                    if word_idx is None:
                        label_ids.append(-100)
                    elif word_idx != previous_word_idx:
                        label_ids.append(self.label2id[ner_tags[word_idx]])
                    else:
                        label_ids.append(-100)
                    previous_word_idx = word_idx
                self.items.append({
                    "input_ids": torch.tensor(tokenized_inputs["input_ids"][i]),
                    "attention_mask": torch.tensor(tokenized_inputs["attention_mask"][i]),
                    "labels": torch.tensor(label_ids),
                })
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        return self.items[idx]

def sghead_collate(batch, pad_token_id):
    '''
    Collate function for our pytorch DataLoader
    '''
    input_ids = [b["input_ids"] for b in batch]
    attention_masks = [b["attention_mask"] for b in batch]
    labels = [b["labels"] for b in batch]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    labels = pad_sequence(labels, batch_first=True, padding_value=-100)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": labels
    }

def sghead_evaluate_model(model, dataloader, dev, id2label, return_rnp = False):
    seqeval = evaluate.load("seqeval")
    model.eval()
    total_eval_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(dev) for k, v in batch.items()}
            outputs = model(**batch)
            total_eval_loss += outputs.loss.item()
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1).cpu().numpy()
            labels = batch["labels"].cpu().numpy()
            # for sentence token list labels in list of batch sentences
            for preds, labs in zip(predictions, labels):
                # holds the label(not id) label for each tok in sentence
                true_preds = []
                true_labs = []
                # for token label in sentence token list labels
                for p, l in zip(preds, labs):
                    if l != -100:
                        true_preds.append(id2label[p])
                        true_labs.append(id2label[l])
                # append sentence token list to list of sentences
                all_preds.append(true_preds)
                all_labels.append(true_labs)
    predictions_fixed = bio_fixing("sghead", all_preds)
    metrics = seqeval.compute(predictions=predictions_fixed, references=all_labels)
    if return_rnp:
        return metrics, predictions_fixed, all_labels
    metrics['avg_eval_loss'] = total_eval_loss / len(dataloader)
    return metrics

TARGET_MODULES_DICT={
    "microsoft/deberta-v3-base":["query_proj","key_proj","value_proj","dense"],
    "FacebookAI/xlm-roberta-base":["query","key","value","dense"],
    "dslim/bert-base-NER-uncased":["query","key","value","dense"],
    "answerdotai/ModernBERT-base":["attn.Wqkv","attn.Wo","mlp.Wi","mlp.Wo"]
}

def finetune_sghead_model(model_name, label_list, model_save_addr, dsdct_dir, r, params = None):
    '''
    Docstring for finetune_sghead_model
    
    :param model_name: model to finetune
    :param label_list: list of labels
    :param model_save_addr: where to save model
    :param dsdct_dir: location of datasetdicts
    :param r: which dataset split dict to use
    :param params: hyperparameter dict including num_epochs, lr, weight_decay, batch_size, num_warmup_steps, and patience
    '''
    st = time.time()
    if not params:
        params = {
            "num_epochs": 10,
            "lr": 3e-5,
            "weight_decay": 0.01,
            "batch_size":16,
            "num_warmup_steps":0,
            "patience": 3,
            "max_length": 512
        }
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for i, l in enumerate(label_list)}
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    #dataset_dict = DatasetDict.load_from_disk(f"{dsdct_dir}/dsdct_r{r}")
    # CHANGED FOR HYPERPARAMETER TUNING ONLY
    dataset_dict = DatasetDict.load_from_disk(f"{dsdct_dir}/dsdct_{r}")
    if model_name == "answerdotai/ModernBERT-base":
        train_dataset = SgheadDataset(dataset_dict["train"], tokenizer, label2id, max_length=params['max_length'])
        dev_dataset = SgheadDataset(dataset_dict["dev"], tokenizer, label2id, max_length=params['max_length'])
    else:
        train_dataset = SgheadDataset(dataset_dict["train"], tokenizer, label2id)
        dev_dataset = SgheadDataset(dataset_dict["dev"], tokenizer, label2id)
    train_loader = DataLoader(
        train_dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        collate_fn=lambda b: sghead_collate(b, tokenizer.pad_token_id)
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=params["batch_size"],
        shuffle=False,
        collate_fn=lambda b: sghead_collate(b, tokenizer.pad_token_id)
    )
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # incorporating QLoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=["classifier"]
    )
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
        quantization_config=bnb_config,
        ignore_mismatched_sizes=(model_name == "dslim/bert-base-NER-uncased"),
        device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)
    #if hasattr(model, "classifier"):
    #    model.classifier.to(torch.float32) #force cast to avoid bitsandbytes error??
    model.classifier.to(dtype=torch.float32, device=dev)
    lora_config = LoraConfig(
        r=9,
        lora_alpha=32,
        target_modules=TARGET_MODULES_DICT[model_name],
        lora_dropout=0.1,
        bias="none",
        task_type="TOKEN_CLS",
        modules_to_save=["classifier"]
    )
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    # end qlora
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    num_training_steps = params["num_epochs"] * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=params["num_warmup_steps"],
        num_training_steps=num_training_steps
    )
    # adding early stopping
    best_eval_loss = float("inf")
    best_epoch_metrics = {}
    epochs_no_improvement = 0
    model_epoch = 0
    for epoch in range(params["num_epochs"]):
        model.train()
        total_train_loss = 0.0
        for batch in tqdm.tqdm(train_loader):
            batch = {k: v.to(dev) for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch)
            train_loss = outputs.loss
            train_loss.backward()
            optimizer.step()
            scheduler.step()
            total_train_loss += train_loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        metrics = sghead_evaluate_model(model, dev_loader, dev, id2label)
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Eval Loss: {metrics['avg_eval_loss']:.4f} | Precision: {metrics['overall_precision']:.4f} | Recall: {metrics['overall_recall']:.4f} | F1: {metrics['overall_f1']:.4f}")
        if metrics['avg_eval_loss'] < best_eval_loss:
            best_eval_loss = metrics['avg_eval_loss']
            best_epoch_metrics = metrics
            model_epoch = epoch
            epochs_no_improvement = 0
            save_path = f"{model_save_addr}/{model_name.split('/')[-1]}_{r}"
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
        else:
            epochs_no_improvement += 1
            if epochs_no_improvement >= params['patience']:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break
    best_epoch_metrics['epoch_saved'] = model_epoch
    best_epoch_metrics['time_min'] = round((time.time()-st)/60,2)
    print(best_epoch_metrics)
    with open(f"{model_save_addr}/{model_name.split('/')[-1]}_{r}/params.json", "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=4)
    metrics_clean = convert_numpy_torch_to_python(best_epoch_metrics)
    with open(f"{model_save_addr}/{model_name.split('/')[-1]}_{r}/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_clean, f, ensure_ascii=False, indent=4)
    # cleanup
    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    return best_epoch_metrics

########
# main #
########

def main():
    cwd = os.getcwd()
    
    ########### one-off ###########
    for mode in ["a"]:#,"b","c", "d"]:
        model_save_addr = f"{cwd}/models/{mode}/sghead"
        dsdct_dir = f"{cwd}/inputs/{mode}/sghead_dsdcts"
        label_list = get_label_set(mode, "sghead")
        #
        model_name = "microsoft/deberta-v3-base"
        r = 0
        params = {
            "num_epochs": 15,
            "lr": 3e-5, #3e-5, also try 5e-5
            "weight_decay": 0.01,
            "batch_size":16,
            "num_warmup_steps":0,
            "patience": 5,
            "max_length": 512
            }
        finetune_sghead_model(model_name, label_list, model_save_addr, dsdct_dir, r, params)
    '''
    ########### subprocess ###########
    for model_name in ["microsoft/deberta-v3-base","FacebookAI/xlm-roberta-base","dslim/bert-base-NER-uncased"]:# ["answerdotai/ModernBERT-base"]:#
        for mode in ["a","b","c","d","e"]:
            for r in list(range(5)):
                model_save_addr = f"{cwd}/models/{mode}/sghead"
                dsdct_dir = f"{cwd}/inputs/{mode}/sghead_dsdcts"
                print(f"\n--- Starting '{mode}' run {model_name} r{r} ---")
                run_st = time.time()
                subprocess.run([
                    "python", "train_sghead.py",
                    mode,
                    model_name,
                    str(r),
                    model_save_addr,
                    dsdct_dir
                ],
                    check=True, capture_output=True, text=True)
                print(f"\n--- Finished '{mode}' run {model_name} r{r} ---")
                print(f'\nRun done in {round((time.time()-run_st)/60,2)} min')
                time.sleep(2)
    '''
if __name__=="__main__":
    main()