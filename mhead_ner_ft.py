'''
A custom token classification model with five separate classifier heads for five separate feature types.
'''
import os
import numpy as np
from collections import Counter
from transformers import AutoModel, AutoConfig, AutoTokenizer, AutoModelForTokenClassification, get_linear_schedule_with_warmup, PreTrainedModel
from transformers import AutoTokenizer, Trainer
from transformers.modeling_outputs import TokenClassifierOutput
from datasets import DatasetDict
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from typing import Any, Dict, List
from sklearn.metrics import f1_score
import subprocess
import gc
import time
import json
import evaluate
import tqdm
from create_datasets import get_label_set
from auxil import bio_fixing, convert_numpy_torch_to_python

#########################
# classes and functions #
#########################

class MheadDataset(Dataset):
    def __init__(self, hf_dataset, head_lst, tokenizer, label2id, max_length=512, chunk_overlap = 128):
        self.dataset = hf_dataset
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        self.chunk_overlap = chunk_overlap
        self.heads = head_lst
        # prechunk 
        self.items = []
        for entry in hf_dataset:
            tokens = entry["tokens"]
            ner_tag_dct = {head:entry["labels_"+head] for head in self.heads}
            tokenized_inputs = self.tokenizer(tokens, truncation=True, is_split_into_words=True, max_length=self.max_length, return_tensors=None,
                                    stride=self.chunk_overlap, return_overflowing_tokens=True, return_special_tokens_mask=False, padding=False)
            for i in range(len(tokenized_inputs["input_ids"])):
                word_ids = tokenized_inputs.word_ids(batch_index=i)
                # for each label type/list
                head_lbl_dct = {}
                for head in self.heads:
                    previous_word_idx = None
                    label_ids = []
                    for word_idx in word_ids:
                        if word_idx is None:
                            label_ids.append(-100)
                        elif word_idx != previous_word_idx:
                            label_ids.append(self.label2id[ner_tag_dct[head][word_idx]])
                        else:
                            label_ids.append(-100)
                        previous_word_idx = word_idx
                    head_lbl_dct[head] = label_ids
                self.items.append({
                    "input_ids": torch.tensor(tokenized_inputs["input_ids"][i]),
                    "attention_mask": torch.tensor(tokenized_inputs["attention_mask"][i]),
                    "labels": {head:torch.tensor(head_lbl_dct[head]) for head in self.heads}
                })
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        return self.items[idx]

class MheadTokenClassifier(nn.Module):
    def __init__(self, base_model_name, head_lst, num_labels=3, class_wgt_dct = None, head_wgt_dct = None, dropout = 0.1):
        super().__init__()
        self.model_name = base_model_name
        self.encoder = AutoModel.from_pretrained(base_model_name) #ignore_mismatched_sizes= model_name == "dslim/bert-base-NER-uncased"
        hidden_size = self.encoder.config.hidden_size
        self.heads = head_lst
        self.num_labels = num_labels
        #sep linear head for each feature type classification
        self.classifiers = nn.ModuleDict({
            head: nn.Linear(hidden_size, num_labels) for head in self.heads
        })
        self.dropout = nn.Dropout(dropout)
        self.class_weights = class_wgt_dct if class_wgt_dct else {}
        self.head_weights = head_wgt_dct if head_wgt_dct else {}
    def forward(self, input_ids, attention_mask=None, labels=None):
        # batch of inputs encoded by base model
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        # only uses last hidden state... for now
        # will look into averaging/concatenating last few hidden states
        sequence_output = outputs.last_hidden_state
        # passes encoded input sequence to each classifier to get logits
        logits = {head: self.classifiers[head](sequence_output) for head in self.classifiers}
        loss = None
        if labels:
            loss = 0.0
            # flatten attn mask
            active_mask = attention_mask.view(-1) == 1
            for head in self.heads:
                # get active logits, flatten to (num_act_tokens, num_classes) 
                # then apply active loss mask (to both logits and labels)
                head_logits = logits[head].view(-1, self.num_labels)[active_mask]
                head_labels = labels[head].view(-1)[active_mask]
                # weighting BIO classes for this feature
                weight = self.class_weights.get(head, None)
                if weight is not None:
                    weight = weight.to(head_logits.device)
                # computing loss for this head
                loss_fn = nn.CrossEntropyLoss(weight=weight)
                head_loss = loss_fn(head_logits, head_labels)
                # weight the loss for this head
                head_loss *= self.head_weights.get(head, 1.0)
                # sum loss across heads for single update to train simultaneously
                loss += head_loss
        return loss, logits
    def save_model(self, model, save_dir, params):
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
        config = {
            "base_model_name": self.model_name,
            "heads": self.heads,
            "num_labels": self.num_labels,
            "params": params,
            "dropout": params['dropout'],
            "class_weights": self.class_weights,#{k: v.tolist() for k, v in (class_weights or {}).items()},
            "head_weights": self.head_weights
        }
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=4)

def mhead_collate(batch, pad_token_id):
    '''
    Collate function for our pytorch DataLoader
    '''
    input_ids = [b["input_ids"] for b in batch]
    attention_masks = [b["attention_mask"] for b in batch]
    return {
        "input_ids": pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id),
        "attention_mask": pad_sequence(attention_masks, batch_first=True, padding_value=0),
        "labels": {head: pad_sequence([b['labels'][head] for b in batch], batch_first=True, padding_value=-100) for head in list(batch[0]['labels'])}
    }

def mhead_evaluate_model(model, dataloader, dev, id2label, return_rnp = False):
    seqeval = evaluate.load("seqeval")
    model.eval()
    meta_metrics = {}
    total_eval_loss = 0.0
    if return_rnp:
        heads = model.heads
        pred_coll = {name: [] for name in heads}
        real_coll = {name: [] for name in heads}
    with torch.no_grad():
        for batch in dataloader:
            batch = {
                "input_ids": batch["input_ids"].to(dev),
                "attention_mask": batch["attention_mask"].to(dev),
                "labels": {k: v.to(dev) for k, v in batch['labels'].items()}
            }
            loss, logits = model(**batch)
            total_eval_loss += loss.item()
            predictions= {head: torch.argmax(logits[head], dim=-1).cpu().numpy() for head in logits}
            labels = {head: batch["labels"][head].cpu().numpy() for head in batch["labels"]}
            for head in predictions:
                # for this batch, for this head in the batch
                # records the list of prediction labels in sentence list form
                head_preds =[]
                head_labels = []
                # predictions is list of lists of sentence labels from batch
                # preds is the list of labels for a single sentence
                for preds, labs in zip(predictions[head], labels[head]):
                    # art_preds holds the label(not id) versions of the predictions
                    # for each sentence
                    art_preds = []
                    art_labs = []
                    # for p,s [each label]
                    for p, l in zip(preds, labs):
                        if l != -100:
                            art_preds.append(id2label[p])
                            art_labs.append(id2label[l])
                    # now appends the sentence/list of labels to head_preds list
                    head_preds.append(art_preds)
                    head_labels.append(art_labs)
                fixed_predictions = bio_fixing("mhead", head_preds)
                # for pred_coll's head list, extend with the list of lists
                # not append bc that makes lists of lists of lists
                pred_coll[head].extend(fixed_predictions)
                real_coll[head].extend(head_labels)
    for head in list(pred_coll):
        meta_metrics[head] = seqeval.compute(predictions=pred_coll[head], references=real_coll[head])
    if return_rnp:
        return meta_metrics, pred_coll, real_coll
    meta_metrics['avg_eval_loss'] = total_eval_loss / len(dataloader)
    return meta_metrics

def finetune_mhead_model(model_name, head_lst, model_save_addr, dsdct_dir, r, params):
    '''
    Docstring for finetune_mhead_model
    
    :param model_name: name of hf model to be used as tokenizer and base model for fine-tuning
    :param model_save_addr: directory address where to save the model directories
    :param dsdct_dir: directory address (mhead_dsdcts) where the datasetdictionary directories (dsdct_r{#}) are stored
    :param r: which # run
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
            "dropout": 0.1
        }
    label_list = ['O', 'B', 'I']
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for i, l in enumerate(label_list)}
    save_path = f"{model_save_addr}/{model_name.split('/')[-1]}_{r}"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset_dict = DatasetDict.load_from_disk(f"{dsdct_dir}/dsdct_r{r}")
    if model_name == "answerdotai/ModernBERT-base":
        train_dataset = MheadDataset(dataset_dict["train"], head_lst, tokenizer, label2id, max_length=2048)
        dev_dataset = MheadDataset(dataset_dict["dev"], head_lst, tokenizer, label2id, max_length=2048)
    else:
        train_dataset = MheadDataset(dataset_dict["train"], head_lst, tokenizer, label2id)
        dev_dataset = MheadDataset(dataset_dict["dev"], head_lst, tokenizer, label2id)
    train_loader = DataLoader(
        train_dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        collate_fn=lambda b: mhead_collate(b, tokenizer.pad_token_id)
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=params["batch_size"],
        shuffle=False,
        collate_fn=lambda b: mhead_collate(b, tokenizer.pad_token_id)
    )
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MheadTokenClassifier(model_name, head_lst, dropout = params['dropout']).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    num_training_steps = params["num_epochs"] * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=params["num_warmup_steps"],
        num_training_steps=num_training_steps
    )
    # adding early stopping
    best_eval_loss = float("inf")
    epochs_no_improvement = 0
    model_epoch = 0
    for epoch in range(params["num_epochs"]):
        model.train()
        total_train_loss = 0.0
        for batch in tqdm.tqdm(train_loader):
            #batch = {k: v.to(dev) for k, v in batch.items()}
            batch = {
                "input_ids": batch["input_ids"].to(dev),
                "attention_mask": batch["attention_mask"].to(dev),
                "labels": {k: v.to(dev) for k, v in batch['labels'].items()}
            }
            optimizer.zero_grad(set_to_none=True)
            train_loss, _ = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            train_loss.backward()
            optimizer.step()
            scheduler.step()
            total_train_loss += train_loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        metrics = mhead_evaluate_model(model, dev_loader, dev, id2label)
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Eval Loss: {metrics['avg_eval_loss']:.4f}")
        if metrics['avg_eval_loss'] < best_eval_loss:
            best_eval_loss = metrics['avg_eval_loss']
            model_epoch = epoch
            epochs_no_improvement = 0
            model.save_model(model, save_path, params)
        else:
            epochs_no_improvement += 1
            if epochs_no_improvement >= params['patience']:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break
    metrics['epoch_saved'] = model_epoch
    metrics['time_min'] = round((time.time()-st)/60,2)
    print(metrics)
    metrics_clean = convert_numpy_torch_to_python(metrics)
    with open(f"{save_path}/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_clean, f, ensure_ascii=False, indent=4)
    # cleanup
    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()
    
########
# main #
########

def main():
    cwd = os.getcwd()
    
    ########### one-off ###########
    for mode in ["a"]:#,"b"]:#,"c", "d"]:
        model_save_addr = f"{cwd}/models/{mode}/mhead"
        dsdct_dir = f"{cwd}/inputs/{mode}/mhead_dsdcts"
        label_list = get_label_set(mode, "mhead")
        params = {
                "num_epochs": 25,
                "lr": 3e-5,
                "weight_decay": 0.01,
                "batch_size":16,
                "num_warmup_steps":0,
                "patience": 5,
                "dropout": 0.1
            }
        model_name = "answerdotai/ModernBERT-base"
        r = 0
        finetune_mhead_model(model_name, label_list, model_save_addr, dsdct_dir, r, params)
    '''
    ########### subprocess ###########
    for model_name in ["FacebookAI/xlm-roberta-base"]:#["microsoft/deberta-v3-base","FacebookAI/xlm-roberta-base","dslim/bert-base-NER-uncased", "answerdotai/ModernBERT-base"]:
        for mode in ["a","b"]:#,"c","d"]:
            for r in list(range(5)):
                model_save_addr = f"{cwd}/models/{mode}/mhead"
                dsdct_dir = f"{cwd}/inputs/{mode}/mhead_dsdcts"
                print(f"\n--- Starting '{mode}' run {model_name} r{r} ---")
                run_st = time.time()
                subprocess.run([
                    "python", "train_mhead.py",
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
