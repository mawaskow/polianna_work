import numpy as np
import torch
from torch.utils.data import Sampler

class TokenBasedSampler(Sampler):
    def __init__(self, lengths, max_tokens, shuffle=True):
        self.lengths = lengths
        self.max_tokens = max_tokens
        self.shuffle = shuffle
    def __iter__(self):
        # sorting by length
        indices = np.argsort(self.lengths)
        batches = []
        current_batch = []
        for idx in indices:
            seq_len = self.lengths[idx]
            # Batch token count = (max length in batch) * (number of samples)
            # This accounts for the padding that will be added
            potential_max_len = max(current_batch_max_len if current_batch else 0, seq_len)
            potential_total_tokens = potential_max_len * (len(current_batch) + 1)
            if potential_total_tokens > self.max_tokens and current_batch:
                batches.append(current_batch)
                current_batch = []
            current_batch.append(idx)
            current_batch_max_len = max([self.lengths[i] for i in current_batch])
        if current_batch:
            batches.append(current_batch)  
        if self.shuffle:
            np.random.shuffle(batches)
        for batch in batches:
            yield batch
    def __len__(self):
        # Rough estimate; can vary slightly if using dynamic logic
        n_batches = len(self.lengths) // (self.max_tokens // np.mean(self.lengths))
        return int(n_batches)

def convert_numpy_torch_to_python(obj):
    '''
    For converting our metrics dict to something json serializable
    '''
    if isinstance(obj, dict):
        return {k: convert_numpy_torch_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_torch_to_python(v) for v in obj]
    elif isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    else:
        return obj
    
def bio_fixing(htype, preds):
    preds_fixed = []
    if htype == "mhead":
        for a, art in enumerate(preds):
            art_fixed = []
            for i, tok in enumerate(art):
                if i == 0:
                    if tok == "I":
                        art_fixed.append("B")
                    else:
                        art_fixed.append(tok)
                elif tok=="I" and art_fixed[i-1]=="O":
                    art_fixed.append("B")
                else:
                    art_fixed.append(tok)
            assert len(preds[a])==len(art_fixed)
            preds_fixed.append(art_fixed)
        assert len(preds)==len(preds_fixed)
    elif htype == "sghead":
        for a, art in enumerate(preds):
            art_fixed = []
            for i, tok in enumerate(art):
                if i == 0:
                    if tok[0] == "I":
                        art_fixed.append(f"B{tok[1:]}")
                    else:
                        art_fixed.append(tok)
                elif tok[0]=="I" and art_fixed[i-1]=="O":
                    art_fixed.append(f"B{tok[1:]}")
                elif tok[0]=="I" and art_fixed[i-1][1:]!=tok[1:]:
                    art_fixed.append(f"B{tok[1:]}")
                else:
                    art_fixed.append(tok)
            assert len(preds[a])==len(art_fixed)
            preds_fixed.append(art_fixed)
        assert len(preds)==len(preds_fixed)
    return preds_fixed

HYPERPARAM_DCT = {
    'sghead': 
    {'a': 
    {'microsoft/deberta-v3-base': {'num_epochs': 30,
        'batch_size': 16,
        'lr': 0.0008,
        'max_length': 512,
        'weight_decay': 0.005,
        'num_warmup_steps': 0,
        'patience': 5,
        'dropout': 0.1},
    'FacebookAI/xlm-roberta-base': {'num_epochs': 30,
        'batch_size': 8,
        'lr': 0.00052,
        'max_length': 256,
        'weight_decay': 0.02,
        'num_warmup_steps': 0,
        'patience': 5,
        'dropout': 0.1},
    'dslim/bert-base-NER-uncased': {'num_epochs': 30,
        'batch_size': 8,
        'lr': 0.00041,
        'max_length': 256,
        'weight_decay': 0.095,
        'num_warmup_steps': 0,
        'patience': 5,
        'dropout': 0.1},
    'answerdotai/ModernBERT-base': {'num_epochs': 30,
        'batch_size': 8,
        'lr': 0.00064,
        'max_length': 256,
        'weight_decay': 0.05,
        'num_warmup_steps': 0,
        'patience': 5,
        'dropout': 0.1}},
    'b':
    {'microsoft/deberta-v3-base': {'num_epochs': 30,
        'batch_size': 8,
        'lr': 0.00071,
        'max_length': 512,
        'weight_decay': 0.04,
        'num_warmup_steps': 0,
        'patience': 5,
        'dropout': 0.1},
    'FacebookAI/xlm-roberta-base': {'num_epochs': 30,
        'batch_size': 16,
        'lr': 0.00083,
        'max_length': 512,
        'weight_decay': 0.075,
        'num_warmup_steps': 0,
        'patience': 5,
        'dropout': 0.1},
    'dslim/bert-base-NER-uncased': {'num_epochs': 30,
        'batch_size': 8,
        'lr': 0.00085,
        'max_length': 256,
        'weight_decay': 0.07,
        'num_warmup_steps': 0,
        'patience': 5,
        'dropout': 0.1},
    'answerdotai/ModernBERT-base': {'num_epochs': 30,
        'batch_size': 8,
        'lr': 0.00057,
        'max_length': 256,
        'weight_decay': 0.09,
        'num_warmup_steps': 0,
        'patience': 5,
        'dropout': 0.1}},
    'c': 
    {'microsoft/deberta-v3-base': {'num_epochs': 30,
        'batch_size': 8,
        'lr': 0.0007,
        'max_length': 256,
        'weight_decay': 0.03,
        'num_warmup_steps': 0,
        'patience': 5,
        'dropout': 0.1},
    'FacebookAI/xlm-roberta-base': {'num_epochs': 30,
        'batch_size': 8,
        'lr': 0.00049,
        'max_length': 256,
        'weight_decay': 0.055,
        'num_warmup_steps': 0,
        'patience': 5,
        'dropout': 0.1},
    'dslim/bert-base-NER-uncased': {'num_epochs': 30,
        'batch_size': 32,
        'lr': 0.00074,
        'max_length': 256,
        'weight_decay': 0.08,
        'num_warmup_steps': 0,
        'patience': 5,
        'dropout': 0.1},
    'answerdotai/ModernBERT-base': {'num_epochs': 30,
        'batch_size': 16,
        'lr': 0.0009,
        'max_length': 256,
        'weight_decay': 0.04,
        'num_warmup_steps': 0,
        'patience': 5,
        'dropout': 0.1}},
    'd': 
    {'microsoft/deberta-v3-base': {'num_epochs': 30,
        'batch_size': 16,
        'lr': 0.00065,
        'max_length': 512,
        'weight_decay': 0.095,
        'num_warmup_steps': 0,
        'patience': 5,
        'dropout': 0.1},
    'FacebookAI/xlm-roberta-base': {'num_epochs': 30,
        'batch_size': 32,
        'lr': 0.00046,
        'max_length': 256,
        'weight_decay': 0.07,
        'num_warmup_steps': 0,
        'patience': 5,
        'dropout': 0.1},
    'dslim/bert-base-NER-uncased': {'num_epochs': 30,
        'batch_size': 8,
        'lr': 0.00047,
        'max_length': 512,
        'weight_decay': 0.005,
        'num_warmup_steps': 0,
        'patience': 5,
        'dropout': 0.1},
    'answerdotai/ModernBERT-base': {'num_epochs': 30,
        'batch_size': 16,
        'lr': 0.00081,
        'max_length': 256,
        'weight_decay': 0.01,
        'num_warmup_steps': 0,
        'patience': 5,
        'dropout': 0.1}},
    'e': 
    {'microsoft/deberta-v3-base': {'num_epochs': 30,
        'batch_size': 8,
        'lr': 0.00059,
        'max_length': 512,
        'weight_decay': 0.1,
        'num_warmup_steps': 0,
        'patience': 5,
        'dropout': 0.1},
    'FacebookAI/xlm-roberta-base': {'num_epochs': 30,
        'batch_size': 16,
        'lr': 0.00082,
        'max_length': 256,
        'weight_decay': 0.075,
        'num_warmup_steps': 0,
        'patience': 5,
        'dropout': 0.1},
    'dslim/bert-base-NER-uncased': {'num_epochs': 30,
        'batch_size': 8,
        'lr': 0.00069,
        'max_length': 512,
        'weight_decay': 0.0,
        'num_warmup_steps': 0,
        'patience': 5,
        'dropout': 0.1},
    'answerdotai/ModernBERT-base': {'num_epochs': 30,
        'batch_size': 8,
        'lr': 0.00034,
        'max_length': 512,
        'weight_decay': 0.02,
        'num_warmup_steps': 0,
        'patience': 5,
        'dropout': 0.1}}},
 'mhead': 
    {'a': 
    {'microsoft/deberta-v3-base': {'num_epochs': 30,
        'batch_size': 16,
        'lr': 0.00062,
        'max_length': 512,
        'weight_decay': 0.08,
        'num_warmup_steps': 0,
        'patience': 5,
        'dropout': 0.1},
   'FacebookAI/xlm-roberta-base': {'num_epochs': 30,
        'batch_size': 8,
        'lr': 0.00074,
        'max_length': 512,
        'weight_decay': 0.05,
        'num_warmup_steps': 0,
        'patience': 5,
        'dropout': 0.1},
   'dslim/bert-base-NER-uncased': {'num_epochs': 30,
        'batch_size': 8,
        'lr': 0.00051,
        'max_length': 256,
        'weight_decay': 0.0,
        'num_warmup_steps': 0,
        'patience': 5,
        'dropout': 0.1},
   'answerdotai/ModernBERT-base': {'num_epochs': 30,
        'batch_size': 16,
        'lr': 0.00075,
        'max_length': 768,
        'weight_decay': 0.085,
        'num_warmup_steps': 0,
        'patience': 5,
        'dropout': 0.1}},
  'b': 
  {'microsoft/deberta-v3-base': {'num_epochs': 30,
        'batch_size': 8,
        'lr': 0.00063,
        'max_length': 256,
        'weight_decay': 0.065,
        'num_warmup_steps': 0,
        'patience': 5,
        'dropout': 0.1},
   'FacebookAI/xlm-roberta-base': {'num_epochs': 30,
        'batch_size': 8,
        'lr': 0.00082,
        'max_length': 256,
        'weight_decay': 0.015,
        'num_warmup_steps': 0,
        'patience': 5,
        'dropout': 0.1},
   'dslim/bert-base-NER-uncased': {'num_epochs': 30,
        'batch_size': 8,
        'lr': 0.00042,
        'max_length': 256,
        'weight_decay': 0.1,
        'num_warmup_steps': 0,
        'patience': 5,
        'dropout': 0.1},
   'answerdotai/ModernBERT-base': {'num_epochs': 30,
        'batch_size': 8,
        'lr': 0.00077,
        'max_length': 768,
        'weight_decay': 0.1,
        'num_warmup_steps': 0,
        'patience': 5,
        'dropout': 0.1}},
  'c': 
  {'microsoft/deberta-v3-base': {'num_epochs': 30,
        'batch_size': 8,
        'lr': 0.00088,
        'max_length': 512,
        'weight_decay': 0.055,
        'num_warmup_steps': 0,
        'patience': 5,
        'dropout': 0.1},
   'FacebookAI/xlm-roberta-base': {'num_epochs': 30,
        'batch_size': 8,
        'lr': 0.00031,
        'max_length': 256,
        'weight_decay': 0.05,
        'num_warmup_steps': 0,
        'patience': 5,
        'dropout': 0.1},
   'dslim/bert-base-NER-uncased': {'num_epochs': 30,
        'batch_size': 8,
        'lr': 0.00047,
        'max_length': 512,
        'weight_decay': 0.095,
        'num_warmup_steps': 0,
        'patience': 5,
        'dropout': 0.1},
   'answerdotai/ModernBERT-base': {'num_epochs': 30,
        'batch_size': 8,
        'lr': 0.00082,
        'max_length': 1024,
        'weight_decay': 0.02,
        'num_warmup_steps': 0,
        'patience': 5,
        'dropout': 0.1}},
  'd': 
  {'microsoft/deberta-v3-base': {'num_epochs': 30,
        'batch_size': 8,
        'lr': 0.00055,
        'max_length': 512,
        'weight_decay': 0.025,
        'num_warmup_steps': 0,
        'patience': 5,
        'dropout': 0.1},
   'FacebookAI/xlm-roberta-base': {'num_epochs': 30,
        'batch_size': 32,
        'lr': 0.00066,
        'max_length': 256,
        'weight_decay': 0.015,
        'num_warmup_steps': 0,
        'patience': 5,
        'dropout': 0.1},
   'dslim/bert-base-NER-uncased': {'num_epochs': 30,
        'batch_size': 8,
        'lr': 0.00044,
        'max_length': 256,
        'weight_decay': 0.07,
        'num_warmup_steps': 0,
        'patience': 5,
        'dropout': 0.1},
   'answerdotai/ModernBERT-base': {'num_epochs': 30,
        'batch_size': 8,
        'lr': 0.0006,
        'max_length': 768,
        'weight_decay': 0.08,
        'num_warmup_steps': 0,
        'patience': 5,
        'dropout': 0.1}},
  'e': {'microsoft/deberta-v3-base': {'num_epochs': 30,
        'batch_size': 8,
        'lr': 0.00079,
        'max_length': 512,
        'weight_decay': 0.03,
        'num_warmup_steps': 0,
        'patience': 5,
        'dropout': 0.1},
   'FacebookAI/xlm-roberta-base': {'num_epochs': 30,
        'batch_size': 8,
        'lr': 0.00055,
        'max_length': 256,
        'weight_decay': 0.01,
        'num_warmup_steps': 0,
        'patience': 5,
        'dropout': 0.1},
   'dslim/bert-base-NER-uncased': {'num_epochs': 30,
        'batch_size': 8,
        'lr': 0.00037,
        'max_length': 512,
        'weight_decay': 0.0,
        'num_warmup_steps': 0,
        'patience': 5,
        'dropout': 0.1},
   'answerdotai/ModernBERT-base': {'num_epochs': 30,
        'batch_size': 8,
        'lr': 0.00087,
        'max_length': 768,
        'weight_decay': 0.02,
        'num_warmup_steps': 0,
        'patience': 5,
        'dropout': 0.1}}}}