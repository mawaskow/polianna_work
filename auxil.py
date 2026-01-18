import numpy as np
import torch

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