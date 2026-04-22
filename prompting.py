import os
import json
import dspy
from dspy.teleprompt import MIPROv2
import time
from datasets import DatasetDict, concatenate_datasets
from create_datasets import get_label_set, convert_tokens_to_entities
from tqdm import tqdm
import numpy as np
import ast
from collections import Counter

class PolProc(dspy.Signature):
    """Extract entities in the provided policy text for each policy design element category: 'Actor', 'InstrumentType', 'Objective', 'Resource', and 'Time'. If there are no entities for a class, give an empty list. Include reasoning and step-by-step thinking before producing final output."""
    text: str = dspy.InputField(desc="Text from a policy document.")
    reasoning: str = dspy.OutputField(
        desc="Step-by-step reasoning and thinking."
    )
    output: str = dspy.OutputField(desc="Dictionary of the list of entities found for each class.")

class PolDesElExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extractor = dspy.Predict(PolProc)
    def forward(self, text):
        return self.extractor(text=text)

def entity_ds_to_dspy_ds(en_ds):
    examples = []
    for entry in en_ds:
        example = dspy.Example(
            text = entry['text'],
            output = {lbl: entry[lbl] for lbl in list(entry) if lbl not in ["id","text","tokens"]}
        ).with_inputs('text')
        examples.append(example)
    return examples

def zero_shot_call(model_name, r, dsdct_dir, results_addr, port=8000):
    dataset_dict = DatasetDict.load_from_disk(f"{dsdct_dir}/dsdct_r{r}")
    test_dataset = dataset_dict["test"]
    lm = dspy.LM(f"openai/{model_name}",
             api_base=f"http://localhost:{port}/v1",
             api_key="local", model_type="chat", max_tokens=10000)
    dspy.configure(lm=lm)
    test_proc = dspy.Predict(PolProc)
    results = []
    for entry in tqdm(test_dataset):
        est = time.time()
        try:
            pred = test_proc(text=entry['text'])
        except:
            pred = test_proc(text=" ".join(entry['tokens']))
        results.append({
            "id": entry['id'],
            "output": pred.output,
            "time_m": round((time.time()-est)/60,2),
            "reasoning": pred.reasoning
        })
    save_path = f"{results_addr}/{model_name.split('/')[-1]}_{r}"
    with open(f"{save_path}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

def dspy_icl_eval(example, pred, catch=False):
    f1s = []
    try:
        preds = ast.literal_eval(pred.output)
    except:
        preds = ast.literal_eval(pred.output[1:-1])
    for label in list(example.output):
        gold_set = example.output[label]
        pred_set = preds[label]
        res = { "tp": 0, "fp": 0, "fn": 0, "n_gold": 0, "n_pred": 0}
        tp = list((Counter(gold_set) & Counter(pred_set)).elements())
        res['tp'] = len(tp)
        fn = [ent for ent in gold_set if ent not in pred_set]
        res['fn'] = len(fn)
        fp = [ent for ent in pred_set if ent not in gold_set]
        res['fp'] = len(fp)
        res['n_gold'] = len(gold_set)
        res['n_pred'] = len(pred_set)
        pres = res['tp'] / res['n_pred'] if res['n_pred'] > 0 else 0
        reca = res['tp'] / res['n_gold'] if res['n_gold'] > 0 else 0
        if pres + reca > 0:
            f1 = 2 * pres * reca / (pres + reca)
        else:
            f1 = 0  
        scores =  {
            'precision': pres,
            'recall': reca,
            'f1': f1
        }
        f1s.append(scores['f1'])
    print(f1s)
    return np.average(f1s)
    #except Exception as exp:
    #    print("Exception:", exp)
    #    return 0.0

def in_context_fxn(model_name, r, dsdct_dir, model_addr, n_icl=5, port=8005):
    st = time.time()
    os.makedirs(model_addr, exist_ok=True)
    lm = dspy.LM(f"openai/{model_name}",
             api_base=f"http://localhost:{port}/v1",
             api_key="local", model_type="chat", max_tokens=10000)
    dspy.configure(lm=lm)
    dsdct = DatasetDict.load_from_disk(f"{dsdct_dir}/dsdct_r{r}")
    sample_ds = concatenate_datasets([dsdct['train'], dsdct['dev']])
    en_ds = convert_tokens_to_entities(sample_ds)
    dds = entity_ds_to_dspy_ds(en_ds)
    program = PolDesElExtractor()
    teleprompter = MIPROv2(
                metric=dspy_icl_eval,
                auto='light',#"medium",
                max_bootstrapped_demos=5, 
                max_labeled_demos=n_icl)
    optimized_program = teleprompter.compile(program, trainset=dds)
    model_save_path = os.path.join(model_addr, f"{model_name.split("/")[-1]}_{n_icl}_{r}.json")
    optimized_program.save(model_save_path)
    print(f"Finished in {round((time.time()-st)/60,2)}")

def use_opt_dspy(model_name, r, dsdct_dir, dspy_model_addr, port=8005):
    st = time.time()
    lm = dspy.LM(f"openai/{model_name}",
             api_base=f"http://localhost:{port}/v1",
             api_key="local", model_type="chat", max_tokens=10000)
    dspy.configure(lm=lm)
    dsdct = DatasetDict.load_from_disk(f"{dsdct_dir}/dsdct_r{r}")
    en_ds = convert_tokens_to_entities(dsdct['test'])
    program = PolDesElExtractor()
    program.load(path=dspy_model_addr)
    results = []
    for example in tqdm(en_ds):
        prediction = program(text=example['text'])
        results.append({
            "id": example['id'],
            "text": example['text'],
            'real': {lbl: example[lbl] for lbl in list(example) if lbl not in ["id","text","tokens"]},
            "prediction": prediction.output
        })
    print(f"Done in {round((time.time()-st)/60,2)} min")
    return results

def main():
    cwd = os.getcwd()
    interest = "icl"#"zero-shot"#"use-icl"#
    lvl = "art"
    mode = "a"
    model_name = "Qwen/Qwen3.5-9B"
    dsdct_dir = f"{cwd}/inputs/{mode}/mhead_dsdcts"
    # if lvl == "sent":
    # dsdct_dir = f"{cwd}/inputs/{mode}/sent/mhead_dsdcts"
    r=0
    results_addr = f"{cwd}/results/{mode}/dspy/{interest}/{lvl}"
    model_addr = f"{cwd}/models/{mode}/dspy/{interest}/{lvl}"
    
    for r in list(range(1, 5)):
        if lvl == "sent":
            dsdct_dir = f"{cwd}/inputs/{mode}/{lvl}/mhead_dsdcts"
        else:
            dsdct_dir = f"{cwd}/inputs/{mode}/mhead_dsdcts"
        print(f"\n--- Starting '{mode}' run {model_name} r{r} ---")
        st = time.time()
        if interest == "zero-shot":
            zero_shot_call(model_name, r, dsdct_dir, results_addr, port=8000)
        elif interest == "icl":
            in_context_fxn(model_name, r, dsdct_dir, model_addr, port=8000)
        print(f"\n--- Completed in {round((time.time()-st)/60,2)} ---")
    '''
    results = use_opt_dspy(model_name, r, dsdct_dir, f"{model_addr}/Qwen3.5-9B_5_0.json", port=8000)
    with open(f"{results_addr}/r{r}/dspy_0.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    '''

if __name__ == "__main__":
    main()