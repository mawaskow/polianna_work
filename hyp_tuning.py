import optuna
from train_sghead import finetune_sghead_model
from train_mhead import finetune_mhead_model
from create_datasets import get_label_set
import os

def sg_objective(trial, model_name, letter):
    params = {
        "num_epochs": 30,
        "lr": trial.suggest_float("lr", 1e-6, 9e-4, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
        "max_length": trial.suggest_categorical("max_length", [256, 512]),#, 1024, ]),
        "num_warmup_steps": 0,
        "patience": 5
    }
    try:
        metrics = finetune_sghead_model(
            model_name=model_name,
            label_list=get_label_set(letter, "sghead"),
            model_save_addr=f"./models/{letter}/sghead/optuna/{model_name.split('/')[-1]}",
            dsdct_dir=f"./inputs/{letter}/sghead_dsdcts",
            r="hyptune",
            params=params
        )
        return metrics["avg_eval_loss"], metrics["overall_f1"]
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        raise e

def mh_objective(trial, model_name, letter):
    params = {
        "num_epochs": 30,
        "lr": trial.suggest_float("lr", 5e-5, 9e-4, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
        "max_length": trial.suggest_categorical("max_length", [256, 512, 768, 1024]),
        "num_warmup_steps": 0,
        "patience": 5,
        "dropout": 0.1,
    }
    try:
        metrics = finetune_mhead_model(
            model_name=model_name,
            head_lst=get_label_set(letter, "mhead"),
            model_save_addr=f"./models/{letter}/mhead/optuna/{model_name.split('/')[-1]}",
            dsdct_dir=f"./inputs/{letter}/mhead_dsdcts",
            r="hyptune",
            params=params
        )
        return metrics["avg_eval_loss"], metrics["overall_f1"]
    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        raise e

def main():
    cwd = os.getcwd()
    for l in ["b","c","d","e"]:#"a",
        for mn in ["answerdotai/ModernBERT-base"]:#["microsoft/deberta-v3-base","FacebookAI/xlm-roberta-base","dslim/bert-base-NER-uncased"]:# :
            print(f"\n\n\n----- Beginning optuna runs for {l} {mn} ----")
            study = optuna.create_study(directions=["minimize","maximize"])
            #study.optimize(lambda t: sg_objective(t, mn, l), n_trials=50)
            study.optimize(lambda t: mh_objective(t, mn, l), n_trials=50)
            df = study.trials_dataframe()
            #df.to_csv(f"{cwd}/results/{l}/sghead/hyptuning_{mn.split('/')[-1]}.csv")
            df.to_csv(f"{cwd}/results/{l}/mhead/hyptuning_{mn.split('/')[-1]}.csv")
            del study
            print(f"\n\n\n----- Completed optuna runs for {l} {mn} ----")

if __name__ == "__main__":
    main()
