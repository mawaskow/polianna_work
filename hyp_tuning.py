import optuna
from train_sghead import finetune_sghead_model
from train_mhead import finetune_mhead_model
from create_datasets import get_label_set

def sg_objective(trial):
    params = {
        "num_epochs": 30,
        "lr": trial.suggest_float("lr", 1e-6, 5e-4, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
        "max_length": trial.suggest_categorical("max_length", [256, 512]),#, 1024, ]),
        "num_warmup_steps": 0,
        "patience": 5
    }

    metrics = finetune_sghead_model(
        model_name="microsoft/deberta-v3-base",
        label_list=get_label_set("a", "sghead"),
        model_save_addr="./models/a/sghead/optuna",
        dsdct_dir="./inputs/a/sghead_dsdcts",
        r=0,
        params=params
    )

    return metrics["overall_f1"]


study = optuna.create_study(direction="maximize")
study.optimize(sg_objective, n_trials=40)
print("Best params:", study.best_params)

import optuna

def mh_objective(trial):
    params = {
        "num_epochs": 30,
        "lr": trial.suggest_float("lr", 1e-6, 5e-4, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
        "max_length": trial.suggest_categorical("max_length", [256, 512]),#, 768, 1024]),
        "num_warmup_steps": 0,
        "patience": 5,
        "dropout": 0.1,
    }

    # run training
    metrics = finetune_mhead_model(
        model_name="microsoft/deberta-v3-base",
        head_lst=get_label_set("a", "mhead"),
        model_save_addr="./models/a/mhead/optuna",
        dsdct_dir="./inputs/a/mhead_dsdcts",
        r=0,
        params=params
    )

    # Optuna minimizes the returned value
    return metrics["avg_eval_loss"]