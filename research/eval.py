""" Show all scores per task/model/config """

from argparse import ArgumentParser
from pathlib import Path
import os
import json
from typing import Dict, Set
import sys

import numpy as np
import evaluate

from constants import TASK_PRETTY, TASK_GROUPS, MODEL_GROUPS, MODEL_PRETTY, MODEL_EMOJI, TASK_METRICS_PRETTY, BASELINE_MODEL, MODEL_INFO

TASK_ORDER = [t for g in TASK_GROUPS.values() for t in g]
MODEL_ORDER = [t for g in MODEL_GROUPS.values() for t in g]

TASK_METRICS = {
    "sonar-ne": ("seqeval", {}, "overall_f1"),
    "dalc": ("f1", {
        "average": "macro"
    }, "f1"),
}


def read_labels(task, filename="gold.txt"):
    path = Path("exports") / "predictions" / task / filename
    with open(path) as f:
        preds = [line.rstrip().split() for line in f]
    return preds


def vote(alts: tuple):
    alt_set = list(set(alts))
    if len(alt_set) == 1:
        return alt_set[0]

    x, n = None, 0
    for x_ in alt_set:
        n_ = alts.count(x_)
        if n_ > n or (n_ == n and alts.index(x_) < alts.index(x)):
            x, n = x_, n_

    return x


def load_predictions(task, model, ensemble):
    preds = [read_labels(task, f"{model}/{run}.txt") for run in range(1, 6)]

    if ensemble:
        pred = []
        for seqs in zip(*preds):
            pred.append([vote(labels) for labels in zip(*seqs)])  # type: ignore

        return [pred]

    return preds


def _make_serializable(obj):
    if isinstance(obj, np.generic):
        return obj.item()

    if type(obj) == list:
        return [_make_serializable(x) for x in obj]
    if type(obj) == dict:
        return {key: _make_serializable(val) for key, val in obj.items()}
    return obj


def get_runs_results(task, model, results, ensemble):
    """ results["runs"] """

    if "runs" in results:
        return results["runs"]

    gold = read_labels(task)
    preds = load_predictions(task, model, ensemble)

    metric_name, metric_kwargs, _ = TASK_METRICS.get(task, ("accuracy", {}, "accuracy"))
    metric = evaluate.load(metric_name)

    if metric_name in ["accuracy", "f1"]:
        label_set = sorted({x for seq in gold for x in seq})  # type: ignore
        gold = [label_set.index(x) for seq in gold for x in seq]  # type: ignore
        preds = [[label_set.index(x) if x in label_set else len(label_set) for seq in pred for x in seq]
                 for pred in preds]  # type: ignore

    results["runs"] = []
    for pred in preds:
        res = metric.compute(predictions=pred, references=gold, **metric_kwargs)
        res = _make_serializable(res)
        # else:  # accuracy
        # correct = [gold[i][j] == pred[i][j] for i in range(len(gold)) for j in range(len(gold[i]))]  # type: ignore
        # res = {"accuracy": sum(correct) / len(correct)}

        results["runs"].append(res)

    return results["runs"]


def get_run_scores(task, model, results, ensemble):
    # if task == "squadnl":

    #     results = load_results(task, model, args.samples, args.all)
    #     best_results = find_best_results(results)
    #     run_results = []
    #     preds = [read_labels(task, f"{model}/{run}.txt") for run in range(1, 6)]
    #     for []

    if task == "squadnl":
        x = {
            "bertje": [71.11173487920773, 70.80965333821672, 69.45080517572502, 69.45182579144128, 70.79273406876007],
            "robbert-v1": [64.57417498998457, 64.1151464804342, 65.24005399177827, 64.3825143375367, 64.450456789326],
            "robbert-v2":
            [69.70218055422004, 70.69581189593735, 72.08157515485068, 71.51703208713612, 70.94518377424832],
            "robbert-2022":
            [70.55515857100048, 70.06025992259865, 70.34919287443735, 69.62687190729017, 70.69877926436817],
            "mbert": [73.17472321821853, 70.76255665343761, 72.71153266770763, 71.78859834533712, 73.36102334311128],
            "xlmr-base": [74.31947385075583, 74.92336680327699, 73.11097268161794, 73.467639306693,
                           73.99524265053526],
            "mdeberta": [79.34847172037838, 79.31607019758957, 78.3126147790197, 78.99695171566137,
                          78.91086312905524],
            "xlmr-large":
            [81.28702225650756, 81.24688918731889, 81.66927984044442, 80.94575422399778, 81.62422406565916],
            "bert-base":
            [63.06890498412648, 62.33640132697372, 61.99400440003178, 62.01829069536716, 63.25958507551621],
            "roberta-base":
            [70.77690270172369, 69.87373767252113, 70.0278595391674, 69.1468448610056, 68.69699632887007],
            "deberta-v3-base":
            [77.98469272438754, 79.2531817267882, 78.57979198522536, 79.88616055003125, 79.98783851146266],
            "bert-large":
            [69.32228007692174, 64.76300246769206, 68.5197014859595, 65.36831953242444, 68.2250015423088],
            "roberta-large":
            [77.08634427918967, 77.59141978123088, 75.08055975425104, 75.89855272264512, 75.2096092838834],
            "deberta-v3-large":
            [84.52383876964502, 84.39200770467777, 84.74246243949197, 85.30620166168809, 84.4443900710226]
        }[model]
        return [y / 100 for y in x]

    runs_results = get_runs_results(task, model, results, ensemble)
    _, _, key = TASK_METRICS.get(task, ("accuracy", {}, "accuracy"))
    return [res[key] for res in runs_results]


def _results_path(task, model, ensemble):
    results_filename = f"{model}_ensemble.json" if ensemble else f"{model}.json"
    return Path("exports") / "results" / task / results_filename


def load_results(task, model, ensemble):
    results_path = _results_path(task, model, ensemble)
    if results_path.exists():
        with open(results_path) as f:
            return json.load(f)
    return {}


def save_results(task, model, ensemble, results):
    results_path = _results_path(task, model, ensemble)
    os.makedirs(results_path.parent, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)


def aggregate_results(tasks, models, ensemble):
    """ table[model][task] = {mean,std} """

    if BASELINE_MODEL not in models:
        models = [BASELINE_MODEL] + models

    table = {}
    for model in models:
        table[model] = {}
        for task in tasks:
            results = load_results(task, model, ensemble)
            scores = get_run_scores(task, model, results, ensemble)
            table[model][task] = {
                "mean": np.mean(scores),
                "std": np.std(scores),
            }
            save_results(task, model, ensemble, results)
    return table


def _load_correct_predictions(task, model, ensemble, gold):
    preds = load_predictions(task, model, ensemble)
    corrects = [int(t_ == p_) for pred in preds for t, p in zip(gold, pred) for t_, p_ in zip(t, p)]  # type: ignore
    return corrects


def _fit_lmer(task, ref_model, ensemble):
    import pandas as pd
    from pymer4.models import Lmer

    print(f"comparing models with `{ref_model}` for task `{task}`", file=sys.stderr)

    other_models = [m for m in MODEL_ORDER if m != ref_model]

    gold = read_labels(task)
    corrects_ref = _load_correct_predictions(task, ref_model, ensemble, gold)
    corrects_others = [_load_correct_predictions(task, m, ensemble, gold) for m in other_models]

    n_items = len([x for seq in gold for x in seq])
    n_runs = len(corrects_ref) // n_items
    assert n_runs == 1 if ensemble else 5

    model_order = ["(Intercept)"] + other_models

    corrects = [y for x in ([corrects_ref] + corrects_others) for y in x]
    models = [m for m in model_order for _ in range(n_items * n_runs)]
    items = np.tile(np.arange(n_items), n_runs * len(model_order))

    df = pd.DataFrame({
        "correct": corrects,
        "model": models,
        "item": items,
        # "item": np.tile(np.arange(n_items), n_runs + n_runs),
    })

    lm = Lmer("correct ~ model + (1|item)", data=df, family="binomial")
    res_df: pd.DataFrame = lm.fit()

    print(res_df)
    return res_df


def is_different(task, model1, model2, ensemble, lazy):
    results = load_results(task, model1, ensemble)
    if "model" not in results:
        if lazy:
            return True
        res = _fit_lmer(task, model1, ensemble).to_dict()
        results = load_results(task, model1, ensemble)
        results["model"] = res
        save_results(task, model1, ensemble, results)
    return results["model"]["P-val"][f"model{model2}"] < 0.05


def _best_task_models(tasks, models, ensemble, table, lazy):
    best_models: Dict[str, Set[str]] = {}
    for task in tasks:
        sorted_models = list(sorted(models, key=lambda model: table[model][task]["mean"], reverse=True))
        best_model = sorted_models[0]
        best_models[task] = {best_model}
        for model in sorted_models[1:]:
            if is_different(task, best_model, model, ensemble, lazy):
                break
            best_models[task].add(model)
    return best_models


def generate_table(tasks, models, ensemble, table, lazy):
    task_groups = {group: [task for task in subtasks if task in tasks] for group, subtasks in TASK_GROUPS.items()}
    last_group = list(task_groups.keys())[-1]

    cols_str = [" | ".join(["r r" for _ in range(len(g))]) for _, g in task_groups.items()]
    group_names_str = [
        f"\\multicolumn{{{len(g) * 2}}}{{{('c' if group == last_group else 'c||')}}}{{{group}}}"
        for group, g in task_groups.items()
    ]

    def _get_sep(group_name, task):
        if task == task_groups[group_name][-1]:
            if group_name == last_group:
                return ""
            return "||"
        return "|"

    task_names_str = [
        f"\\multicolumn{{2}}{{c{_get_sep(group, t)}}}{{\\textbf{{{TASK_PRETTY[t]}}}}}"
        for group, g in task_groups.items() for t in g
    ]
    metric_names_str = [
        f"\\multicolumn{{1}}{{c}}{{RER}} & \\multicolumn{{1}}{{c{_get_sep(group, t)}}}{{{TASK_METRICS_PRETTY[t]}}}"
        for group, g in task_groups.items() for t in g
    ]

    n_cols = 2 + len(metric_names_str)

    print("\\begin{tabular}{l || r || " + " || ".join(cols_str) + " }")
    print("\\toprule")
    print("\\multicolumn{2}{l||}{} & " + " & ".join(group_names_str) + " \\\\")
    print("\\midrule")
    print("& \\multicolumn{1}{c||}{\\textbf{Avg}} & " + " & ".join(task_names_str) + " \\\\")
    print("\\textbf{Model} & \\multicolumn{1}{c||}{RER} & " + " & ".join(metric_names_str) + " \\\\")
    print("\\midrule")

    baseline = {task: table[BASELINE_MODEL][task]["mean"] for task in tasks}

    # rer_scores = {model: {task: 1 - (1 - table[model][task]["mean"]) / (1 - baseline[task]) for task in tasks} for model in models}  # error reduction
    rer_scores = {
        model: {task: 1 - (1 - table[model][task]["mean"]) / (1 - baseline[task])
                for task in tasks}
        for model in models
    }  # error reduction
    rer_scores_avg = {model: np.mean(list(rer_scores[model].values())) for model in models}

    # not significantly below baseline
    ok_models: Dict[str, Set[str]] = {}
    for task in tasks:
        below_baseline_models = [model for model in models if rer_scores[model][task] < 0]
        ok_models[task] = set()
        for model in sorted(below_baseline_models, key=lambda model: table[model][task]["mean"], reverse=True):
            if is_different(task, BASELINE_MODEL, model, ensemble, lazy):
                break
            ok_models[task].add(model)

    def _format_score(task, model, score, bold, underline):
        score_str = f"{score * 100:.1f}"

        if model == BASELINE_MODEL:
            rel_score_str = "0"
        else:
            rel_score = rer_scores[model][task]
            rel_score_str = f"{rel_score * 100:.1f}"

        if bold:
            score_str = f"\\textbf{{{score_str}}}"
            rel_score_str = f"\\textbf{{{rel_score_str}}}"
        if underline:
            score_str = f"\\underline{{{score_str}}}"
            rel_score_str = f"\\underline{{{rel_score_str}}}"

        # below baseline
        if score < baseline[task] and model not in ok_models[task]:
            score_str = f"\\color{{gray}}{{{score_str}}}"
            rel_score_str = f"\\color{{gray}}{{{rel_score_str}}}"

        return f"{rel_score_str} & {score_str}"

    best_task_models = _best_task_models(tasks, models, ensemble, table, lazy)

    # for rank, model in enumerate(sorted(models, key=lambda m: rel_scores_avg[m], reverse=True), start=1):

    best_model = max(rer_scores_avg, key=rer_scores_avg.get)  # type: ignore

    rows = []
    for group, group_models in MODEL_GROUPS.items():
        group_rel_scores_avg = {model: rer_scores_avg[model] for model in group_models}
        group_best_model = max(group_rel_scores_avg, key=group_rel_scores_avg.get)  # type: ignore
        # group_best_task_models = _best_task_models(tasks, group_models, ensemble, table, lazy)

        # print(f"\\multicolumn{{{n_cols}}}{{l}}{{{group}}} \\\\")

        for model in group_models:
            if model == BASELINE_MODEL:
                rel_score_avg_str = "0"
            else:
                rel_score_avg = rer_scores_avg[model]
                rel_score_avg_str = f"{rel_score_avg * 100:.1f}"
                if rel_score_avg < 0:
                    rel_score_avg_str = f"\\color{{gray}}{{{rel_score_avg_str}}}"

            if model == best_model:
                rel_score_avg_str = f"\\textbf{{{rel_score_avg_str}}}"
            # if len(group_models) > 1 and model == group_best_model:
            #     rel_score_avg_str = f"\\underline{{{rel_score_avg_str}}}"

            # scores_str = [_format_score(task, model, table[model][task]['mean'], model in best_task_models[task], len(group_models) > 1 and model in group_best_task_models[task]) for task in tasks]
            scores_str = [
                _format_score(task, model, table[model][task]['mean'], model in best_task_models[task], False)
                for task in tasks
            ]

            print(f"\\includegraphics[height=0.8em]{{{MODEL_EMOJI[model]}}} {MODEL_PRETTY[model]} & {rel_score_avg_str} & " +
                  " & ".join(scores_str) + " \\\\")

            rows.append({
                **MODEL_INFO[model],
                # "model": model,
                "rer": rer_scores_avg[model],
                "tasks": {
                    task: {
                        "score": table[model][task]['mean'],
                        "rer": rer_scores[model][task],
                        "best": model in best_task_models[task],
                    }
                    for task in tasks
                }
            })
        print("\\midrule")

    task_rer_scores_mean = {task: np.mean([rer_scores[model][task] for model in models]) for task in tasks}
    task_scores_mean = {task: np.mean([table[model][task]["mean"] for model in models]) for task in tasks}

    task_scores_mean_str = []
    for task in tasks:
        rer_str = f"{task_rer_scores_mean[task] * 100:.1f}"
        score_str = f"{task_scores_mean[task] * 100:.1f}"
        if task_rer_scores_mean[task] < 0:
            rer_str = f"\\color{{gray}}{{{rer_str}}}"
            score_str = f"\\color{{gray}}{{{score_str}}}"
        task_scores_mean_str.append(f"{rer_str} & {score_str}")

    # global_avg = np.mean(list(rer_scores_avg.values()))
    # global_avg_str = f"{global_avg * 100:.1f}"
    # if global_avg < 0:
    #     global_avg_str = f"\\color{{gray}}{{{global_avg_str}}}"
    # print(f"\\multicolumn{{1}}{{r||}}{{Average:}} & {global_avg_str} & " + " & ".join(task_scores_mean_str) + " \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")

    return rows


def main():
    parser = ArgumentParser()
    parser.add_argument("-t", "--tasks", nargs="*", default=TASK_ORDER)
    parser.add_argument("-m", "--models", nargs="*", default=MODEL_ORDER)
    parser.add_argument("--ensemble", action="store_true")
    parser.add_argument("--unlazy", action="store_true")
    args = parser.parse_args()

    # base_path = Path("exports") / "predictions"

    table = aggregate_results(args.tasks, args.models, args.ensemble)
    rows = generate_table(args.tasks, args.models, args.ensemble, table, not args.unlazy)

    with open("exports/table.json", "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
