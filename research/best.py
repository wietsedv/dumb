""" Show all scores per task/model/config for the best hyperparameters (highest dev score) """

from argparse import ArgumentParser
from pathlib import Path
import json
from glob import glob
from typing import Dict, Any
import numpy as np

from constants import get_train_params, get_test_seeds, EVAL_SEED, CONFIG_KEYS, TASK_ORDER, TASK_PRETTY, MODEL_ORDER, MODEL_PRETTY


def load_results(tasks: list, models: list, samples: int, all: bool):
    results = {}
    for task_name in tasks:
        for model_name in models:
            for path in glob(f"output/{task_name}/{model_name}/*/eval_results.json"):
                path = Path(path)

                # task_name = path.parent.parent.parent.name
                # model_name = path.parent.parent.name
                config_name = path.parent.name

                whitelist = get_train_params(task_name)

                with open(path) as f:
                    res = json.load(f)

                if "eval_pearsonr" in res:
                    metric = "pearsonr"
                elif "eval_f1" in res:
                    metric = "f1"
                else:
                    metric = "accuracy"
                score = res[f"eval_{metric}"]

                cfg = {c[0]: float(c[1:]) for c in path.parent.name.split("-")}
                cfg["e"] = int(cfg["e"])
                cfg["b"] = int(cfg["b"])
                if int(cfg.pop("s")) != EVAL_SEED:
                    continue
                if int(cfg.pop("m")) != samples:
                    continue

                if not all:
                    ok = True
                    for key, values in whitelist.items():
                        if cfg[key] not in values:
                            ok = False
                            break
                    if not ok:
                        continue

                cfg = {CONFIG_KEYS.get(k, k): v for k, v in cfg.items()}
                if (task_name, model_name) not in results:
                    results[(task_name, model_name)] = []
                results[(task_name, model_name)].append({
                    **cfg, "config": config_name,
                    "metric": metric,
                    "dev_score": score
                })
    return results


def find_best_results(results):
    best_results = {}
    for (task, model) in results:
        res_list = results[(task, model)]
        res = max(res_list, key=lambda x: x["dev_score"])

        res["dev_score"] = res["dev_score"] * 100

        # eval and test scores with different seeds
        res["eval_scores"], res["test_scores"] = [], []
        res["eval_seeds"] = []
        res["eval_paths"] = []
        test_seeds = get_test_seeds(task, model)
        for test_seed in test_seeds:
            config_name = res["config"].replace(f"s{EVAL_SEED}", f"s{test_seed}")
            path = Path("output") / task / model / config_name / "all_results.json"
            if path.exists():
                with open(path) as f:
                    d = json.load(f)
                    if f"predict_{res['metric']}" in d:
                        res["eval_scores"].append(d[f"eval_{res['metric']}"] * 100)
                        res["test_scores"].append(d[f"predict_{res['metric']}"] * 100)
                        res["eval_seeds"].append(test_seed)
                        res["eval_paths"].append(path.parent)
        res["test_score_mean"] = float(np.mean(res["test_scores"])) if len(res["test_scores"]) > 0 else 0
        res["test_score_std"] = float(np.std(res["test_scores"])) if len(
            res["test_scores"]) == len(test_seeds) else (len(test_seeds) - len(res["test_scores"]))

        res["num_runs"] = len(res_list)
        best_results[(task, model)] = res
    return best_results


def show_best_parameters(args, results, best_results, latex: bool, incomplete: bool):
    if latex:
        print('\\begin{tabular}{l l | c c c c | c c }')
        print("\\toprule")
        print('Task & Model & Epochs & Warmup & LR & Dropout & Dev & Test \\\\')
        print("\\midrule")
    else:
        print("| task | model | runs | epochs | batchsize | warmup | lr | dropout | decay | dev | test |" +
              (" path |" if args.path else ""))
        print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: |---: |" + (" --- |" if args.path else ""))

    all_dev_scores, all_test_scores = [], []
    for (task, model) in sorted(results, key=lambda x: (TASK_ORDER.index(x[0]), MODEL_ORDER.index(x[1]))):
        res_list = results[(task, model)]
        res = best_results[(task, model)]
        path = f"output/{task}/{model}/{res.pop('config')}"

        # train runs
        n_runs = len(res_list)
        n_runs_tgt = 1
        for v in get_train_params(task).values():
            n_runs_tgt *= len(v)

        dev_score = res.pop("dev_score")
        all_dev_scores.append(dev_score)

        test_score_mean = res.pop("test_score_mean")
        test_score_std = res.pop("test_score_std")

        test_scores = res.pop("test_scores")
        all_test_scores.append(test_score_mean)

        test_score_std = float(np.std(test_scores))

        def _fmt(key):
            value_list = [r[key] for r in res_list]
            values = sorted(set(value_list))
            values_str = [f"<{e}>" if e == res[key] else str(e) for e in values]
            # values_str = [f"{s} ({value_list.count(v)})" for v, s in zip(values, values_str)]
            s = f"{{{', '.join(values_str)}}}"
            if latex:
                s = s.replace("{", "\\{").replace("}", "\\}").replace("<", "\\textbf{").replace(">", "}")
            return s

        # eval runs
        test_seeds = get_test_seeds(task, model)
        test_prefix = "" if len(test_scores) == len(test_seeds) else f"({len(test_scores)}/{len(test_seeds)}) "

        if incomplete and n_runs == n_runs_tgt and len(test_scores) == len(test_seeds):
            continue

        test_score_std_str = f"s={test_score_std:.2f}" if type(test_score_std) == float else test_score_std

        if latex:
            print(
                f"{TASK_PRETTY[task]} & {MODEL_PRETTY[model]} & {_fmt('num_train_epochs')} & {_fmt('warmup_ratio')} & {_fmt('learning_rate')} & {_fmt('hidden_dropout_prob')} & {dev_score:.1f} & {test_score_mean:.1f} \\\\"
            )
        else:
            print(
                f"| {task} | {model} | {n_runs} / {n_runs_tgt} | {_fmt('num_train_epochs')} | {_fmt('train_batch_size')} | "
                +
                f"{_fmt('warmup_ratio')} | {_fmt('learning_rate')} | {_fmt('hidden_dropout_prob')} | {_fmt('weight_decay')} | "
                + f"{dev_score:.1f} | {test_prefix}{test_score_mean} ({test_score_std_str}) |" +
                (f" {path} |" if args.path else ""))

    if not latex:
        dev_score = sum(all_dev_scores) / len(all_dev_scores)
        test_score = sum(all_test_scores) / len(all_test_scores)
        print(f"| Average |  |  |  |  |  |  |  |  | {dev_score:.1f} | {test_score:.1f} |" +
              ("  |" if args.path else ""))
    else:
        print("\\end{tabular}")


def _aggregate_summary(tasks, models, best_results):
    model_scores = {}
    model_runs = {}
    task_metrics = {}
    for model in models:
        model_scores[model] = [(best_results[(task, model)]["test_score_mean"],
                                best_results[(task, model)]["test_score_std"]) if (task, model) in best_results else
                               (None, None) for task in tasks]
        model_runs[model] = [
            best_results[(task, model)]["num_runs"] if (task, model) in best_results else None for task in tasks
        ]

        # determine used metric
        for task in tasks:
            if (task, model) not in best_results:
                continue
            metric = best_results[(task, model)]["metric"]
            if task in task_metrics:
                assert task_metrics[task] == metric
            else:
                task_metrics[task] = metric

    target_runs = {}
    for task in tasks:
        n = 1
        for v in get_train_params(task).values():
            n *= len(v)
            target_runs[task] = n

    return model_scores, model_runs, target_runs, task_metrics


def _format_scores_markdown(tasks, model, scores, scores_str, min_scores, max_scores, max_scores_base, target_runs,
                            model_runs):
    runs = model_runs[model]

    # mark worst overall
    scores_str = [
        f"<i style='color:red'>{score_str}</i>" if score is not None and score - min_score < 0.05 else score_str
        for score_str, (score, std), min_score in zip(scores_str, scores, min_scores)
    ]

    # mark best overall
    scores_str = [
        f"<strong style='color:blue'>{score_str}</strong>"
        if score is not None and max_score - score < 0.05 else score_str
        for score_str, (score, std), max_score in zip(scores_str, scores, max_scores)
    ]

    # mark best base model
    if not model.endswith("-large"):
        scores_str = [
            f"<ins style='color:green'>{score_str}</ins>"
            if score is not None and max_score - score < 0.05 else score_str
            for score_str, (score, std), max_score in zip(scores_str, scores, max_scores_base)
        ]

    # mark std
    stds_str = [f's={std:.2f}' if type(std) == float else std for score, std in scores]
    scores_str = [
        f'{score_str} <sub>{std_str}</sub>' if std_str is not None else score_str
        for score_str, std_str in zip(scores_str, stds_str)
    ]
    # scores_str = [f"<ins>{score_str}</ins>" if max_score - score < 0.005 else score_str for score_str, score, max_score in zip(scores_str, scores, max_scores)]
    # scores_str = [f"({task_scores.index(score) + 1}) {score_str}" for score_str, score, task_scores in zip(scores_str, scores, scores_per_task)]

    # mark incomplete
    avg_score_str = scores_str[-1]
    scores_str = [
        f"({runs}/{target_runs[task]}) {score_str}" if runs != target_runs[task] else score_str
        for score_str, runs, task in zip(scores_str, runs, tasks)
    ]
    scores_str.append(avg_score_str)
    return scores_str


def _format_scores_latex(model, scores, scores_str, min_scores, max_scores, max_scores_base):
    # mark worst overall
    scores_str = [
        f"\\textit{{{score_str}}}" if score - min_score < 0.05 else score_str
        for score_str, (score, std), min_score in zip(scores_str, scores, min_scores)
    ]

    # mark best overall
    scores_str = [
        f"\\textbf{{{score_str}}}" if max_score - score < 0.01 else score_str
        for score_str, (score, std), max_score in zip(scores_str, scores, max_scores)
    ]

    # mark best base
    if not model.endswith("-large"):
        scores_str = [
            f"\\underline{{{score_str}}}" if max_score - score < 0.01 else score_str
            for score_str, (score, std), max_score in zip(scores_str, scores, max_scores_base)
        ]

    return scores_str


# def _dumb_scale(x):
#     return 100 - np.sqrt(1 - x / 100) * 100
#     # return 100 - (1 - x / 100) ** (1/10) * 100

# def _dumb_score(scores):
#     return np.mean([_dumb_scale(score or np.nan) for score, _ in scores])


def show_summary(best_results: dict, tasks, models, latex: bool, sort):
    if latex:
        print('\\begin{tabular}{l | c c | c c | c c c | c c | c }')
        print('\\toprule')
        print(
            '& \\multicolumn{2}{c|}{Word} & \\multicolumn{2}{c|}{Word Pair} & \\multicolumn{3}{c|}{Sentence Pair} & \\multicolumn{2}{c|}{Document} & \\\\'
        )
        print('Model & ' + ' & '.join([f"\\textbf{{{TASK_PRETTY[t]}}}" for t in tasks]) + " & Avg \\\\")
        print('\\midrule')
    else:
        print("| model | " + " | ".join(tasks) + " | score |")
        print("| ----- | " + " ---: |" * (len(tasks) + 1))

    model_scores, model_runs, target_runs, task_metrics = _aggregate_summary(tasks, models, best_results)

    if not latex:
        print("|       | " + " | ".join([task_metrics.get(t, "n/a") for t in tasks]) + " |  |")

    task_scores = [[model_scores[model][i][0] or np.nan for model in models] for i in range(len(tasks))]

    task_ranks = [np.argsort(scores).tolist() for scores in task_scores]
    model_ranks = {
        model: np.mean([len(ranks) - ranks.index(i) for ranks in task_ranks])
        for i, model in enumerate(models)
    }

    task_scores.append([np.mean([score or np.nan for score, _ in model_scores[model]]) for model in models])

    min_scores = [min(scores) for scores in task_scores]
    max_scores = [max(scores) for scores in task_scores]
    max_scores_base = [
        max([score for score, model in zip(scores, models) if not model.endswith("-large")]) for scores in task_scores
    ]

    if sort:

        def _sort_key(model):
            if (sort, model) not in best_results:
                return 0
            return best_results[(sort, model)]["test_score_mean"]

        models = sorted(models, key=_sort_key, reverse=True)

    for model in models:
        scores = model_scores[model]
        scores_str = [f'{score:.1f}' if score is not None else "n/a" for score, std in scores]

        # dumb_score = _dumb_score(model_scores[model])
        avg_rank = model_ranks[model]
        avg_score = np.mean([score or np.nan for score, _ in model_scores[model]])

        scores = scores + [(None if np.isnan(avg_score) else avg_score.item(), None)]
        scores_str.append(f"{avg_score:.1f} / {avg_rank:.1f}")

        if latex:
            if model == "mbert":
                print('\\midrule')
            scores_str = _format_scores_latex(model, scores, scores_str, min_scores, max_scores, max_scores_base)
            print(f"{MODEL_PRETTY[model]} & " + " & ".join(scores_str) + " \\\\")
        else:
            scores_str = _format_scores_markdown(tasks, model, scores, scores_str, min_scores, max_scores,
                                                 max_scores_base, target_runs, model_runs)
            print(f"| {model} | " + " | ".join(scores_str) + f" |")

    mean_scores_str = [f"{np.mean(scores):.1f}" for scores in task_scores]
    std_scores_str = [f"{np.std(scores):.1f}" for scores in task_scores]

    if latex:
        print('\\midrule')
        print("\\multicolumn{1}{r|}{\\textit{Average}} & " + " & ".join(mean_scores_str) + " \\\\")
        print('\\bottomrule')
        print('\\end{tabular}')
    else:
        avgs_str = [f'{score} <sub>s={std}</sub>' for score, std in zip(mean_scores_str, std_scores_str)]
        print("| Average | " + " | ".join(avgs_str) + " |")


def main():
    parser = ArgumentParser()
    parser.add_argument("-t", "--tasks", nargs="*", default=TASK_ORDER)
    parser.add_argument("-m", "--models", nargs="*", default=MODEL_ORDER)
    parser.add_argument("--samples", type=int, default=0)
    parser.add_argument("--path", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--params", action="store_true")
    parser.add_argument("--latex", action="store_true")
    parser.add_argument("--incomplete", action="store_true")
    parser.add_argument("--sort", default=None, choices=TASK_ORDER)
    args = parser.parse_args()

    results = load_results(args.tasks, args.models, args.samples, args.all)
    best_results = find_best_results(results)

    if args.params:
        show_best_parameters(args, results, best_results, args.latex, args.incomplete)
    else:
        show_summary(best_results, args.tasks, args.models, args.latex, args.sort)


if __name__ == "__main__":
    main()
