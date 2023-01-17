import json
from scipy.stats import pearsonr
import numpy as np

tasks = {"lassy-pos": "POS", "sonar-ne": "NER", "wicnl": "WSD", "dpr": "PR", "copanl": "CR", "sicknl-nli": "NLI", "dbrd": "SA", "dalc": "ALD"}

with open("exports/table.json") as f:
    table = json.load(f)

scores = {}
for task in tasks:
    scores[task] = []
    for model in table:
        scores[task].append(model["tasks"][task]["rer"])

task_labels = [f"\\textbf{{{tasks[t]}}}" for t in tasks]

print(f"""\\begin{{tabular}}{{l | {' '.join(["c" for _ in tasks])} }}
\\toprule""")
print(f" & " + " & ".join(task_labels) + " \\\\")
print("\\midrule")
totals = {}

rows = []
indices = []
for task1 in tasks:
    maxr = maxi = 0
    row = []
    totals[task1] = []
    for task2 in tasks:
        if task1 == task2:
            row.append("-")
            continue
        res = pearsonr(scores[task1], scores[task2])
        r = res.statistic
        s = f"{r:.2f}"
        if res.pvalue > 0.05:
            s = f"\\color{{gray}}{{{s}}}"
        if r > maxr:
            maxr = r
            maxi = len(row)
        row.append(s)
        totals[task1].append(r)
    # row[maxi] = f"\\textbf{{{row[maxi]}}}"
    rows.append(row)
    indices.append(maxi)

for i, j in enumerate(indices):
    rows[j][i] = f"\\textbf{{{rows[j][i]}}}"

for label, row in zip(task_labels, rows):
    print(label + " & " + " & ".join(row) + " \\\\")

print("\\midrule")
avg_corrs = [f"{np.mean(totals[task]):.2f}" for task in tasks]
print(" & " + " & ".join(avg_corrs) + " \\\\")

print("""\\bottomrule
\\end{tabular}""")
