import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np

sns.set_style("whitegrid")

# df = pd.read_csv("exports/table.csv", sep=";")

with open("exports/table.json") as f:
    data = json.load(f)

rows = [
    {
        "Language": row["lang"],
        "Model": row["type"],
        "Size": row["size"],
        "RER": np.mean([t["rer"] * 100 for t in row["tasks"].values()]),
    }
    for row in data
]

df = pd.DataFrame(rows)
# print(df)

ma = sm.OLS.from_formula("RER ~ Language", df).fit()
mb = sm.OLS.from_formula("RER ~ Model", df).fit()
mc = sm.OLS.from_formula("RER ~ Size", df).fit()

print(ma.summary())
print(mb.summary())
print(mc.summary())
print()

m0 = sm.OLS.from_formula("RER ~ Language + Model + Size", df).fit()

print(anova_lm(ma, m0))
print(anova_lm(mb, m0))
print(anova_lm(mc, m0))
print()

m1 = sm.OLS.from_formula("RER ~ Language * Model + Size", df).fit()
m2 = sm.OLS.from_formula("RER ~ Language + Model * Size", df).fit()
m3 = sm.OLS.from_formula("RER ~ Language * Size + Model", df).fit()

print(anova_lm(m0, m1))
print(anova_lm(m0, m2))
print(anova_lm(m0, m3))
print()

print(m0.summary())

# fig = sm.graphics.plot_ccpr_grid(m0)
# fig.tight_layout(pad=1.0)
# plt.savefig("exports/ccpr.png")
# plt.close()

# sns.violinplot(data=df, x="Language", y="RER")
# plt.savefig(f"exports/violin-language.png")
# plt.close()
# sns.violinplot(data=df, x="Model", y="RER")
# plt.savefig(f"exports/violin-model.png")
# plt.close()
# sns.violinplot(data=df, x="Size", y="RER")
# plt.savefig(f"exports/violin-size.png")
# plt.close()

missing = [
    { "Language": "dutch", "Model": "bert", "Size": "large" },
    { "Language": "dutch", "Model": "roberta", "Size": "large"},
    { "Language": "dutch", "Model": "debertav3", "Size": "base"},
    { "Language": "dutch", "Model": "debertav3", "Size": "large"},
    { "Language": "multilingual", "Model": "bert", "Size": "large"},
    { "Language": "multilingual", "Model": "debertav3", "Size": "large"},
]

print("\nEstimated RERs:")
# for lang, model, size in missing:
#     rer = m0.params["Intercept"] + m0.params.get(f"Language[T.{lang}]", 0) + m0.params.get(f"Model[T.{model}]", 0) + m0.params.get(f"Size[T.{size}]", 0)
#     print(f"{lang:>15} {model:>12} {size:>8} {rer:>8.1f}")

print("| Language | Model | Size | Est. RER | SE |")
print("| --- | --- | --- | ---: | ---: |")
for ex in missing:
    p = m0.get_prediction(exog=ex)
    print(f"| {ex['Language']:>8} | {ex['Model']:>8} | {ex['Size']:>8} | {p.predicted_mean[0]:>8.1f} | {p.se_mean[0]:>8.1f} |")

print()
# print(m0.conf_int(0.05))

# print()
# p = m0.get_prediction(exog=dict(Language="dutch", Model="bert", Size="large"))
# s = p.summary_frame(alpha=0.05)
# print(s)