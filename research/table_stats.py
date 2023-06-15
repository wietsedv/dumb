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

rows = [{"Language": row["lang"], "Model": row["type"], "Size": row["size"], "RER": np.mean([t["rer"] * 100 for t in row["tasks"].values()]) } for row in data]

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
