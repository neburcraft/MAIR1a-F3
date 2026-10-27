import numpy as np
import pandas as pd
import pingouin as pg

df = pd.read_csv("resultsOrdered.csv")

# age group as defined in the design: 18-28 young, 55-75 old
df["group"] = np.select([df["Age"].between(18, 28), df["Age"].between(55, 75)],
                        ["young", "old"], default="other")

# stop if someone falls outside both age ranges
assert (df["group"] != "other").all(), df.loc[df["group"] == "other", ["ID", "Age"]]

# a 0 is impossible on a 1-5 scale, so treat it as missing
df.loc[df["PoliteA"] == 0, "PoliteA"] = np.nan

items = ["Polite", "Realistic", "UnderstoodWell", "Useful", "Easy", "Clear"]

# one satisfaction score per participant per version
df["polite_version"] = df[[i + "A" for i in items]].mean(axis=1)
df["youthful_version"] = df[[i + "B" for i in items]].mean(axis=1)

# means per group and version
print(df.groupby("group")[["polite_version", "youthful_version"]].mean().round(2))

# long format, then the mixed anova (style within, group between)
long = df.melt(id_vars=["ID", "group"],
               value_vars=["polite_version", "youthful_version"],
               var_name="style", value_name="score")
print(pg.mixed_anova(data=long, dv="score", within="style",
                     subject="ID", between="group").round(3))
