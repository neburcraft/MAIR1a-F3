import numpy as np
import pandas as pd
import pingouin as pg
from scipy import stats

df = pd.read_csv("resultsOrdered.csv")

# age group as defined in the design: 18-28 young, 55-75 old
df["group"] = np.select([df["Age"].between(18, 28), df["Age"].between(55, 75)],
                        ["young", "old"], default="other")

# stop if someone falls outside both age ranges
assert (df["group"] != "other").all(), df.loc[df["group"] == "other", ["ID", "Age"]]

print(df.groupby("group")["Age"].agg(["count", "mean", "std", "min", "max"]).round(2))

# a 0 is impossible on a 1-5 scale, so treat it as missing
df.loc[df["PoliteA"] == 0, "PoliteA"] = np.nan

items = ["Polite", "Realistic", "UnderstoodWell", "Useful", "Easy", "Clear"]

# one satisfaction score per participant per version
df["polite_version"] = df[[i + "A" for i in items]].mean(axis=1)
df["youthful_version"] = df[[i + "B" for i in items]].mean(axis=1)

# means per group and version
print("mean and std per group and version")
print(df.groupby("group")[["polite_version", "youthful_version"]].agg(["mean", "std"]).round(2))

# long format, then the mixed anova (style within, group between)
long = df.melt(id_vars=["ID", "group"],
               value_vars=["polite_version", "youthful_version"],
               var_name="style", value_name="score")
print(pg.mixed_anova(data=long, dv="score", within="style",
                     subject="ID", between="group").round(3))

print("std en mean van english profenciency and chatbot experience for young and old")
print(df.groupby("group")[["EnglishProficiency", "ChatbotExperience"]].agg(["mean", "std"]).round(2))

print("ttest for young and old in english proficiency")
young = df.loc[df["group"] == "young", "EnglishProficiency"]
old = df.loc[df["group"] == "old", "EnglishProficiency"]
print(stats.ttest_ind(young, old))

print("ttest for young and old in chatbotexperience")
young = df.loc[df["group"] == "young", "ChatbotExperience"]
old = df.loc[df["group"] == "old", "ChatbotExperience"]
print(stats.ttest_ind(young, old))

# StartVersion says which version was done first, so build first/second scores
def first_second(a, b):
    first = np.where(df["StartVersion"] == "A", a, b)
    second = np.where(df["StartVersion"] == "A", b, a)
    return first, second

df["sat_first"], df["sat_second"] = first_second(df["polite_version"], df["youthful_version"])
df["und_first"], df["und_second"] = first_second(df["UnderstoodWellA"], df["UnderstoodWellB"])

def run(first, second, label):
    long = df.melt(id_vars=["ID", "group"], value_vars=[first, second],
                   var_name="order", value_name="score")
    print("\n" + label)
    print(df.groupby("group")[[first, second]].agg(["mean", "std"]).round(2))
    print(pg.mixed_anova(data=long, dv="score", within="order",
                         subject="ID", between="group").round(3))

run("sat_first", "sat_second", "satisfaction by order")
run("und_first", "und_second", "understood me well by order")

print(df["StartVersion"].value_counts())
print(pd.crosstab(df["group"], df["StartVersion"]))
