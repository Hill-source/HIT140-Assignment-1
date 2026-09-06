
import pandas as pd
import numpy as np
import scipy.stats as st

# PART 1: DATA WRANGLING
# Purpose: read, inspect and merge the two source datasets.
# Player and Squad identify a player and are used as the merge keys.
# Pos identifies goalkeepers and creates position-based sampling groups.
# Age separates players into the Younger and Older groups.
# Min records playing time and applies the minimum 90-minute requirement.
# Fld records the total number of fouls suffered by a player.

# The first CSV row contains category labels such as "Performance".
# header=1 skips it and uses the second row as the column names.
standard = pd.read_csv("dataset_fifaworldcup.csv", header=1)
criterias = pd.read_csv("dataset_criterias.csv", header=1)

standard = standard[["Player", "Squad", "Min"]].copy()
criterias = criterias[["Player", "Squad", "Pos", "Age", "Fld"]].copy()

# Preview each table and check its structure and data quality.
for table_name, table in {"Standard": standard, "Criterias": criterias}.items():
    print(f"\n{table_name.upper()} DATA CHECK")
    print(table.head())
    print("Shape:", table.shape)
    print("Dtypes:\n", table.dtypes)
    print("Missing values:\n", table.isna().sum())
    print("Duplicate Player-Squad keys:", table.duplicated(["Player", "Squad"]).sum())

# Merge matching players from both tables using Player and Squad.
players = pd.merge(
    criterias,
    standard,
    on=["Player", "Squad"],
    how="inner",
    validate="one_to_one",)

# PART 2: DATA PREPARATION AND SAMPLING
# Purpose: define the eligible population and draw a representative sample.
# Check missing values and duplicate player records before filtering.
required_columns = ["Player", "Squad", "Pos", "Age", "Min", "Fld"]
print("\nMissing values after conversion:\n", players[required_columns].isna().sum())
print("Duplicate rows:", players.duplicated(["Player", "Squad"]).sum())
players = players.dropna(subset=required_columns)

# Keep non-goalkeepers who played at least 90 minutes.
eligible_population = players[
    (players["Pos"] != "GK") &
    (players["Min"] >= 90)
].copy()

eligible_population["Age_Group"] = np.where(
    eligible_population["Age"] <= 27,
    "Younger",
    "Older")

# Standardise fouls suffered to a rate per 90 minutes.
eligible_population["Fouls_Suffered_per90"] = (
    eligible_population["Fld"] / eligible_population["Min"] * 90)

print("\nELIGIBLE POPULATION")
print("Size:", len(eligible_population))
print("Age-group counts:\n", eligible_population["Age_Group"].value_counts())
print("Position counts:\n", eligible_population["Pos"].value_counts().sort_index())

# Some players have mixed positions, such as MFFW, FWMF, DFMF or MFDF.
# Use the first listed position to group each player as DF, MF or FW.

eligible_population["Pos_Group"] = eligible_population["Pos"].str[:2]
print("Position groups in the eligible population:")
print(eligible_population["Pos_Group"].value_counts())
# Use a fixed sample of 100 players.
sample_size = 100
# Allocate the sample proportionally across position and age groups.
position_age_counts = eligible_population.groupby(
    ["Pos_Group", "Age_Group"]
).size()

position_age_sample = (
    position_age_counts / len(eligible_population) * sample_size
).round().astype(int)

print("\nSample allocation by position and age group:")
print(position_age_sample)

# Randomly sample within each position-age group.
# random_state=42 makes the sample reproducible.
sample_parts = []
for (position, age_group), n in position_age_sample.items():
    group = eligible_population[
        (eligible_population["Pos_Group"] == position)
        & (eligible_population["Age_Group"] == age_group)]
    sample_parts.append(
        group.sample(n=n, random_state=42))
    
# Combine the sampled groups into one DataFrame.
player_sample = pd.concat(sample_parts, ignore_index=True)

print("\nFinal sample size:", len(player_sample))
print("\nSample position counts:")
print(player_sample["Pos_Group"].value_counts())
print("\nSample age-group counts:")
print(player_sample["Age_Group"].value_counts())

# PART 3: DESCRIPTIVE STATISTICS
# Purpose: summarise fouls suffered per 90 for the sample and both age groups.
def descriptive_statistics(values):
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    return pd.Series(
        {
            "n": values.count(),
            "mean": values.mean(),
            "median": values.median(),
            "standard deviation": values.std(ddof=1),
            "variance": values.var(ddof=1),
            "min": values.min(),
            "max": values.max(),
            "Q1": q1,
            "Q3": q3,
            "IQR": q3 - q1,})

sample_values = player_sample["Fouls_Suffered_per90"]
younger_values = player_sample.loc[
    player_sample["Age_Group"] == "Younger", "Fouls_Suffered_per90"]
older_values = player_sample.loc[
    player_sample["Age_Group"] == "Older", "Fouls_Suffered_per90"]

summary_table = pd.DataFrame(
    {
        "Overall sample": descriptive_statistics(sample_values),
        "Younger": descriptive_statistics(younger_values),
        "Older": descriptive_statistics(older_values),}).T

print("\nDESCRIPTIVE STATISTICS")
print(summary_table.round(4).to_string())

# Describe the distribution and flag potential outliers using the 1.5 x IQR rule.
q1 = sample_values.quantile(0.25)
q3 = sample_values.quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
outliers = player_sample[
    (player_sample["Fouls_Suffered_per90"] < lower_bound)
    | (player_sample["Fouls_Suffered_per90"] > upper_bound)]

print("\nDISTRIBUTION AND POTENTIAL OUTLIERS")
print(f"Overall skewness: {sample_values.skew():.4f}")
print("Skewness by age group:\n", player_sample.groupby("Age_Group")["Fouls_Suffered_per90"].skew())
print(f"Outlier bounds: [{lower_bound:.4f}, {upper_bound:.4f}]")

# PART 4: INFERENTIAL STATISTICS (95% CONFIDENCE INTERVAL)
# Purpose: estimate the eligible population mean from the sample mean.
n = sample_values.count()
sample_mean = sample_values.mean()
sample_std = sample_values.std(ddof=1)
standard_error = sample_std / np.sqrt(n)
critical_value = st.t.ppf(1 - 0.05 / 2, df=n - 1)
margin_of_error = critical_value * standard_error
ci_lower = sample_mean - margin_of_error
ci_upper = sample_mean + margin_of_error

print("\n95% CONFIDENCE INTERVAL")
print(f"Mean: {sample_mean:.4f}")
print(f"Standard error: {standard_error:.4f}")
print(f"Critical t-value: {critical_value:.4f}")
print(f"Margin of error: {margin_of_error:.4f}")
print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
print(
    "We are 95% confident that the mean fouls suffered per 90 minutes among "
    f"eligible outfield players is between {ci_lower:.2f} and {ci_upper:.2f}.")

# PART 5: INFERENTIAL STATISTICS (TWO-SAMPLE T-TEST)
# Purpose: compare mean fouls suffered per 90 between the two age groups.
sample1 = younger_values.to_numpy()
sample2 = older_values.to_numpy()

younger_mean = st.tmean(sample1)
younger_std = st.tstd(sample1)
younger_n = len(sample1)

older_mean = st.tmean(sample2)
older_std = st.tstd(sample2)
older_n = len(sample2)

# H0: the Younger and Older population means are equal.
# Ha: the Younger and Older population means are different.
# Use a two-sided Welch test because equal variances are not assumed.
t_statistic, p_value = st.ttest_ind_from_stats(
    younger_mean,
    younger_std,
    younger_n,
    older_mean,
    older_std,
    older_n,
    equal_var=False,
    alternative="two-sided",
)
mean_difference = younger_mean - older_mean

print("\nTWO-SAMPLE T-TEST")
print(f"Younger: n = {younger_n}, mean = {younger_mean:.4f}")
print(f"Older: n = {older_n}, mean = {older_mean:.4f}")
print(f"Difference (Younger - Older): {mean_difference:.4f}")
print(f"t-statistic: {t_statistic:.4f}")
print(f"p-value: {p_value:.4f}")

if p_value <= 0.05:
    print(
        "CONCLUSION: Reject H0. The sample provides evidence that Younger and "
        "Older players differ in mean fouls suffered per 90 minutes.")
else:
    print(
        "CONCLUSION: Fail to reject H0. The sample does not provide sufficient "
        "evidence that Younger and Older players differ in mean fouls suffered "
        "per 90 minutes.")
    
# PART 6: VISUALISATION
# Purpose: present the distribution and age-group comparison visually.

# 6.1 T-distribution for the two-sided Welch t-test

import matplotlib.pyplot as plt

# Calculate Welch's approximate degrees of freedom.
var_young = younger_std ** 2
var_old = older_std ** 2

welch_df = (
    (var_young / younger_n + var_old / older_n) ** 2
    /
    (
        (var_young / younger_n) ** 2 / (younger_n - 1)
        +
        (var_old / older_n) ** 2 / (older_n - 1)
    )
)

# Use a 5% significance level.
alpha = 0.05

# Find the critical value for a two-sided test.
critical_t = st.t.ppf(
    1 - alpha / 2,
    df=welch_df
)

# Generate the coordinates for the t-distribution curve.
x = np.linspace(-4, 4, 1000)
y = st.t.pdf(x, df=welch_df)

plt.figure(figsize=(9, 5))

# Draw the t-distribution.
plt.plot(x, y, linewidth=2)

# Shade both tails beyond the absolute observed t-statistic.
left_tail = x <= -abs(t_statistic)
right_tail = x >= abs(t_statistic)

plt.fill_between(
    x[left_tail],
    y[left_tail],
    alpha=0.4
)

plt.fill_between(
    x[right_tail],
    y[right_tail],
    alpha=0.4
)

# Mark the observed t-statistic in both tails.
plt.axvline(
    -abs(t_statistic),
    linestyle="--",
    linewidth=2
)

plt.axvline(
    abs(t_statistic),
    linestyle="--",
    linewidth=2
)

# Mark the null-hypothesis centre at zero.
plt.axvline(
    0,
    linewidth=1
)

# Label the null value and observed t-statistic.
plt.text(
    0,
    max(y) * 1.03,
    "H₀: Difference = 0",
    ha="center",
    fontsize=11
)

plt.text(
    abs(t_statistic),
    max(y) * 0.45,
    f"t = {t_statistic:.2f}",
    ha="left",
    fontsize=10
)

plt.text(
    -abs(t_statistic),
    max(y) * 0.45,
    f"t = {-abs(t_statistic):.2f}",
    ha="right",
    fontsize=10
)

# Display the calculated two-sided p-value.
plt.text(
    0,
    max(y) * 0.78,
    f"Two-sided p-value = {p_value:.3f}",
    ha="center",
    fontsize=12
)

plt.title(
    "Two-Sided t-Test: Younger vs Older Players",
    fontsize=14
)

plt.xlabel("t-statistic")
plt.ylabel("Probability Density")

plt.tight_layout()

# Save a high-resolution image for the presentation.
plt.savefig(
    "task4_t_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nT-distribution chart saved as task4_t_distribution.png")
print(f"Welch degrees of freedom: {welch_df:.2f}")
print(f"Critical t-value: ±{critical_t:.2f}")

plt.figure(figsize=(8, 5))

plt.hist(
    sample_values,
    bins=10,
    edgecolor="black"
)

plt.title("Distribution of Fouls Suffered per 90 Minutes")
plt.xlabel("Fouls Suffered per 90 Minutes")
plt.ylabel("Number of Players")

plt.tight_layout()

# Save the histogram as a high-resolution image.
plt.savefig(
    "task4_histogram.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Histogram saved as task4_histogram.png")
print(os.path.abspath("task4_histogram.png"))

# 6.2 Mean fouls suffered per 90 with 95% confidence intervals

import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import os


# Calculate the sample size, mean and standard deviation for each age group.

younger_n = len(younger_values)
older_n = len(older_values)

younger_mean = younger_values.mean()
older_mean = older_values.mean()

younger_std = younger_values.std(ddof=1)
older_std = older_values.std(ddof=1)



# Calculate the 95% confidence-interval margin for each age group.

alpha = 0.05

# Younger group confidence interval
younger_se = younger_std / np.sqrt(younger_n)

younger_t_critical = st.t.ppf(
    1 - alpha / 2,
    df=younger_n - 1
)

younger_margin = younger_t_critical * younger_se

# Older group confidence interval
older_se = older_std / np.sqrt(older_n)

older_t_critical = st.t.ppf(
    1 - alpha / 2,
    df=older_n - 1
)

older_margin = older_t_critical * older_se

# Prepare the group labels, means and error-bar values.

groups = [
    "Younger\n(27 or under)",
    "Older\n(28 or over)"
]

means = [
    younger_mean,
    older_mean
]

error_bars = [
    younger_margin,
    older_margin
]

# Draw the group means with 95% confidence-interval error bars.

plt.figure(figsize=(8, 5))

bars = plt.bar(
    groups,
    means,
    yerr=error_bars,
    capsize=8,
    edgecolor="black"
)

# Display each sample mean above its error bar.

for bar, mean, margin in zip(bars, means, error_bars):

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        mean + margin + 0.05,
        f"{mean:.2f}",
        ha="center",
        fontweight="bold"
    )

# Add a clear title and axis labels.

plt.title(
    "Mean Fouls Suffered per 90 by Age Group\n"
    "(bars show 95% confidence intervals)"
)

plt.ylabel("Mean Fouls Suffered per 90 Minutes")
plt.xlabel("Age Group")

# Add the Welch t-statistic and p-value below the chart.

plt.figtext(
    0.5,
    0.01,
    f"Welch t = {t_statistic:.2f}    p = {p_value:.3f}",
    ha="center"
)

# Save the chart as a high-resolution image.

plt.tight_layout(rect=[0, 0.05, 1, 1])

plt.savefig(
    "task4_ttest_mean_ci.png",
    dpi=300,
    bbox_inches="tight"
)

print("\nMean and confidence-interval chart saved at:")
print(os.path.abspath("task4_ttest_mean_ci.png"))

plt.show()
