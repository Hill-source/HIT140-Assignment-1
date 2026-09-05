# HIT140 - FIFA World Cup 2026 Player Analysis
# Question 1: Mean Playing Time per Appearance
# Main steps:
# 1. Load and inspect the data
# 2. Define the eligible player population
# 3. Calculate playing time per appearance
# 4. Create position and age groups
# 5. Calculate the population mean
# 6. Draw a proportionate stratified random sample
# 7. Calculate descriptive statistics
# 8. Construct a 95% confidence interval
# 9. Simulate the sampling distribution
# 10. Conduct a two-sample t-test

# 1. IMPORT LIBRARIES

from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


# 2. LOAD DATA

# Get the folder where this Python file is located.
current_folder = Path(__file__).parent

# The CSV file is stored in the same folder as this Python file.
file_path = current_folder / "2026_WC_Player_Playing_time.csv"
df = pd.read_csv(file_path, header=1)

# 3. SELECT VARIABLES NEEDED FOR QUESTION 1

task1_df = df[
    [
        "Player",
        "Pos",
        "Squad",
        "Age",
        "MP",
        "Min",
        "Mn/MP"
    ]
].copy()

# Check missing values in Min

missing_minutes = task1_df[task1_df["Min"].isna()]

print("\nNumber of players with missing minutes:")
print(len(missing_minutes))

print("\nMP values for players with missing minutes:")
print(missing_minutes["MP"].value_counts())


# 4. DEFINE THE ELIGIBLE PLAYER POPULATION

# The population includes:
# - players who made at least one appearance (MP >= 1)
# - outfield players only

population_df = task1_df[
    (task1_df["MP"] >= 1) &
    (~task1_df["Pos"].str.contains("GK", na=False))
].copy()

population_df.reset_index(drop=True, inplace=True)

print("\n ELIGIBLE POPULATION ")

print("Original number of players:", len(task1_df))
print("Eligible population size:", len(population_df))

print("\nPosition distribution:")
print(population_df["Pos"].value_counts())


# 5. CALCULATE PLAYING TIME PER APPEARANCE

# Playing time per appearance is calculated as:
#
#       Total Minutes Played
#       ---------------------
#          Number of Matches

population_df["Playing_Time_Per_Appearance"] = (
    population_df["Min"] / population_df["MP"]
)

print("\n PLAYING TIME PER APPEARANCE ")

print(
    population_df[
        [
            "Player",
            "MP",
            "Min",
            "Mn/MP",
            "Playing_Time_Per_Appearance"
        ]
    ].head(10)
)


# 6. CREATE POSITION GROUPS

# For stratified sampling, we simplify positions into three main outfield groups:
# - DF = Defender
# - MF = Midfielder
# - FW = Forward

population_df["Position_Group"] = (
    population_df["Pos"]
    .astype(str)
    .str.extract(r"^(DF|MF|FW)", expand=False)
)

print("\n POSITION GROUPS ")

print(population_df["Position_Group"].value_counts())

print(
    "\nPlayers without a position group:",
    population_df["Position_Group"].isna().sum()
)


# 7. CREATE AGE GROUPS

# Convert Age to numeric in case the CSV contains text values.
population_df["Age"] = pd.to_numeric(
    population_df["Age"],
    errors="coerce"
)

# Use the population median as the age cut-off.
# This creates two approximately balanced groups:
# - Younger: age <= median
# - Older: age > median
median_age = population_df["Age"].median()

population_df["Age_Group"] = pd.cut(
    population_df["Age"],
    bins=[
        float("-inf"),
        median_age,
        float("inf")
    ],
    labels=[
        "Younger",
        "Older"
    ],
    include_lowest=True
)

print("\n AGE GROUPS ")

print("Median age:", median_age)

print("\nPlayers by age group:")
print(population_df["Age_Group"].value_counts())

print("\nAge group × Position:")
print(
    pd.crosstab(
        population_df["Age_Group"],
        population_df["Position_Group"],
        margins=True
    )
)


# 8. CALCULATE POPULATION MEAN


population_mean = population_df[
    "Playing_Time_Per_Appearance"
].mean()

print("\n POPULATION MEAN ")

print(
    "Population mean:",
    round(population_mean, 2),
    "minutes per appearance"
)


# 9. VISUALISE AGE DISTRIBUTION

age_plot_df = population_df.dropna(
    subset=["Age", "Age_Group"]
).copy()

# Keep the same colours used in the presentation.
age_palette = {
    "Younger": "#2F80ED",
    "Older": "#081A2C"
}

sns.set_theme(
    style="whitegrid",
    context="talk"
)

fig, ax = plt.subplots(figsize=(11, 6))

sns.histplot(
    data=age_plot_df,
    x="Age",
    hue="Age_Group",
    hue_order=["Younger", "Older"],
    palette=age_palette,
    multiple="stack",
    discrete=True,
    edgecolor="white",
    linewidth=0.8,
    alpha=0.95,
    ax=ax
)

age_cutoff_line = median_age + 0.5

ax.axvline(
    age_cutoff_line,
    color="#D7B56D",
    linestyle="--",
    linewidth=2.5
)

# Add the median age label to the chart.
y_max = ax.get_ylim()[1]

ax.text(
    age_cutoff_line + 0.3,
    y_max * 0.93,
    f"Median age = {median_age:.0f}",
    color="#8A6D2F",
    fontsize=13,
    fontweight="bold"
)

ax.set_title(
    "Age Distribution of Eligible Outfield Players",
    fontsize=19,
    fontweight="bold",
    color="#223F66",
    pad=18
)

ax.set_xlabel(
    "Player Age on 11 June 2026",
    fontsize=13
)

ax.set_ylabel(
    "Number of Players",
    fontsize=13
)

# Display every second age to avoid overcrowding.
minimum_age = int(age_plot_df["Age"].min())
maximum_age = int(age_plot_df["Age"].max())

ax.set_xticks(
    range(minimum_age, maximum_age + 1, 2)
)

legend = ax.get_legend()

if legend is not None:
    legend.set_title("Age group")

# Remove unnecessary chart borders.
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Keep only horizontal gridlines.
ax.grid(axis="x", visible=False)
ax.grid(axis="y", alpha=0.25)

plt.tight_layout()

age_chart_path = (
    current_folder / "age_distribution.png"
)

plt.savefig(
    age_chart_path,
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

print("\nAge distribution chart saved to:")
print(age_chart_path)

plt.show()
plt.close()


# ==========================================================
# 10. PROPORTIONATE STRATIFIED RANDOM SAMPLING

sample_size = 100

position_counts = (
    population_df["Position_Group"]
    .value_counts()
    .sort_index()
)

position_proportions = (
    position_counts / len(population_df)
)

# Calculate how many players should be selected
# from each position group.
sample_sizes = (
    position_proportions * sample_size
).round().astype(int)

# Rounding can cause the total to differ slightly from 100.

difference = sample_size - sample_sizes.sum()

if difference != 0:
    largest_stratum = sample_sizes.idxmax()
    sample_sizes[largest_stratum] += difference

print("\n STRATIFIED SAMPLE ")

print("\nPopulation by position:")
print(position_counts)

print("\nPopulation proportions (%):")
print(
    (position_proportions * 100).round(2)
)

print("\nRequired sample size by position:")
print(sample_sizes)


# ----------------------------------------------------------
# Randomly sample players within each position group
# ----------------------------------------------------------

sample_parts = []

for position, size in sample_sizes.items():

    position_data = population_df[
        population_df["Position_Group"] == position
    ]

    position_sample = position_data.sample(
        n=size,
        random_state=140
    )

    sample_parts.append(position_sample)


# Combine all position strata into one sample.
sample_df = pd.concat(
    sample_parts,
    ignore_index=True
)

# Shuffle the final sample so that players are not grouped
# by position in the resulting DataFrame.
sample_df = (
    sample_df
    .sample(frac=1, random_state=140)
    .reset_index(drop=True)
)

print("\nFinal sample size:")
print(len(sample_df))

print("\nSample by position:")
print(sample_df["Position_Group"].value_counts())


# 11. CHECK POPULATION VS SAMPLE

population_counts = (
    population_df["Position_Group"]
    .value_counts()
    .sort_index()
)

sample_counts = (
    sample_df["Position_Group"]
    .value_counts()
    .sort_index()
)

position_comparison = pd.DataFrame({
    "Population": population_counts,
    "Sample": sample_counts
})

position_comparison.loc["Total"] = [
    position_comparison["Population"].sum(),
    position_comparison["Sample"].sum()
]

print("\n POPULATION VS SAMPLE ")
print(position_comparison)


# ----------------------------------------------------------
# Visualise population vs sample

plt.figure(figsize=(8, 4.5))

sns.heatmap(
    position_comparison,
    annot=True,
    fmt="d",
    cmap="Blues",
    linewidths=1,
    linecolor="white",
    cbar=False
)

plt.title(
    "Population vs Sample by Position",
    fontsize=16,
    fontweight="bold"
)

plt.xlabel("")
plt.ylabel("Position Group")

plt.tight_layout()

position_chart_path = (
    current_folder /
    "population_vs_sample_position.png"
)

plt.savefig(
    position_chart_path,
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

plt.show()
plt.close()


# 12. DESCRIPTIVE STATISTICS

playing_time_sample = sample_df[
    "Playing_Time_Per_Appearance"
]

sample_count = playing_time_sample.count()
sample_mean = playing_time_sample.mean()
sample_median = playing_time_sample.median()
sample_std = playing_time_sample.std()
sample_min = playing_time_sample.min()
sample_max = playing_time_sample.max()

print("\n DESCRIPTIVE STATISTICS ")

print("Sample size:", sample_count)
print("Mean:", round(sample_mean, 2))
print("Median:", round(sample_median, 2))
print("Standard deviation:", round(sample_std, 2))
print("Minimum:", round(sample_min, 2))
print("Maximum:", round(sample_max, 2))


# Descriptive statistics by age group are useful for understanding the sample before conducting the t-test.
age_group_statistics = (
    sample_df
    .groupby("Age_Group", observed=True)
    ["Playing_Time_Per_Appearance"]
    .agg(
        Count="count",
        Mean="mean",
        Median="median",
        Standard_Deviation="std",
        Minimum="min",
        Maximum="max"
    )
    .round(2)
)

print("\nDescriptive statistics by age group:")
print(age_group_statistics)


# 13. 95% CONFIDENCE INTERVAL FOR POPULATION MEAN

# The t-distribution is used because the population standard deviation is not assumed to be known when estimating the
# population mean from the sample.

confidence_level = 0.95
degrees_of_freedom = sample_count - 1

standard_error = (
    sample_std / (sample_count ** 0.5)
)

t_critical = stats.t.ppf(
    (1 + confidence_level) / 2,
    df=degrees_of_freedom
)

margin_of_error = (
    t_critical * standard_error
)

ci_lower = sample_mean - margin_of_error
ci_upper = sample_mean + margin_of_error

print("\n 95% CONFIDENCE INTERVAL ")

print("Sample mean:", round(sample_mean, 2))
print("Standard error:", round(standard_error, 2))
print("Degrees of freedom:", degrees_of_freedom)
print("Critical t-value:", round(t_critical, 3))
print("Margin of error:", round(margin_of_error, 2))

print(
    "95% confidence interval:",
    f"({ci_lower:.2f}, {ci_upper:.2f})"
)


# 14. SAMPLING DISTRIBUTION SIMULATION

# Repeatedly draw samples using the same stratified sampling
# procedure to demonstrate the sampling distribution of
# the sample mean.

number_of_samples = 5000
sampling_means = []

for _ in range(number_of_samples):

    simulated_parts = []

    for position, size in sample_sizes.items():

        position_data = population_df[
            population_df["Position_Group"] == position
        ]

        simulated_sample = position_data.sample(
            n=size,
            replace=False
        )

        simulated_parts.append(simulated_sample)

    simulated_sample = pd.concat(
        simulated_parts,
        ignore_index=True
    )

    sampling_means.append(
        simulated_sample[
            "Playing_Time_Per_Appearance"
        ].mean()
    )

sampling_means = pd.Series(sampling_means)

print("\n SAMPLING DISTRIBUTION ")

print(
    "Number of simulated samples:",
    number_of_samples
)

print(
    "Simulated sample size:",
    sample_size
)

print(
    "Mean of sampling distribution:",
    round(sampling_means.mean(), 2)
)

print(
    "SD of sampling distribution:",
    round(sampling_means.std(), 2)
)


# 15. VISUALISE SAMPLE STATISTICS

sample_statistics = pd.DataFrame({
    "Statistic": [
        "Mean",
        "Standard Deviation",
        "Standard Error"
    ],
    "Value": [
        sample_mean,
        sample_std,
        standard_error
    ]
})

fig, ax = plt.subplots(figsize=(10, 5))

bars = ax.barh(
    sample_statistics["Statistic"],
    sample_statistics["Value"]
)

# Display the exact value beside each bar.
for bar, value in zip(
    bars,
    sample_statistics["Value"]
):

    ax.text(
        bar.get_width() + 0.2,
        bar.get_y() + bar.get_height() / 2,
        f"{value:.2f}",
        va="center",
        fontsize=14,
        fontweight="bold"
    )

ax.set_title(
    "Sample Statistics",
    fontsize=20,
    fontweight="bold",
    color="#223F66",
    pad=15
)

ax.set_xlabel("Minutes", fontsize=13)
ax.set_ylabel("")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

sample_statistics_path = (
    current_folder /
    "sample_statistics.png"
)

plt.savefig(
    sample_statistics_path,
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

print("\nSample statistics chart saved to:")
print(sample_statistics_path)

plt.show()
plt.close()


# 16. VISUALISE 95% CONFIDENCE INTERVAL

fig, ax = plt.subplots(figsize=(11, 4.5))

# Draw the confidence interval.
ax.plot(
    [ci_lower, ci_upper],
    [0, 0],
    linewidth=8,
    color="#3766C3"
)

# Mark the confidence interval endpoints.
ax.scatter(
    [ci_lower, ci_upper],
    [0, 0],
    s=120,
    color="#3766C3",
    zorder=3
)

# Mark the sample mean.
ax.scatter(
    sample_mean,
    0,
    s=350,
    color="#D7B56D",
    edgecolor="white",
    linewidth=2,
    zorder=4
)

ax.axvline(
    population_mean,
    color="#223F66",
    linestyle="--",
    linewidth=2.5
)

ax.text(
    ci_lower,
    -0.12,
    f"{ci_lower:.2f}",
    ha="center",
    fontsize=14,
    fontweight="bold"
)

ax.text(
    sample_mean,
    0.12,
    f"Sample Mean\n{sample_mean:.2f}",
    ha="center",
    fontsize=13,
    fontweight="bold",
    color="#223F66"
)

ax.text(
    ci_upper,
    -0.12,
    f"{ci_upper:.2f}",
    ha="center",
    fontsize=14,
    fontweight="bold"
)

ax.text(
    population_mean,
    0.28,
    f"Population Mean\n{population_mean:.2f}",
    ha="center",
    fontsize=12,
    fontweight="bold",
    color="#223F66"
)

ax.set_title(
    "95% Confidence Interval for Mean Playing Time per Appearance",
    fontsize=19,
    fontweight="bold",
    color="#223F66",
    pad=20
)

ax.set_xlabel(
    "Minutes per appearance",
    fontsize=13
)

ax.set_yticks([])
ax.set_ylim(-0.3, 0.4)

ax.spines["left"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

plt.tight_layout()

ci_chart_path = (
    current_folder /
    "95_confidence_interval.png"
)

plt.savefig(
    ci_chart_path,
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

print("\n95% confidence interval chart saved to:")
print(ci_chart_path)

plt.show()
plt.close()


# 17. SEPARATE TWO-SAMPLE T-TEST

# This sample is intentionally created separately from
# the previous sample of 100 players.
# The t-test sample contains:
# - 50 Younger players
# - 50 Older players
#
# Within each age group, players are proportionally
# stratified by Position_Group.


t_test_sample_parts = []

group_sample_size = 50

for age_group in ["Younger", "Older"]:

    # Select only players from the current age group.
    age_data = population_df[
        population_df["Age_Group"] == age_group
    ].copy()

    # Determine the position distribution within this
    # particular age group.
    position_counts = (
        age_data["Position_Group"]
        .value_counts()
        .sort_index()
    )

    position_proportions = (
        position_counts / len(age_data)
    )

    # Allocate 50 players proportionally across positions.
    position_sample_sizes = (
        position_proportions * group_sample_size
    ).round().astype(int)

    # Ensure that the group contains exactly 50 players
    # after rounding.
    difference = (
        group_sample_size -
        position_sample_sizes.sum()
    )

    if difference != 0:

        largest_position = (
            position_sample_sizes.idxmax()
        )

        position_sample_sizes[
            largest_position
        ] += difference

    print("\n================================")
    print(age_group.upper())
    print("================================")

    print("\nPopulation by position:")
    print(position_counts)

    print("\nSample size by position:")
    print(position_sample_sizes)

    # Randomly select players within each position stratum.
    for position, n in position_sample_sizes.items():

        position_data = age_data[
            age_data["Position_Group"] == position
        ]

        selected_players = position_data.sample(
            n=n,
            random_state=140
        )

        t_test_sample_parts.append(
            selected_players
        )


# Combine both age groups.
t_test_sample = pd.concat(
    t_test_sample_parts,
    ignore_index=True
)

# Shuffle the final t-test sample.
t_test_sample = (
    t_test_sample
    .sample(frac=1, random_state=140)
    .reset_index(drop=True)
)


# 18. CHECK T-TEST SAMPLE

print("\n T-TEST SAMPLE ")

print(
    "Total observations:",
    len(t_test_sample)
)

print("\nAge group:")
print(
    t_test_sample["Age_Group"].value_counts()
)

print("\nAge group × Position:")
print(
    pd.crosstab(
        t_test_sample["Age_Group"],
        t_test_sample["Position_Group"],
        margins=True
    )
)


# 19. SPLIT INTO YOUNGER AND OLDER GROUPS

younger = t_test_sample.loc[
    t_test_sample["Age_Group"] == "Younger",
    "Playing_Time_Per_Appearance"
]

older = t_test_sample.loc[
    t_test_sample["Age_Group"] == "Older",
    "Playing_Time_Per_Appearance"
]


# 20. DESCRIPTIVE STATISTICS FOR T-TEST

younger_n = len(younger)
older_n = len(older)

younger_mean = younger.mean()
older_mean = older.mean()

younger_std = younger.std()
older_std = older.std()

print("\n T-TEST DESCRIPTIVE STATISTICS ")

print("\n===== T-TEST DESCRIPTIVE STATISTICS =====")

print("\nYounger players")
print("n =", younger_n)
print(
    "Mean playing time per appearance =",
    round(younger_mean, 2),
    "minutes"
)
print(
    "Standard deviation =",
    round(younger_std, 2),
    "minutes"
)

print("\nOlder players")
print("n =", older_n)
print(
    "Mean playing time per appearance =",
    round(older_mean, 2),
    "minutes"
)
print(
    "Standard deviation =",
    round(older_std, 2),
    "minutes"
)


# 21. HYPOTHESES

print("\n HYPOTHESES ")

print(
    "H0: There is no difference in mean playing time "
    "per appearance between younger and older players."
)

print(
    "H1: There is a difference in mean playing time "
    "per appearance between younger and older players."
)


# 22. TWO-SAMPLE T-TEST

# equal_var=False means we do not assume that the two
# age groups have equal population variances.
t_test_result = stats.ttest_ind(
    younger,
    older,
    equal_var=False
)

t_statistic = t_test_result.statistic
p_value = t_test_result.pvalue

# Positive value means Younger - Older is positive.
# Negative value means Younger - Older is negative.
mean_difference = (
    younger_mean - older_mean
)

print("\n T-TEST RESULTS ")

print(
    "Mean difference (Younger - Older):",
    round(mean_difference, 2)
)

print(
    "t-statistic:",
    round(t_statistic, 3)
)

print(
    "p-value:",
    round(p_value, 4)
)


# Decision at the 5% significance level.
if p_value < 0.05:

    print("\nDecision: Reject H0")

    print(
        "There is sufficient statistical evidence "
        "of a difference in mean playing time "
        "between the two age groups."
    )

else:

    print("\nDecision: Fail to reject H0")

    print(
        "There is insufficient statistical evidence "
        "of a difference in mean playing time "
        "between the two age groups."
    )


# 23. 95% CI FOR EACH AGE-GROUP MEAN

# Calculate the standard error for each age group.
younger_se = (
    younger_std / (younger_n ** 0.5)
)

older_se = (
    older_std / (older_n ** 0.5)
)

# Critical t-values for 95% confidence intervals.
younger_t_critical = stats.t.ppf(
    0.975,
    df=younger_n - 1
)

older_t_critical = stats.t.ppf(
    0.975,
    df=older_n - 1
)

# Margin of error for each group.
younger_margin_error = (
    younger_t_critical * younger_se
)

older_margin_error = (
    older_t_critical * older_se
)

# Confidence interval for Younger players.
younger_ci_lower = (
    younger_mean - younger_margin_error
)

younger_ci_upper = (
    younger_mean + younger_margin_error
)

# Confidence interval for Older players.
older_ci_lower = (
    older_mean - older_margin_error
)

older_ci_upper = (
    older_mean + older_margin_error
)

print("\n 95% CONFIDENCE INTERVALS ")

print(
    "Younger:",
    f"({younger_ci_lower:.2f}, "
    f"{younger_ci_upper:.2f})"
)

print(
    "Older:",
    f"({older_ci_lower:.2f}, "
    f"{older_ci_upper:.2f})"
)


# 24. VISUALISE TWO-SAMPLE T-TEST

fig, ax = plt.subplots(figsize=(11, 6))

groups = [
    "Younger",
    "Older"
]

means = [
    younger_mean,
    older_mean
]

errors = [
    younger_margin_error,
    older_margin_error
]

bars = ax.bar(
    groups,
    means,
    yerr=errors,
    capsize=8,
    width=0.55,
    error_kw={
        "elinewidth": 2,
        "capthick": 2
    }
)


# Add mean values beside the bars.
for bar, mean in zip(bars, means):

    ax.text(
        bar.get_x() + bar.get_width() + 0.04,
        bar.get_height(),
        f"{mean:.2f}",
        ha="left",
        va="center",
        fontsize=14,
        fontweight="bold",
        color="#223F66"
    )


# Display the t-statistic and p-value on the chart.
ax.text(
    0.98,
    0.95,
    f"t = {t_statistic:.3f}\n"
    f"p = {p_value:.4f}",
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=13,
    fontweight="bold",
    color="#223F66"
)

ax.set_title(
    "Mean Playing Time per Appearance by Age Group",
    fontsize=20,
    fontweight="bold",
    color="#223F66",
    pad=18
)

ax.set_xlabel(
    "Age group",
    fontsize=13
)

ax.set_ylabel(
    "Minutes per appearance",
    fontsize=13
)

# Leave enough space above the error bars and labels.
ax.set_ylim(
    0,
    max(
        younger_mean + younger_margin_error,
        older_mean + older_margin_error
    ) + 10
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.grid(
    axis="y",
    alpha=0.25
)

ax.grid(
    axis="x",
    visible=False
)

plt.tight_layout()


# 25. SAVE T-TEST CHART

t_test_chart_path = (
    current_folder /
    "mean_playing_time_by_age_group_t_test.png"
)

plt.savefig(
    t_test_chart_path,
    dpi=300,
    bbox_inches="tight",
    facecolor="white"
)

print("\nT-test chart saved to:")
print(t_test_chart_path)

plt.show()
plt.close()