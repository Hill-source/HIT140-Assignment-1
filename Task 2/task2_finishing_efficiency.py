import math
import statistics as stats

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st


# 0. Settings

data_file = (
    r"C:\Users\Ly Trung Hao\OneDrive - Charles Darwin University\Units"
    r"\Foundation of data science\Assigment 2\Claude"
    r"\fifa_attacking_distribution.xlsx"
)

output_folder = (
    r"C:\Users\Ly Trung Hao\OneDrive - Charles Darwin University\Units"
    r"\Foundation of data science\Assigment 2\Claude\outputs"
)

attempts_column = "Attacking - Attempts At Goal"
conversion_column = "Attacking - Attempts At Goal Conv. Rate (%)"

opening_date = pd.Timestamp("2026-06-11")  # tournament opening date
minimum_attempts = 5      # Task 2 eligibility: at least five attempts at goal
main_sample_size = 60     # main-question sample (>= 30 required by the brief)
group_sample_size = 30    # each age-group sample for the t-test (>= 30 required)
random_seed = 42          # fixed seed so the random sample is reproducible
positions = ["Defender", "Midfielder", "Forward"]


# -----------------------------------------------------------------------------
# 1. Read the data
# -----------------------------------------------------------------------------
data = pd.read_excel(data_file)
print("Players in the dataset:", len(data))


# -----------------------------------------------------------------------------
# 2. Clean the position column 
# -----------------------------------------------------------------------------
# Put the codes into one case before anything is counted or filtered, then
# replace them with readable names.
data["Position"] = data["Position"].str.strip().str.upper()
data["Position"] = data["Position"].replace({
    "GK": "Goalkeeper",
    "DF": "Defender",
    "MF": "Midfielder",
    "FW": "Forward",
})
print("Positions:", data["Position"].value_counts().to_dict())


# -----------------------------------------------------------------------------
# 3. Feature engineering: age and age group from Birth Date
# -----------------------------------------------------------------------------
birth_date = pd.to_datetime(data["Birth Date"])

# Completed years on the tournament opening date: subtract the birth year, then
# take one year off if the player's birthday falls after 11 June.
birthday_not_yet = (
    (birth_date.dt.month > opening_date.month)
    | ((birth_date.dt.month == opening_date.month)
       & (birth_date.dt.day > opening_date.day))
)
data["Age"] = opening_date.year - birth_date.dt.year - birthday_not_yet.astype(int)

# The cut-off is the median age of the whole player population.
age_cut_off = stats.median(data["Age"].tolist())
print("Median age of the player population:", age_cut_off)

# Bin the ages into two groups (discretisation, Week 5).
min_age = data["Age"].min()
max_age = data["Age"].max()
data["Age Group"] = pd.cut(
    data["Age"],
    bins=[min_age - 1, age_cut_off, max_age],
    labels=["Younger", "Older"],
)
print("All players by age group:", data["Age Group"].value_counts().to_dict())


# -----------------------------------------------------------------------------
# 4. Base population and eligible population
# -----------------------------------------------------------------------------
# Base population (the group's shared rule for all four tasks):
# outfield players who made at least one appearance. A player with recorded
# minutes must have been on the pitch, so minutes played is used as the
# appearance test.
base_population = data[
    (data["Position"] != "Goalkeeper")
    & (data["Minutes Played"] > 0)
].copy()
print("Base population (non-GK, played at least once):", len(base_population))
print("  by age group:",
      base_population["Age Group"].value_counts().to_dict())

# Note on missing data: the only rows without a minutes value are goalkeepers,
# so the non-goalkeeper rule removes them anyway.
print("  rows with no minutes value:", data["Minutes Played"].isna().sum(),
      "(all goalkeepers)")

# Task 2 eligible population: base players with at least five attempts at goal.
eligible = base_population[
    base_population[attempts_column] >= minimum_attempts
].copy()

# The response variable: FIFA's conversion rate (percentage of attempts scored).
eligible["Conversion Rate"] = eligible[conversion_column]

# Data quality check: a conversion rate cannot exceed 100%.
impossible = data[data[conversion_column] > 100]
print("Rows with a conversion rate above 100%%: %d (eligible ones: %d)"
      % (len(impossible), len(eligible[eligible["Conversion Rate"] > 100])))


# -----------------------------------------------------------------------------
# 5. Proportionate stratified random sampling
# -----------------------------------------------------------------------------
# Helper: take one proportionate stratified sample (by playing position) from a
# given population. The number drawn from each position keeps the same
# proportion that position has in that population.
def stratified_sample(population, total_size, seed):
    pieces = []
    allocation_rows = []
    for position in positions:
        group = population[population["Position"] == position]
        share = len(group) / len(population)
        exact_size = share * total_size
        draw = round(exact_size)
        pieces.append(group.sample(n=draw, random_state=seed))
        allocation_rows.append(
            {
                "Position": position,
                "Population Size": len(group),
                "Share (%)": share * 100,
                "Exact Sample Size": exact_size,
                "Sample Size": draw,
            }
        )
    sample = pd.concat(pieces, ignore_index=True)
    allocation = pd.DataFrame(allocation_rows)
    return sample, allocation


# Sample 1 - answers the main analytic question (all eligible players).
main_sample, main_allocation = stratified_sample(
    eligible, main_sample_size, random_seed
)

# Sample 2 - younger eligible players, for the t-test.
younger_population = eligible[eligible["Age Group"] == "Younger"]
younger_sample, younger_allocation = stratified_sample(
    younger_population, group_sample_size, random_seed
)

# Sample 3 - older eligible players, for the t-test.
older_population = eligible[eligible["Age Group"] == "Older"]
older_sample, older_allocation = stratified_sample(
    older_population, group_sample_size, random_seed
)


# -----------------------------------------------------------------------------
# 6. Main question - descriptive statistics and 95% confidence interval
# -----------------------------------------------------------------------------
main_rates = main_sample["Conversion Rate"].tolist()

main_mean = stats.mean(main_rates)
main_median = stats.median(main_rates)
main_std = np.array(main_rates).std(ddof=1)
main_q1 = np.percentile(main_rates, 25)
main_q3 = np.percentile(main_rates, 75)

# 95% confidence interval of the mean (z-based method, Week 3).
z_score = st.norm.ppf(q=0.975)
standard_error = main_std / math.sqrt(len(main_rates))
margin_of_error = z_score * standard_error
ci_low = main_mean - margin_of_error
ci_high = main_mean + margin_of_error


# -----------------------------------------------------------------------------
# 7. Two-sample t-test - younger vs older conversion rate
# -----------------------------------------------------------------------------
younger_rates = younger_sample["Conversion Rate"].tolist()
older_rates = older_sample["Conversion Rate"].tolist()

younger_mean = stats.mean(younger_rates)
older_mean = stats.mean(older_rates)
younger_std = np.array(younger_rates).std(ddof=1)
older_std = np.array(older_rates).std(ddof=1)

# H0: younger mean = older mean.  H1: younger mean != older mean (two-sided).
# equal_var=False -> Welch's t-test (we do not assume equal variances).
t_statistic, p_value = st.ttest_ind_from_stats(
    younger_mean, younger_std, len(younger_rates),
    older_mean, older_std, len(older_rates),
    equal_var=False,
    alternative="two-sided",
)


# -----------------------------------------------------------------------------
# 8. Print the results
# -----------------------------------------------------------------------------
print()
print("TASK 2 - FINISHING EFFICIENCY")
print("=" * 58)
print("Eligible population (>=5 attempts) :", len(eligible))
print("  Younger eligible:", len(younger_population))
print("  Older eligible  :", len(older_population))
print()

print("SAMPLE 1 - MAIN QUESTION (stratified by position)")
print(main_allocation.to_string(index=False))
print("  Sample size        : %d" % len(main_sample))
print("  Mean conversion    : %.2f%%" % main_mean)
print("  Median conversion  : %.2f%%" % main_median)
print("  Std deviation      : %.2f percentage points" % main_std)
print("  Q1 / Q3            : %.2f%% / %.2f%%" % (main_q1, main_q3))
print("  95%% confidence interval for the mean: %.2f%% to %.2f%%"
      % (ci_low, ci_high))
print()

print("SAMPLE 2 & 3 - t-TEST (younger vs older)")
print("  Younger: n=%d  mean=%.2f%%  std=%.2f"
      % (len(younger_rates), younger_mean, younger_std))
print("  Older  : n=%d  mean=%.2f%%  std=%.2f"
      % (len(older_rates), older_mean, older_std))
print("  Difference (younger - older): %.2f percentage points"
      % (younger_mean - older_mean))
print("  t-statistic: %.3f" % t_statistic)
print("  p-value    : %.3f" % p_value)
if p_value < 0.05:
    print("  Decision   : reject H0 (means are different).")
else:
    print("  Decision   : do not reject H0 (no significant difference).")


# -----------------------------------------------------------------------------
# 9. Save the samples and the allocation table
# -----------------------------------------------------------------------------
keep_columns = [
    "Player ID", "Player", "Nation", "Birth Date", "Position", "Age",
    "Age Group", "Minutes Played", attempts_column, "Conversion Rate",
]

main_sample[keep_columns].to_csv(
    output_folder + r"\task2_main_sample.csv", index=False)
younger_sample[keep_columns].to_csv(
    output_folder + r"\task2_younger_sample.csv", index=False)
older_sample[keep_columns].to_csv(
    output_folder + r"\task2_older_sample.csv", index=False)
main_allocation.to_csv(
    output_folder + r"\task2_allocation.csv", index=False)


# -----------------------------------------------------------------------------
# 10. Figure 1 - histogram of the main-sample conversion rates
# -----------------------------------------------------------------------------
# Bin width is set first and the bin count derived from the range, the same
# way as the Week 2 histogram exercise.
sample = np.array(main_rates)
max_val = sample.max()
min_val = sample.min()
the_range = max_val - min_val
bin_width = 5
bin_count = int(the_range / bin_width)
print()
print("Histogram: range %.0f to %.0f, bin width %d, %d bins"
      % (min_val, max_val, bin_width, bin_count))

plt.figure(figsize=(7.4, 5))
plt.hist(sample, color="blue", edgecolor="black", bins=bin_count)
plt.title("Histogram of Attempt-at-Goal Conversion Rates")
plt.xlabel("Conversion Rate (%)")
plt.ylabel("Players")
plt.tight_layout()
plt.savefig(output_folder + r"\fig_conversion_hist.png", dpi=200)
plt.close()


# -----------------------------------------------------------------------------
# 11. Figure 2 - younger vs older conversion-rate distributions
# -----------------------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.hist(younger_rates, bins=8, alpha=0.6, color="#4C78A8",
         edgecolor="black", label="Younger (n=%d)" % len(younger_rates))
plt.hist(older_rates, bins=8, alpha=0.6, color="#12324A",
         edgecolor="black", label="Older (n=%d)" % len(older_rates))
plt.title("Conversion Rate: Younger vs Older Players")
plt.xlabel("Conversion Rate (%)")
plt.ylabel("Number of Players")
plt.legend()
plt.tight_layout()
plt.savefig(output_folder + r"\fig_age_hist.png", dpi=200)
plt.close()


# -----------------------------------------------------------------------------
# 12. Figure 3 - mean conversion rate by age group with 95% confidence intervals
# -----------------------------------------------------------------------------
younger_error = z_score * younger_std / math.sqrt(len(younger_rates))
older_error = z_score * older_std / math.sqrt(len(older_rates))

plt.figure(figsize=(7.2, 5))
plt.bar(
    ["Younger\n(%d or under)" % age_cut_off, "Older\n(over %d)" % age_cut_off],
    [younger_mean, older_mean],
    yerr=[younger_error, older_error],
    capsize=10,
    color=["#4C78A8", "#12324A"],
    edgecolor="black",
    width=0.55,
)
plt.text(0, younger_mean + younger_error + 0.6, "%.1f%%" % younger_mean,
         ha="center", fontsize=12, fontweight="bold")
plt.text(1, older_mean + older_error + 0.6, "%.1f%%" % older_mean,
         ha="center", fontsize=12, fontweight="bold")
plt.ylabel("Mean Conversion Rate (%)")
plt.title("Mean Conversion Rate by Age Group\n"
          "(bars show 95% confidence interval)")
plt.ylim(0, max(younger_mean + younger_error, older_mean + older_error) + 4)
plt.figtext(0.5, 0.01, "t* = %.3f   p = %.3f" % (t_statistic, p_value),
            ha="center", fontsize=10, color="#444444")
plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig(output_folder + r"\fig_age_comparison.png", dpi=200)
plt.close()


# -----------------------------------------------------------------------------
# 13. Figure 4 - age of the base population, split at the cut-off
# -----------------------------------------------------------------------------
base_younger = base_population[
    base_population["Age Group"] == "Younger"]["Age"].tolist()
base_older = base_population[
    base_population["Age Group"] == "Older"]["Age"].tolist()

base_edges = range(int(base_population["Age"].min()),
                   int(base_population["Age"].max()) + 2)

plt.figure(figsize=(8, 4.6))
plt.hist(base_younger, bins=base_edges, color="#4C78A8", edgecolor="white",
         label="Younger (%d or under)" % age_cut_off)
plt.hist(base_older, bins=base_edges, color="#12324A", edgecolor="white",
         label="Older (over %d)" % age_cut_off)
plt.axvline(age_cut_off + 1, color="#7FA6CC", linestyle="--", linewidth=1.5)
plt.text(age_cut_off + 1.3, plt.ylim()[1] * 0.9,
         "Median age = %d" % age_cut_off, fontsize=9, color="#4C78A8")
plt.title("Age of the Base Population (outfield players who played)")
plt.xlabel("Player Age on 11 June 2026")
plt.ylabel("Number of Players")
plt.legend(title="Age group")
plt.tight_layout()
plt.savefig(output_folder + r"\fig_base_age_distribution.png", dpi=200)
plt.close()


# -----------------------------------------------------------------------------
# 14. Figure 5 - age of the Task 2 eligible players
# -----------------------------------------------------------------------------
elig_younger = eligible[eligible["Age Group"] == "Younger"]["Age"].tolist()
elig_older = eligible[eligible["Age Group"] == "Older"]["Age"].tolist()

elig_edges = range(int(eligible["Age"].min()), int(eligible["Age"].max()) + 2)

plt.figure(figsize=(8, 4.6))
plt.hist(elig_younger, bins=elig_edges, color="#4C78A8", edgecolor="white",
         label="Younger (%d or under)" % age_cut_off)
plt.hist(elig_older, bins=elig_edges, color="#12324A", edgecolor="white",
         label="Older (over %d)" % age_cut_off)
plt.axvline(age_cut_off + 1, color="#7FA6CC", linestyle="--", linewidth=1.5)
plt.text(age_cut_off + 1.3, plt.ylim()[1] * 0.9,
         "Median age = %d" % age_cut_off, fontsize=9, color="#4C78A8")
plt.title("Age of the Task 2 Eligible Players (5+ attempts at goal)")
plt.xlabel("Player Age on 11 June 2026")
plt.ylabel("Number of Players")
plt.legend(title="Age group")
plt.tight_layout()
plt.savefig(output_folder + r"\fig_age_distribution.png", dpi=200)
plt.close()

# -----------------------------------------------------------------------------
# 15. Figure 6 - the sample keeps the position mix of the eligible population
# -----------------------------------------------------------------------------
mix_population = []
mix_sample = []
for position in positions:
    mix_population.append(
        len(eligible[eligible["Position"] == position]) / len(eligible) * 100)
    mix_sample.append(
        len(main_sample[main_sample["Position"] == position])
        / len(main_sample) * 100)

shades = ["#AFC4D9", "#7FA6CC", "#12324A"]

plt.figure(figsize=(8, 2.6))
left_population = 0
left_sample = 0
for i in range(len(positions)):
    plt.barh("Eligible population\n(174 players)", mix_population[i],
             left=left_population, color=shades[i], edgecolor="white",
             label=positions[i])
    plt.text(left_population + mix_population[i] / 2, 0,
             "%.1f%%" % mix_population[i], ha="center", va="center",
             fontsize=10, color="white", fontweight="bold")
    left_population += mix_population[i]

    plt.barh("My sample\n(60 players)", mix_sample[i], left=left_sample,
             color=shades[i], edgecolor="white")
    plt.text(left_sample + mix_sample[i] / 2, 1,
             "%.1f%%" % mix_sample[i], ha="center", va="center",
             fontsize=10, color="white", fontweight="bold")
    left_sample += mix_sample[i]

plt.xlim(0, 100)
plt.xlabel("Share of players (%)")
plt.title("The sample keeps the same position mix as the population")
plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.42), ncol=3,
           frameon=False, fontsize=10)
plt.tight_layout()
plt.savefig(output_folder + r"\fig_position_mix.png", dpi=200,
            bbox_inches="tight")
plt.close()

print()
print("Position mix  population -> sample:")
for i in range(len(positions)):
    print("  %-11s %5.1f%%  ->  %5.1f%%"
          % (positions[i], mix_population[i], mix_sample[i]))
print()
print("Saved samples, allocation table and figures to:")
print(output_folder)
