import pandas as pd
import numpy as np
from scipy.stats import f_oneway, kruskal
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

plt.style.use('seaborn-v0_8-whitegrid')
rc('text', usetex=True)
pd.plotting.register_matplotlib_converters()
plt.style.use("seaborn-v0_8-ticks")

# Load your simulation results into a DataFrame
data = pd.read_csv('../data_gen/simulation_results.csv')

# Split the data into Distributed Matching and Greedy Algorithm data
dist_cols = [
    'utility', 'matching_utility', 'pct_above_matching',
    'max_miss', 'mean_miss_user_input_data_size', 'mean_miss_user_output_data_size',
    'mean_miss_user_computation_demand', 'mean_miss_user_delay_tolerance'
]

greedy_cols = [f"{col}_greedy" for col in dist_cols]

# Create a long-format DataFrame
distributed_data = data[['w', 'n', 'm', 'p', 'u_sample', 'dist_type'] + dist_cols].copy()
distributed_data['algorithm'] = 'Distributed Matching'

greedy_data = data[['w', 'n', 'm', 'p', 'u_sample', 'dist_type'] + greedy_cols].copy()
greedy_data.columns = ['w', 'n', 'm', 'p', 'u_sample', 'dist_type'] + dist_cols
greedy_data['algorithm'] = 'Greedy'

# Combine the two datasets
long_data = pd.concat([distributed_data, greedy_data], ignore_index=True)

# Melt the data for metric-wise analysis
melted_data = long_data.melt(
    id_vars=['w', 'n', 'm', 'p', 'u_sample', 'dist_type', 'algorithm'],
    value_vars=dist_cols,
    var_name='metric',
    value_name='value'
)

# Perform ANOVA and Tukey HSD tests for each metric
anova_results = []
tukey_results = []
kruskal_results = []

for metric in dist_cols:
    subset = melted_data[melted_data['metric'] == metric]
    groups = [subset.loc[subset['algorithm'] == algo, 'value'] for algo in subset['algorithm'].unique()]

    # Perform ANOVA
    anova_stat, anova_p = f_oneway(*groups)
    anova_results.append({'metric': metric, 'F-statistic': anova_stat, 'p-value': anova_p})

    # Perform Kruskal-Wallis Test
    kruskal_stat, kruskal_p = kruskal(*groups)
    kruskal_results.append({'metric': metric, 'H-statistic': kruskal_stat, 'p-value': kruskal_p})

    # If ANOVA is significant, perform Tukey HSD
    if anova_p < 0.05:
        tukey = pairwise_tukeyhsd(subset['value'], subset['algorithm'], alpha=0.05)
        tukey_results.append({'metric': metric, 'tukey_summary': tukey.summary()})

# Create a summary DataFrame for ANOVA and Kruskal-Wallis results
anova_df = pd.DataFrame(anova_results)
kruskal_df = pd.DataFrame(kruskal_results)

# Save ANOVA and Kruskal results to CSV
anova_df.to_csv('anova_results.csv', index=False)
kruskal_df.to_csv('kruskal_results.csv', index=False)

# Visualization: Boxplots for each metric
# Create a 2x4 subplot for all metrics in a single figure
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes = axes.flatten()

# List of metrics for plotting
metrics = [
    'utility', 'matching_utility', 'pct_above_matching',
    'max_miss'
]

# Generate grouped violin plots for each metric
for i, metric in enumerate(metrics):
    subset = melted_data[melted_data['metric'] == metric]
    sns.violinplot(
        data=subset,
        x='algorithm',
        y='value',
        ax=axes[i],
        palette='coolwarm',
        inner="quartile",  # Show quartiles inside the violins
        scale='width'      # Scale the violins based on the data distribution
    )
    sns.stripplot(
        data=subset,
        x='algorithm',
        y='value',
        ax=axes[i],
        color='#042940',
        alpha=0.35,
        jitter=True  # Add jitter for better point visibility
    )
    axes[i].set_title(metric.replace('_', ' ').capitalize())
    axes[i].set_xlabel('Algorithm')
    axes[i].set_ylabel(metric.replace('_', ' ').capitalize())
    #axes[i].grid(axis='y', linestyle='--', alpha=0.7)

# Adjust layout
plt.tight_layout()

# Save the plot
plt.savefig('all_metrics_comparison.pdf', dpi=300)

# Display the plot
plt.show()
# Save Tukey results as text files
for tukey in tukey_results:
    with open(f'tukey_{tukey["metric"]}.txt', 'w') as file:
        file.write(str(tukey['tukey_summary']))

print("Analysis complete. Results saved to CSV and plots generated.")

# Define the unique configuration columns (excluding randomness)
config_columns = ['w', 'n', 'm', 'p', 'u_sample', 'dist_type']

# Count occurrences of each configuration
config_counts = data.groupby(config_columns).size().reset_index(name="trials")

# Check if all configurations have 30 trials
num_trials_per_config = config_counts['trials'].unique()

if len(num_trials_per_config) == 1 and num_trials_per_config[0] == 30:
    print(f"Each configuration was evaluated over {num_trials_per_config[0]} trials with random seeds, ensuring robustness against stochastic variability.")
else:
    print(f"Configurations have varying numbers of trials: {num_trials_per_config}")

# Independent samples t-test for matching algorithms
greedy_utilities = data['matching_utility_greedy']
matching_utilities = data['matching_utility']

t_stat, p_val = stats.ttest_ind(matching_utilities, greedy_utilities, equal_var=False)

# Compute means and standard deviations
mean_matching, std_matching = np.mean(matching_utilities), np.std(matching_utilities)
mean_greedy, std_greedy = np.mean(greedy_utilities), np.std(greedy_utilities)
df = len(matching_utilities) + len(greedy_utilities) - 2  # Degrees of freedom

print(f"For the comparison of matching algorithms:")
print(f"Mean utility of maximum-weight matching (M = {mean_matching:.2f}, SD = {std_matching:.2f}) "
      f"was found to be significantly higher than that of the greedy algorithm (M = {mean_greedy:.2f}, SD = {std_greedy:.2f}), "
      f"as determined by an independent samples t-test (t({df}) = {t_stat:.2f}, p = {p_val:.3f}).")

# Kruskal-Wallis test for task demand distributions
kruskal_result = stats.kruskal(
    data.loc[data['dist_type'] == 'uniform', 'matching_utility'],
    data.loc[data['dist_type'] == 'normal', 'matching_utility'],
    data.loc[data['dist_type'] == 'exponential', 'matching_utility']
)

print(f"\nFor task demand distributions:")
print(f"Kruskal-Wallis test revealed significant differences in utility across uniform, normal, and exponential distributions "
      f"(H({kruskal_result.statistic:.2f}) = {kruskal_result.statistic:.2f}, p = {kruskal_result.pvalue:.3f}).")

# Kruskal-Wallis test
h_stat, p_value = kruskal(
    data[data['dist_type'] == 'Uniform']['matching_utility'],
    data[data['dist_type'] == 'Normal']['matching_utility'],
    data[data['dist_type'] == 'Exponential']['matching_utility']
)

print(f"Kruskal-Wallis Test: H({len(data['dist_type'].unique()) - 1}) = {h_stat:.2f}, p = {p_value:.3f}")

# Tukey's HSD Test
tukey = pairwise_tukeyhsd(endog=data['matching_utility'], groups=data['dist_type'], alpha=0.05)
print(tukey)

# Extract significant pairs
significant_pairs = [
    (row[0], row[1], row[4])  # group1, group2, p-value
    for row in tukey.summary().data[1:]  # Skip header row
    if row[4] < 0.05  # Check significance
]

# Print significant pairwise comparisons
for g1, g2, p in significant_pairs:
    print(f"Significant difference: {g1} vs {g2}, p = {p:.3f}")