import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import rc

plt.style.use('seaborn-v0_8-whitegrid')
rc('text', usetex=True)
pd.plotting.register_matplotlib_converters()
plt.style.use("seaborn-v0_8-ticks")

def load_dataset():
    df = pd.read_csv('../data_gen/simulation_results.csv')
    df = df[df['m'] <= 367]
    df['m/n'] = df['m']/df['n']
    df['c'] = df['m'] * df['p'] // 100
    df['c/n'] = df['c'] / df['n']
    df['n/c'] = df['n'] / df['c']
    df['avg_utility'] = df['utility'] / df['n']
    df['avg_utility_greedy'] = df['utility_greedy'] / df['n']
    df['avg_matching_utility'] = df['matching_utility'] / df['n']
    df['avg_matching_utility_greedy'] = df['matching_utility_greedy'] / df['n']
    return df

def plot_dist_type(col="dist_type"):
    sns.set_theme(style="whitegrid")
    results = load_dataset()

    g = sns.relplot(
        data=results,
        x="c", y="matching_utility", col=col, hue=col,
        kind="line", palette="crest", linewidth=4, zorder=5,
        col_wrap=2, height=4, aspect=1.5, legend=False, errorbar=None
    )

    # Iterate over each subplot to customize further
    for dist_type, ax in g.axes_dict.items():
        # Add the title as an annotation within the plot
        ax.text(.01, 1.01, dist_type, transform=ax.transAxes, fontweight="bold", color='#164773', fontsize=14)
        sns.lineplot(x="c", y="matching_utility_greedy",
                     data=results[results['dist_type']=='uniform'],
                     ax=ax, color=".5", linewidth=2)

    # Tweak the supporting aspects of the plot
    g.set_titles("")
    g.set_axis_labels("Number of Access Points with Cloudlets(c)", "Utility of the Matching")
    g.tight_layout()

    plt.savefig(f'plot_matching_utility.pdf', dpi=300)
    plt.show()
    plt.close()

def plot_c_by_n_type(col="dist_type"):
    sns.set_theme(style="whitegrid")
    results = load_dataset()

    g = sns.relplot(
        data=results,
        x="c/n", y="pct_above_matching", col=col, hue=col,
        kind="scatter", palette="magma", linewidth=0.5, zorder=5, size="n",
        col_wrap=2, height=4, aspect=1.5, legend=False, sizes=(5, 200),
        alpha=0.9
    )

    # Iterate over each subplot to customize further
    for dist_type, ax in g.axes_dict.items():
        # Add the title as an annotation within the plot
        ax.text(.01, 1.01, dist_type, transform=ax.transAxes, fontweight="bold", color='#164773', fontsize=14)
        sns.scatterplot(x="c/n", y="pct_above_matching_greedy",
                     data=results[results['dist_type']==dist_type], ax=ax, color=".5",
                        linewidth=0.5, size="n", sizes=(5, 200),
                        alpha=0.5)

    # Tweak the supporting aspects of the plot
    g.set_titles("")
    g.set_axis_labels("c/n", "\% of users missing maximum delay tolerance ($T_i$)")
    g.tight_layout()

    plt.savefig(f'plot_c_by_n_type.pdf', dpi=300)
    plt.show()
    plt.close()

def plot_utility_n_dist_type(col="dist_type"):
    sns.set_theme(style="dark")
    results = load_dataset()
    #results = results[results['n']>=6250]
    g = sns.relplot(
        data=results,
        x="n/c", y="matching_utility", col=col, hue=col,
        kind="scatter", palette="crest", linewidth=0.5, zorder=5, size="n",
        col_wrap=3, height=4, aspect=1.5, legend=False,  sizes=(2, 50),
        alpha=0.5
    )

    # Iterate over each subplot to customize further
    for dist_type, ax in g.axes_dict.items():
        # Add the title as an annotation within the plot
        ax.text(.01, 1.01, dist_type, transform=ax.transAxes, fontweight="bold", color='#164773', fontsize='large')
        sns.scatterplot(x="n/c", y="matching_utility_greedy",
                     data=results, ax=ax, color=".3",
                        linewidth=0.5, size="n", sizes=(2, 50),
                        alpha=0.5)

    # Tweak the supporting aspects of the plot
    g.set_titles("")
    g.set_axis_labels("Number Users(n)/ Number of cloudlet(c)", "Utility of the Matching")
    g.tight_layout()

    plt.savefig(f'plot_utility_n_dist_type.pdf', dpi=300)
    plt.show()
    plt.close()

def plot_max_miss_type(col="dist_type"):
    sns.set_theme(style="whitegrid")
    results = load_dataset()
    results = results[(results['n']>20000) & (results['n']<60000) & (results['c']>85)]

    g = sns.relplot(
        data=results,
        x="u_sample", y="matching_utility", col='c', hue=col, row=col,
        kind="line", palette="magma", linewidth=5, zorder=5,
        height=4, aspect=1.5, legend=False,
        alpha=0.9
    )

    # Iterate over each subplot to customize further
    for itms, ax in g.axes_dict.items():
        dist_type = itms[0]
        c = itms[1]
        # Add the title as an annotation within the plot
        ax.text(.01, 1.01, f'{dist_type}, $c$ = {c}', transform=ax.transAxes, fontweight="bold", color='#164773', fontsize=22)

    # Tweak the supporting aspects of the plot
    g.set_titles("")
    g.set_axis_labels("$u_{sample}$", "Utility of the Matching")
    g.tight_layout()

    plt.savefig(f'plot_max_miss_type.pdf', dpi=300)
    plt.show()
    plt.close()

def plot_pct_above(col="dist_type"):
    sns.set_theme(style="dark")
    results = load_dataset()

    g = sns.relplot(
        data=results,
        x="n", y="pct_above", col=col, hue=col,
        kind="line", palette="crest", linewidth=4, zorder=5,
        col_wrap=3, height=4, aspect=1.5, legend=False,
    )

    # Iterate over each subplot to customize further
    for dist_type, ax in g.axes_dict.items():
        # Add the title as an annotation within the plot
        ax.text(.01, 1.01, dist_type, transform=ax.transAxes, fontweight="bold", color='#164773', fontsize='large')
        sns.lineplot(x="n", y="pct_above_greedy",
                     data=results, ax=ax, color=".5", linewidth=1)

    # Tweak the supporting aspects of the plot
    g.set_titles("")
    g.set_axis_labels("Number of Users(n)", "\% of users missing SLA")
    g.tight_layout()

    plt.savefig(f'plot_pct_above_matching.pdf', dpi=300)
    plt.show()
    plt.close()


plot_dist_type()
plot_c_by_n_type()
plot_utility_n_dist_type()
plot_max_miss_type()
plot_pct_above()