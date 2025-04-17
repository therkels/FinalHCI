import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.cm as cm

data = pd.read_csv("results.csv")

analyze_metrics = ['usabilityScore','adoptionScore','accuracyScore','errorCount']
for metric in analyze_metrics:
    grouped_data = data.groupby('design')[metric].apply(list)

    groups = grouped_data.to_dict()
    print(groups)

    colors = cm.get_cmap('Set1', len(groups))

    # Generate histograms for each group
    for i, (design, metric_val) in enumerate(groups.items()):
        color = colors(i)  # Get a unique color for each group
        plt.hist(metric_val, bins=10, alpha=0.7, color=color, label=f'Design {design}')
        # Calculate and plot the median for each group
        mean = pd.Series(metric_val).mean()
        plt.axvline(mean, color=color, linestyle='dashed', linewidth=1, label=f'Mean {design}')

    # Add labels and legend
    plt.xlabel(f'{metric} (1=worst, 5=best)')
    plt.ylabel('Score Frequency')
    plt.title(f'Response {metric} by Design')
    plt.legend()

    # Show the plot
    plt.savefig(f'{metric}_plot.png', transparent=True)
    plt.clf()

    # Perform a one-way ANOVA test
    anova_result = stats.friedmanchisquare(*groups.values())
    # Print the ANOVA test result
    print(f"ANOVA Result ({metric}):", anova_result)
