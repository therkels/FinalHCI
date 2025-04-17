import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.cm as cm

data = pd.read_csv("results.csv")


grouped_data = data.groupby('design')['timeTaken'].apply(list)

groups = grouped_data.to_dict()
print(groups)

colors = cm.get_cmap('Set1', len(groups))

# Generate histograms for each group
for i, (design, accuracies) in enumerate(groups.items()):
    color = colors(i)  # Get a unique color for each group
    plt.hist(accuracies, bins=10, alpha=0.7, color=color, label=f'Design {design}')
    # Calculate and plot the median for each group
    mean = pd.Series(accuracies).mean()
    plt.axvline(mean, color=color, linestyle='dashed', linewidth=1, label=f'Mean {design}')

# Add labels and legend
plt.xlabel('Time (in seconds)')
plt.ylabel('Score Frequency')
plt.title('timeTaken by Design')
plt.legend()

# Show the plot
plt.savefig('timeTaken_plot.png', transparent=True)

# Perform a one-way ANOVA test
anova_result = stats.f_oneway(*groups.values())

# Print the ANOVA test result
print("ANOVA Result:", anova_result)
