import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt


data = pd.read_csv("results.csv")


grouped_data = data.groupby('design')['accuracy'].apply(list)

groups = grouped_data.to_dict()
print(groups)


# Generate histograms for each group
for design, accuracies in groups.items():
    plt.hist(accuracies, bins=10, alpha=0.7, label=f'Design {design}')

# Add labels and legend
plt.xlabel('Accuracy')
plt.ylabel('Frequency')
plt.title('Histograms of Accuracy by Design')
plt.legend()

# Show the plot
plt.show()

# Perform a one-way ANOVA test
anova_result = stats.f_oneway(*groups.values())

# Print the ANOVA test result
print("ANOVA Result:", anova_result)
