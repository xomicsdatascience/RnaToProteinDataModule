import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from collections import Counter

def calculate_replicate_mse(df, id_column, replicate_column, type_column):
    # Group the dataframe by the ID column
    groups = df.groupby(id_column)

    mse_results = []
    random_results = []
    ids = []
    for id, group in groups:
        # Calculate MSE between each pair of replicates
        mse_list = []
        randomized_list = []
        replicates = group[replicate_column].unique()
        if len(replicates) > 1:  # Only calculate MSE if there are multiple replicates
            for replicate1, replicate2 in combinations(replicates, 2):
                values1 = group[group[replicate_column] == replicate1].drop(
                    columns=[id_column, replicate_column, type_column]).values
                values2 = group[group[replicate_column] == replicate2].drop(
                    columns=[id_column, replicate_column, type_column]).values
                mse = mean_squared_error(values1, values2)
                mse_list.append(mse)
            for replicate in replicates:
                values = group[group[replicate_column] == replicate].drop(
                    columns=[id_column, replicate_column, type_column]).values
                shuffled_values = np.array([np.random.permutation(values[0])])
                #print(values)
                #print(shuffled_values)
                random_mse = mean_squared_error(values, shuffled_values)
                randomized_list.append(random_mse)


        # Compute the average MSE for the group
        if not mse_list: continue
        mse_results.append(np.mean(mse_list))
        random_results.append(np.mean(randomized_list))
        ids.append(id)

    return mse_results, random_results, ids


df = pd.read_csv('/Users/cranneyc/Documents/Projects/CPTAC_analysis/makingABetterModel_NAS/RnaToProteinDataModule/scripts/SHAP/1SHAP_output_consolidations/consolidated_SHAP_MMP14_rs1-3.csv')
print(df)
result_dict = df.set_index('id')['type'].to_dict()

# Example usage
# Assuming df is your pandas DataFrame with columns: ID, Replicate, Value1, Value2, ...
# Replace "ID" and "Replicate" with your actual column names
mse_results, random_results, ids = calculate_replicate_mse(df, "id", "randomSeed", "type")
print(mse_results)
print(np.mean(mse_results))
print()
print(random_results)
print(np.mean(random_results))
types = [result_dict[id] for id in ids]
lesserTypes = [types[i] for i in range(len(types)) if random_results[i] < 0.00011]
greaterTypes = [types[i] for i in range(len(types)) if random_results[i] >= 0.00011]
print(Counter(lesserTypes))
print(Counter(greaterTypes))
# Plot histograms
plt.hist(mse_results, alpha=0.5, label='True MSE values')
plt.hist(random_results, alpha=0.5, label='Randomized MSE values')

# Add labels and legend
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Overlapping Histograms')
plt.legend()

# Show plot
plt.savefig('/Users/cranneyc/Documents/Projects/CPTAC_analysis/makingABetterModel_NAS/RnaToProteinDataModule/scripts/SHAP/Figure_1.png')
