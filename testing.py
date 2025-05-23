import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import linregress
from scipy import stats
from DataAnalysis import analysisutil as ana

import util

########################################################################################################################
##
##                    -- Testing --
## This file is for testing and comparing the results of the pruned and original
## instances of the LNS .
##
########################################################################################################################startingPoints = np.array([])




# Initialize arrays to store lines from files
original_best_obj = []
pruned_best_obj = []
original_total_time = []
pruned_total_time = []

def additional_visualisations():
    boxcox_original_total_time, lambda_ = stats.boxcox(np.array(original_total_time))
    print(f"Optimal λ: {lambda_}")
    boxcox_pruned_total_time, lambda_ = stats.boxcox(np.array(pruned_total_time))
    print(f"Optimal λ: {lambda_}")
    log_original_total_time = np.log(np.array(original_total_time))/np.log(10)
    log_pruned_total_time = np.log(np.array(pruned_total_time))/np.log(10)

    # Visualization for individual distributions
    plt.figure(figsize=(14, 7))

    # Histogram for objective values
    plt.subplot(2, 2, 1)
    sns.histplot(original_best_obj, color='blue', kde=True, label='Original Best Obj')
    sns.histplot(pruned_best_obj, color='red', kde=True, label='Pruned Best Obj')
    plt.legend()
    plt.title('Distribution of Original Best Obj and Pruned Best Obj')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    # Histogram for total time
    plt.subplot(2, 2, 2)
    sns.histplot(original_total_time, color='green', kde=True, label='Original Total Time', bins=50)
    sns.histplot(pruned_total_time, color='purple', kde=True, label='Pruned Total Time', bins=50)
    plt.legend()
    plt.title('Distribution of Original Total Time and Pruned Total Time')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    # Boxplot for objective values
    plt.subplot(2, 2, 3)
    plt.boxplot([original_best_obj, pruned_best_obj], tick_labels=['Original Best Obj', 'Pruned Best Obj'],
                patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='blue'),
                medianprops=dict(color='red'))
    plt.title('Boxplot Comparison for Original Best Obj and Pruned Best Obj')
    plt.ylabel('Value')
    # Boxplot for total time
    plt.subplot(2, 2, 4)
    plt.boxplot([original_total_time, pruned_total_time], tick_labels=['Original Total Time', 'Pruned Total Time'],
                patch_artist=True,
                boxprops=dict(facecolor='lightgreen', color='green'),
                medianprops=dict(color='purple'))
    plt.title('Boxplot Comparison for Original Total Time and Pruned Total Time')
    plt.ylabel('Value')
    # Show individual distribution plots
    plt.suptitle('Instance ' + directory, fontsize=16)
    plt.tight_layout()
    plt.savefig("plots/" + directory + ".png")
    plt.show()

    # Combined analysis of (original total time, original best obj) and (pruned total time, pruned best obj)
    plt.figure(figsize=(14, 7))
    # Scatter plot for (original total time, original best obj)
    plt.subplot(1, 2, 1)
    plt.scatter(original_total_time, original_best_obj, color='blue', label='(Original Total Time, Original Best Obj)')
    slope1, intercept1, r_value1, p_value1, std_err1 = linregress(original_total_time, original_best_obj)
    plt.plot(original_total_time, intercept1 + slope1 * np.array(original_total_time), 'r',
             label=f'Fit line: y = {intercept1:.2f} + {slope1:.2f}x')
    plt.xlabel('Original Total Time')
    plt.ylabel('Original Best Obj')
    plt.title('Scatter plot and Linear Fit for (Original Total Time, Original Best Obj)')
    plt.legend()
    plt.grid(True)

    # Scatter plot for (pruned total time, pruned best obj)
    plt.subplot(1, 2, 2)
    plt.scatter(pruned_total_time, pruned_best_obj, color='green', label='(Pruned Total Time, Pruned Best Obj)')
    slope2, intercept2, r_value2, p_value2, std_err2 = linregress(pruned_total_time, pruned_best_obj)
    plt.plot(pruned_total_time, intercept2 + slope2 * np.array(pruned_total_time), 'r',
             label=f'Fit line: y = {intercept2:.2f} + {slope2:.2f}x')
    plt.xlabel('Pruned Total Time')
    plt.ylabel('Pruned Best Obj')
    plt.title('Scatter plot and Linear Fit for (Pruned Total Time, Pruned Best Obj)')
    plt.legend()
    plt.grid(True)

    # Show combined scatter plots
    plt.tight_layout()
    plt.show()

def calculate_statistics(array, name):
    """
    Calculates mean, median, variance and standard deviation for the given list and
    also returns it as a dictionary with the namen given
    :param array: list to calculate statistics
    :param name: name to add to the calculated statistics
    :return: the calculated statistics
    """
    mean = np.mean(array)
    median = np.median(array)
    variance = np.var(array)
    std_dev = np.std(array)
    return {
        "name": name,
        "mean": mean,
        "median": median,
        "variance": variance,
        "std_dev": std_dev,
        "min": np.min(array),
        "max": np.max(array)
    }

def percentage_difference(original_value, pruned_value):
    """
    Calculated the percentage difference between the two given values.
    :param original_value: first value to compare
    :param pruned_value: second value to compare
    :return: the percentage difference
    """
    if original_value != 0:
        return ((original_value - pruned_value) / original_value) * 100
    return 0  # Avoid division by zero


def compare_statistics(original, pruned, original_name, pruned_name, instance_name):
    """
    This takes 2 lists, calculates statistics for each of them, compares those statistics
    both in absolut and percentage terms and prints the results with the given names.

    :param original: first list to calculate statistics for
    :param pruned: second list to calculate statistics for
    :param original_name: the name that should be set for the statistic results of the first list original
    :param pruned_name: the name that should be set for the statistic results of the second list pruned
    :param instance_name: the main title of the printed table
    :return: nothing is returned. The results are only printed.
    """
    original_stats = calculate_statistics(original, original_name)
    pruned_stats = calculate_statistics(pruned, pruned_name)

    # Print comparison results
    print(f"Comparison between {original_name} and {pruned_name} for {instance_name}:")
    print(f"{'Statistic':<20} {'Original':<15} {'Pruned':<15} {'Difference':<15} {'% Difference':<15}")

    for stat in ['mean', 'median', 'variance', 'std_dev', 'min', 'max']:
        original_value = original_stats[stat]
        pruned_value = pruned_stats[stat]
        difference = original_value - pruned_value
        percent_diff = percentage_difference(original_value, pruned_value)

        print(f"{stat.capitalize():<20} {original_value:<15.3f} {pruned_value:<15.3f}"
              f" {difference:<15.3f} {percent_diff:<10.2f}% ")


#directories_original = ["a5-60", "a6-48", "a6-60", "a6-72", "a7-56", "a7-70", "a7-84", "a8-64", "a8-80", "a8-96"]
directories_original = ["a180-3600", "a200-4000", "a220-4400", "a240-4800", "a260-5200"]
directories_pruned = directories_original
#thresholds_pruned = ["-0.0375", "0.0375", "0.0750", "0.1000", "0.1750"]
#time_limits = [1100, 1200, 1300, 1400, 1500]

# Process directories with 100 files each
for i in range(len(directories_original)):
    original_total_time = []
    pruned_total_time = []
    original_best_obj = []
    pruned_best_obj = []
    directory = directories_original[i]
    # Get all txt files in the directory
    files_path = "C:/Users/caspi/Documents/TU_Wien/Bachelorarbeit/Data/k_nearest/10_02_25/0.250/" + directory + "/"
    files = os.listdir(files_path)
    for file in files:
        if file.endswith('.out'):
            file_path = os.path.join(files_path, file)
            with open(file_path, 'r') as f:
                #obj_value = ana.extract_value_at_given_time(file_path, time_limits[i])
                obj_value = util.extract_objectiveFun(f)
                original_best_obj.append(obj_value)
                f.seek(0)
                original_total_time.append(util.extract_run_time(f))
    directory = directories_pruned[i]
    files_path = "C:/Users/caspi/Documents/TU_Wien/Bachelorarbeit/Data/svm/custom_weightings_nystroem_inv/08_11/" + directory + "/1.250/"
    files = os.listdir(files_path)
    for file in files:
        if file.endswith('.out'):
            file_path = os.path.join(files_path, file)
            with open(file_path, 'r') as f:
                pruned_best_obj.append(util.extract_objectiveFun(f))
                f.seek(0)
                pruned_total_time.append(util.extract_run_time(f))
    # Calculate statistics
    compare_statistics(original_best_obj, pruned_best_obj, "Original Best Obj", "Pruned Best Obj", directory)
    compare_statistics(original_total_time, pruned_total_time, "Original Total Time", "Pruned Total Time", directory)
    additional_visualisations()




