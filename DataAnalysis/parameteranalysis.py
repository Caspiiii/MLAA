import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import statistics as stat
from statsmodels.stats.multitest import multipletests
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import MinMaxScaler

import util
import analysisutil as anautil

########################################################################################################################
##
##                      -- parametersARAMETER ANALYSIS --
##  Analyse behavior of the EADARP algorithm with varying parameters
##
########################################################################################################################

def svm_parameter_analysis(parameter_values, objective_values, name):
    """
    For improved performance of the svm in the pruning process, varying possible thresholds for the linear SVM were tested.
    A threshold over 0 leads to a higher percentage of arcs removed. This function analyses the differences in the resulitng
    objective function values of the different settings for the algorithm.
    :param parameter_values: A list of the names of the parameter values.
    :param objective_values: A list of floats each representing the objective function value of one EADARP solution. While
     both lists need to have the same length, the idea is that multiple objective values have been assigned the same parameter
     value. Therefore, the code also groups by parameter values.
    :param name: The name of the instance.
    :return: Prints statistical analysis and plots for visualisation.
    """
    data = pd.DataFrame({
        'parameter': parameter_values,
        'objective_value': objective_values
    })
    # Statistical analysis
    grouped = data.groupby('parameter')
    groups = []
    for _, group in grouped:
        values = group['objective_value'].values
        groups.append(values)
    means = grouped['objective_value'].mean()
    mins = grouped['objective_value'].min()
    maxs = grouped['objective_value'].max()
    stds = grouped['objective_value'].std()
    best_parameter = means.idxmin()
    best_mean_value = means.min()
    min = mins[best_parameter]
    max = maxs[best_parameter]
    print(f"The best parameter is: {best_parameter} with an average objective value of {best_mean_value}, and "
          f"an minimum objective value of {min}, and an maximum objective value of {max}")
    # Visualisation
    plt.figure(figsize=(14, 6))
    data.boxplot(column='objective_value', by='parameter', grid=False)
    plt.title('Objective Value Distribution by Parameter ' + name)
    plt.suptitle('')
    plt.xlabel('Parameter')
    plt.ylabel('Objective Value')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('obj_value_by_parameter_' + name + '_1.png', dpi=600)  # Save plot with desired filename and dpi
    plt.show()
    plt.figure(figsize=(14, 6))
    plt.errorbar(means.index.astype(str), means.values, yerr=stds.values, fmt='o', ecolor='r', capsize=5)
    plt.title('Mean Objective Value by Parameter ' + name)
    plt.xlabel('Parameter')
    plt.ylabel('Mean Objective Value')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('obj_value_by_parameter_' + name + '_2.png', dpi=600)  # Save plot with desired filename and dpi
    plt.show()
    #means_significance_analysis(groups, means.index, name)

def means_significance_analysis(groups, parameters, name):
    """
    Tests if the differences of mean are overall statistically significant via one way ANOVA and also tests
    the statistical significance of the differences of each mean to every other mean.
    :param groups: A python list of the grouped objective function values.
    :param parameters: The names of the parameters corresponding to each mean. That is the mean has been calculated
     for each parameter.
    :param name: The name of the instance.
    :return: Prints the results of both tests.

      Note:
    - The Wilcoxon test in SciPy can be given an 'alternative' argument: 'greater', 'less', or 'two-sided'.
    - Here, we perform two tests for each pair (i, j):
        1. H0: median(groups[i]) = median(groups[j]) against H1: median(groups[i]) > median(groups[j])
        2. H0: median(groups[i]) = median(groups[j]) against H1: median(groups[i]) < median(groups[j])
      This effectively gives you information on the direction of the effect.
    """
    # one way ANOVA to test if the means actually differ significantly (not really necessary tbh,
    # as the very big parameters are just clearly worse at the moment. Still left it in there as
    # later a different selection of parameters might be chosen which are closer together etc.)
    f_statistic, p_value = stats.f_oneway(*groups)
    print(f"ANOVA test results: F-statistic = {f_statistic:.4f}, p-value = {p_value:.4f}")
    # Tukey HSD I thought to be an interesting analysis as there actually is a lot of overlap in parameter
    # that are close to another, especially for smaller parameters. I assume the variability of the calculations
    # on the cluster has a big play in the overlap, but with increasing sample size that should even out.
    tukey_results = stats.tukey_hsd(*groups)
    p_values = tukey_results.pvalue
    header = [" " * 1] + parameters.astype(str)
    print("********** |", " | ".join(header))
    print("-" * (10 + 7 * len(parameters)))
    for i, row_param in enumerate(parameters):
        row = []
        for j, col_param in enumerate(parameters):
            value = f"{p_values[i][j]:.2f}"
            row.append(value)
        print(f"{row_param:<10} | " + " | ".join(f"{value:<8}" for value in row))
    n = len(parameters)
    p_values = np.full((n, n), np.nan)  # Initialize a matrix of p-values
    for i in range(n):
        for j in range(n):
            if i != j:
                # Perform a paired Wilcoxon signed-rank test between the two groups
                stat, p = stats.wilcoxon(groups[i], groups[j])
                p_values[i, j] = p

    header = [" " * 1] + parameters.astype(str)
    print("********** |", " | ".join(header))
    print("-" * (10 + 7 * n))
    for i, row_param in enumerate(parameters):
        # Print p-values in scientific notation with 4 decimal places
        row = [f"{p_values[i][j]:.4e}" if not np.isnan(p_values[i][j]) else "NaN" for j in range(n)]
        print(f"{row_param:<10} | " + " | ".join(f"{val:<12}" for val in row))

    # Transform p-values to -log10 for the heatmap to highlight very small differences
    # Avoid taking log of NaN or zero values by adding a small epsilon
    epsilon = 1e-300  # Add a tiny value to avoid log(0)
    p_values_transformed = -np.log10(np.where(np.isnan(p_values), np.nan, np.maximum(p_values, epsilon)))

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(
        p_values_transformed,
        annot=False,
        fmt=".2f",
        xticklabels=parameters,
        yticklabels=parameters,
        cmap="viridis",
        cbar_kws={'label': '-log10(p-value)'},
        linewidths=0.5
    )
    ax.set_title("Pairwise Wilcoxon Signed-Rank Test (-log10(p-values))")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f'heatmap_willcoxon_{name}_scaled', dpi=300)
    plt.show()
    """
    n = len(parameters)
    p_values_greater = np.full((n, n), np.nan)
    p_values_less = np.full((n, n), np.nan)
    

    # Perform pairwise one-sided Wilcoxon signed-rank tests
    for i in range(n):
        for j in range(n):
            if i != j:
                # Test if groups[i] > groups[j]
                stat_greater, p_greater = stats.wilcoxon(groups[i], groups[j], alternative='greater')
                p_values_greater[i, j] = p_greater

                # Test if groups[i] < groups[j]
                stat_less, p_less = stats.wilcoxon(groups[i], groups[j], alternative='less')
                p_values_less[i, j] = p_less

    # Print results in a table format
    header = [" " * 1] + parameters.astype(str)
    print("********** One-Sided Tests (i > j) **********")
    print("********** |", " | ".join(header))
    print("-" * (10 + 7 * n))
    for i, row_param in enumerate(parameters):
        row = [f"{p_values_greater[i][j]:.2e}" if not np.isnan(p_values_greater[i][j]) else "NaN" for j in range(n)]
        print(f"{row_param:<10} | " + " | ".join(f"{val:<8}" for val in row))

    print("/n********** One-Sided Tests (i < j) **********")
    print("********** |", " | ".join(header))
    print("-" * (10 + 7 * n))
    for i, row_param in enumerate(parameters):
        row = [f"{p_values_less[i][j]:.2e}" if not np.isnan(p_values_less[i][j]) else "NaN" for j in range(n)]
        print(f"{row_param:<10} | " + " | ".join(f"{val:<8}" for val in row))

    # Plotting p-values for i > j
    plt.figure(figsize=(10, 8))
    ax1 = sns.heatmap(
        p_values_greater,
        annot=False,
        fmt=".2e",
        xticklabels=parameters,
        yticklabels=parameters,
        cmap="coolwarm",
        cbar_kws={'label': 'p-value (i > j)'},
        linewidths=0.5
    )
    ax1.set_title("Pairwise Wilcoxon Signed-Rank Test (i > j)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f'heatmap_wilcoxon_greater_{name}.png', dpi=300)
    plt.show()

    # Plotting p-values for i < j
    plt.figure(figsize=(10, 8))
    ax2 = sns.heatmap(
        p_values_less,
        annot=False,
        fmt=".2e",
        xticklabels=parameters,
        yticklabels=parameters,
        cmap="coolwarm",
        cbar_kws={'label': 'p-value (i < j)'},
        linewidths=0.5
    )
    ax2.set_title("Pairwise Wilcoxon Signed-Rank Test (i < j)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f'heatmap_wilcoxon_less_{name}.png', dpi=300)
    plt.show()
    """

def execute_svm_analysis():
    # First iterate over all thresholds
    directories_path = "C:/Users/caspi/Documents/TU_Wien/Bachelorarbeit/Data/svm/custom_threshold_linear_inv/07_11/"
    directories = os.listdir(directories_path)
    for directory in directories:
        # Then per threshold over all instances
        threshold_path = os.path.join(directories_path, directory)
        instance_directories = os.listdir(threshold_path)
        parameter_values = []
        objective_values = []
        for instance_directory in instance_directories:
            instance_path = os.path.join(threshold_path, instance_directory)
            files = os.listdir(instance_path)
            for file in files:
                if file.endswith(".out"):
                    with open(os.path.join(instance_path, file), 'r') as read_file:
                        #objective_value = anautil.extract_value_at_given_time(os.path.join(instance_path, file), 600)
                        #objective_value = util.extract_run_time(read_file)
                        objective_value = util.extract_objectiveFun(read_file)
                        if objective_value == -1:
                            raise Exception(f"For {file} no objective function value has been found")
                        parameter_values.append(f"{float(instance_directory):.4f}")
                        objective_values.append(objective_value)
        print("##########################################################################################")
        print(f"Threshold: {directory}. Linear SVM")
        print("##########################################################################################")
        sorted_pairs = sorted(zip(list(map(float, parameter_values)), objective_values))
        sorted_list1, sorted_list2 = zip(*sorted_pairs)
        parameter_values = list(sorted_list1)
        objective_values = list(sorted_list2)
        svm_parameter_analysis(parameter_values, objective_values, directory)

def execute_svm_analysis_classic():
    """
    Just calculates some basic metrics like average min and max for the different instances
    :return:
    """
    directories_path = "C:/Users/caspi/Documents/TU_Wien/Bachelorarbeit/Data/classic/15_12/"
    directories = os.listdir(directories_path)
    #all instances
    for directory in directories:
        objective_values = []
        directory_path = os.path.join(directories_path, directory)
        files = os.listdir(directory_path)
        for file in files:
            if file.endswith(".out"):
                with open(os.path.join(directory_path, file), 'r') as read_file:
                    #objective_values.append(anautil.extract_value_at_given_time(os.path.join(directory_path, file), 600))
                    objective_values.append(util.extract_objectiveFun(read_file))
        mean = stat.mean(objective_values)
        print(f"Instance {directory} has an average objective value of {mean}, an "
              f"minimum objective value of {min(objective_values)}, and an maximum objective value of {max(objective_values)}")


def check_feasability():
    """
    Checks feasability for the results of a number of directories and also outputs some basic metrics
    """
    directories_threshold_path = "C:/Users/caspi/Documents/TU_Wien/Bachelorarbeit/Data/svm/custom_weightings_nystroem/06_02_25/"
    for threshold_directory in os.listdir(directories_threshold_path):
        directories_path = directories_threshold_path + threshold_directory
        directories = os.listdir(directories_path)
        print("####################################################################################")
        print(f"Threshold: {threshold_directory}")
        print("####################################################################################")

        # all instances
        for directory in directories:
            objective_values = []
            total_run_times = []
            directory_path = os.path.join(directories_path, directory)
            files = os.listdir(directory_path)
            for file in files:
                if file.endswith(".out"):
                    solution_path = os.path.join(directory_path, file)
                    with open(solution_path, 'r') as read_file:
                        #objective_values.append(anautil.extract_value_at_given_time(os.path.join(directory_path, file), 600))
                        objective_values.append(util.extract_objectiveFun(read_file))
                        read_file.seek(0)
                        total_run_times.append(util.extract_run_time(read_file))

            mean = stat.mean(objective_values)
            mean_total_run_times = stat.mean(total_run_times)
            print(f"Threshold {threshold_directory}, Instance {directory} has an avg obj val of {mean}, an max obj val of {max(objective_values)} and "
                  f" a total pruning rate of {util.extract_arcs_removed_percentages(solution_path)[1]}")
            print(f"Corresponding total run time: {mean_total_run_times}")



def obj_value_development_plot(best_pruned_parameters, best_pruned_parameters_nyst):
    """
    Plots the development of the objective function over time for a given solution.
    :param best_pruned_parameters is a python list containing the best parameters for each pruned instance
    :param best_pruned_parameters_nyst is a python list containing the best parameters for each pruned instance based on the nystroem approach
    """
    directories_path = "C:/Users/caspi/Documents/TU_Wien/Bachelorarbeit/Data/24h_maria/24h/"
    directories = os.listdir(directories_path)
    #for directory in directories:
        #directory_path = os.path.join(directories_path, directory)
        #files = os.listdir(directory_path)
    directories_path_pruned = "C:/Users/caspi/Documents/TU_Wien/Bachelorarbeit/Data/svm/24h/linear/"
    pruned_directory = os.listdir(directories_path_pruned)
    linear_list = (file for file in pruned_directory if file.endswith(".out"))
    directories_path_pruned_nyst = "C:/Users/caspi/Documents/TU_Wien/Bachelorarbeit/Data/svm/24h/nystroem/"
    # directory_path_pruned_nyst = os.path.join(directories_path_pruned_nyst, directory,
    #                                     best_pruned_parameters_nyst[directory])
    pruned_directory_nyst = os.listdir(directories_path_pruned_nyst)
    nystroem_list = (file for file in pruned_directory_nyst if file.endswith(".out"))
    for file in directories:
        if file.endswith(".out"):
            route_development_obj, route_development_times = anautil.extract_route_development(os.path.join(directories_path, file))
            #directory_path_pruned = os.path.join(directories_path_pruned, directory, best_pruned_parameters[directory])
            pruned_file = next(linear_list, None)
            route_development_obj_pruned, route_development_times_pruned = anautil.extract_route_development(
                os.path.join(directories_path_pruned, pruned_file))

            pruned_file_nyst = next(nystroem_list, None)
            route_development_obj_pruned_nyst, route_development_times_pruned_nyst = anautil.extract_route_development(
                os.path.join(directories_path_pruned_nyst, pruned_file_nyst))

            """
            for pruned_file in pruned_files:
                if pruned_file.endswith(".out"):
                    route_development_obj_pruned, route_development_times_pruned = anautil.extract_route_development(
                        os.path.join(directories_path_pruned, directory, pruned_file))
                    break
            times_array_pruned = np.array(route_development_times_pruned)
            obj_array_pruned = np.array(route_development_obj_pruned)
            times_array = np.array(route_development_times)
            obj_array = np.array(route_development_obj)
            degree = 1
            coeffs_pruned = np.polyfit(times_array_pruned[len(times_array_pruned) // 2:], obj_array_pruned[len(obj_array_pruned) // 2:], deg=degree)
            poly_model_pruned = np.poly1d(coeffs_pruned)
            future_times_pruned = np.arange(min(times_array_pruned), 1501, 10)
            future_prediction_pruned = poly_model_pruned(future_times_pruned)
            coeffs = np.polyfit(times_array, obj_array, deg=degree)
            poly_model = np.poly1d(coeffs)
            future_times = np.arange(min(times_array), 1501, 10)
            future_prediction = poly_model(future_times)
            """
            plt.figure(figsize=(10, 6))

            # Plot original objective values
            plt.plot(route_development_times, route_development_obj, linewidth=1.5, color='lime',
                     label='Original Objective Value')

            #Plot pruned objective values
            plt.plot(route_development_times_pruned, route_development_obj_pruned, linewidth=1.5, color='cyan',
                     label='Pruned Objective Value Linear')

            plt.plot(route_development_times_pruned_nyst, route_development_obj_pruned_nyst, linewidth=1.5, color='blue',
                     label='Pruned Objective Value Nystroem')
            """
            # Plot the polynomial regression predictions
            plt.plot(future_times, future_prediction, '-', label="forecast original values", color='crimson')
            plt.plot(future_times_pruned, future_prediction_pruned, '-', label=f'forecast pruned values', color='salmon')
            
            plt.ylim(min(min(route_development_obj_pruned), min(future_prediction_pruned)) - 2,
                     max(max(future_prediction), max(route_development_obj_pruned)) + 2)
            """
            plt.grid(visible=True, linestyle='--', alpha=0.7)
            plt.title('Objective Function Progression ' + file[:-7], fontsize=16, fontweight='bold')
            plt.xlabel('Time', fontsize=14)
            plt.ylabel('Objective Function Value', fontsize=14)
            plt.legend(fontsize=12)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.savefig(f'obj_val_progression_{file[:-7]}', dpi=300)
            plt.show()
            #break


#obj_value_development_plot({"a180-3600": "-0.0375",  "a200-4000": "0.0375", "a220-4400": "0.0750", "a240-4800": "0.1000", "a260-5200": "0.1750"},
#                           {"a180-3600": "1.000",  "a200-4000": "1.000", "a220-4400": "1.250", "a240-4800": "1.250", "a260-5200": "1.250"})
"""
obj_value_development_plot({"a180-3600": "-0.0375",  "a200-4000": "0.0375", "a220-4400": "0.0750", "a240-4800": "0.1000", "a260-5200": "0.1750"},
                           {
                              "a5-60": "0.500",
                             "a6-48": "0.500", "a6-60": "0.500", "a6-72": "0.500",
                            "a7-56": "0.500", "a7-70": "0.500", "a7-84": "0.500",
                           "a8-64": "0.500", "a8-80": "0.500", "a8-96": "0.500"
                      }
                     )
"""
#execute_svm_analysis()
#execute_svm_analysis_classic()
check_feasability()
