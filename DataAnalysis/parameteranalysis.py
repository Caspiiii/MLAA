import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats

import util


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
    stds = grouped['objective_value'].std()
    best_parameter = means.idxmin()
    best_mean_value = means.min()
    print(f"The best parameter is: {best_parameter} with an average objective value of {best_mean_value}")
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
    means_significance_analysis(groups, means.index)

def means_significance_analysis(groups, parameters):
    """
    Tests if the differences of mean are overall statistically significant via one way ANOVA and also tests
    the statistical significance of the differences of each mean to every other mean.
    :param groups: A python list of the grouped objective function values.
    :param parameters: The names of the parameters corresponding to each mean. That is the mean has been calculated
     for each parameter.
    :return: Prints the results of both tests.
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

def execute_svm_analysis():
    # First iterate over all thresholds
    directories_path = "C:/Users/caspi/Documents/TU_Wien/Bachelorarbeit/Data/classic/09_11/helper/"
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
                    with open(os.path.join(instance_path, file), 'r') as file:
                        objective_value = util.extract_objectiveFun(file)
                        #objective_value = util.extract_run_time(file)
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


execute_svm_analysis()