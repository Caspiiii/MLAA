import util
import numpy as np
import os
import math
from statistics import mean


########################################################################################################################
##
##                      -- DATA Processing --
##  Extract and Prepares the data necessary for training of the ML models. This includes
##  calculating both labels and feature vectors.
##
########################################################################################################################

def calculate_neighbourhood(vertice, neighbourhood):
    v = vertice
    def sortByDist(verticeToCompare):
        if v != verticeToCompare and math.dist(v, verticeToCompare) > 0:
            return math.dist(v, verticeToCompare)
        return float('inf')
    return sorted(neighbourhood, key = sortByDist)

def calculate_clusterness(vertice, neighbourhood, radius):
    v = vertice
    def inCluster(verticeToCompare):
        if v != verticeToCompare and math.dist(v, verticeToCompare) > 0:
            return math.dist(v, verticeToCompare) < radius
        return False
    return len(list(filter(inCluster, neighbourhood)))

def calculate_nearness(neighbour, neighbourhood):
        return neighbourhood.index(neighbour)


def create_input(path, tightened_windows_path, infeasible_arcs_path, solutions_path):
    ## Creation of input vector
    directory_path = path
    files = os.listdir(directory_path)
    ## calculate the input vector for each solution
    inputEdges = []
    euclidean_avg = []
    for filename in files:
        print(filename)
        # Create the full path to the file
        file_path = os.path.join(directory_path, filename)
        #print("Calculating input for:" + filename)
        # Check if it's a file (and not a directory)
        if os.path.isfile(file_path):
            # Open the file
            with open(file_path, 'r') as file:
                # Read the contents of the file
                contents = file.readlines()
                nodes = contents[1:len(contents) - 13]


            # calculate the labels for the input
            solutionsName = filename[0:len(filename) - 4]
            #labels = create_labels_simple(solutions_path + solutionsName + "/", len(nodes))
            labels = create_labels_simple(solutions_path + solutionsName + "/", len(nodes))

            #get infeasible arcs
            infeasibleEdges = []
            file_path_infeasible = os.path.join(infeasible_arcs_path, filename)
            if os.path.isfile(file_path_infeasible):
                # Open the file
                with open(file_path_infeasible, 'r') as file:
                    # Read the contents of the file
                    for line in file:
                        values = line.split(" ")
                        values = [s for s in values if s != '' and '\n' not in s]
                        infeasibleEdges.append(values)

            # extract necessary data
            vertices = util.extract_vertices(nodes)
            timeWindows = util.extract_timeWindow(tightened_windows_path + filename)
            startingPoints = util.extract_startingPoints(contents)
            startingPointsCenter = util.calculate_centroid(startingPoints)
            destinationPoints = util.extract_endingPoints(contents)
            destinationPointsCenter = util.calculate_centroid(destinationPoints)
            chargingStationPoints = util.extract_charging_stations(contents)
            n_P = util.instance_size(file_path)["n_P"]
            # iterate over each edge
            for i in range(len(vertices)):
                sortedNeighbourhood = calculate_neighbourhood(vertices[i], vertices)
                for j in range(len(vertices)):
                    #if infeasibleEdges[i][j] == 'true':
                    #    continue
                    dataPoint = []
                    if vertices[i] in startingPoints or vertices[i] in destinationPoints or vertices[i] in chargingStationPoints:
                        continue
                    if vertices[j] in startingPoints or vertices[j] in destinationPoints or vertices[j] in chargingStationPoints:
                        continue
                    #if i < n_P and j == i + n_P:
                    #    continue
                    print("Calculating Edge Length")
                    print("-----" * 40)
                    if i != j:
                        # edge length
                        if (math.dist(vertices[i], vertices[j])) == 0:
                            dataPoint.append(100)
                        else:
                            dataPoint.append(math.dist(vertices[i], vertices[j]))
                            if (labels[i, j] == 1):
                                euclidean_avg.append(math.dist(vertices[i], vertices[j]))
                        print("Calculating Nearness of Nodes")
                        print("-----" * 40)
                        # nearness of nodes
                        #dataPoint.append(calculate_nearness(vertices[j], sortedNeighbourhood))
                        """
                        # distance from start
                        # Typically there are more than one starting point
                        # To look at the distance from the start we take the geometric center of all starting points
                        # For the edge we look at the center of the edge
                        print("Calculating Distance from Start")
                        print("-----" * 40)
                        edgeCenter = util.calculate_centroid([vertices[i], vertices[j]])
                        minimumStartingPoint = util.find_smallest_distance(startingPoints, edgeCenter)
                        maximumStartingPoint = util.find_biggest_distance(startingPoints, edgeCenter)
                        minimumDestinationPoint = util.find_smallest_distance(destinationPoints, edgeCenter)
                        maximumDestinationPoint = util.find_biggest_distance(destinationPoints, edgeCenter)
                        dataPoint.append(math.dist(startingPointsCenter, edgeCenter))
                        dataPoint.append(math.dist(max(startingPoints, key=lambda item: abs(item[0] + item[1])), edgeCenter))
                        dataPoint.append(math.dist(min(startingPoints, key=lambda item: abs(item[0] + item[1])), edgeCenter))
                        dataPoint.append(math.dist(minimumStartingPoint, edgeCenter))
                        dataPoint.append(math.dist(maximumStartingPoint, edgeCenter))
                        # distance from end
                        # same principle as for start
                        print("Calculating Distance To End")
                        print("-----" * 40)
                        dataPoint.append(math.dist(destinationPointsCenter, edgeCenter))
                        dataPoint.append(math.dist(min(destinationPoints, key=lambda item: abs(item[0] + item[1])), edgeCenter))
                        dataPoint.append(math.dist(max(destinationPoints, key=lambda item: abs(item[0] + item[1])), edgeCenter))
                        dataPoint.append(math.dist(minimumDestinationPoint, edgeCenter))
                        dataPoint.append(math.dist(maximumDestinationPoint, edgeCenter))
                        """
                        """
                        print("Calculating Clusterness")
                        print("-----" * 40)
                        # density
                        
                        dataPoint.append(calculate_clusterness(vertices[i], vertices, 5))
                        dataPoint.append(math.dist(vertices[i], vertices[j]) <= 5)
                        """
                        """
                        print("Calculating Charging Station Edge")
                        print("-----" * 40)
                        # charging station edge
                        dataPoint.append(vertices[i] in chargingStationPoints)
                        dataPoint.append(vertices[j] in chargingStationPoints)
                        dataPoint.append(vertices[j] in chargingStationPoints and vertices[i] in chargingStationPoints)
                        """
                        """
                        print("Calculating Alpha-Nearness")
                        print("-----" * 40)
                        alphanearness
                        dataPoint.append(util.calculatealphaNearness(i, j))
                        dataPoint.append(util.calculateMinimumSpanningTreeWithEdge(i, j))
                        dataPoint.append(util.calculateMinimumSpanningTree())
                        """
                        print("Calculating Time Windows")
                        print("-----" * 40)
                        #timewindows
                        dataPoint.append(
                            ((timeWindows[i][1] - timeWindows[j][1]) + (timeWindows[i][0] - timeWindows[j][0])) / 2)
                        dataPoint.append(timeWindows[i][1] - timeWindows[j][1])
                        dataPoint.append(timeWindows[i][0] - timeWindows[j][0])
                        dataPoint.append(timeWindows[i][1] - timeWindows[j][0])
                        dataPoint.append(labels[i, j])
                        inputEdges.append(dataPoint)
    print("Euclidean mean is", mean(euclidean_avg))
    return inputEdges

def create_input_knn(directory_path, infeasible_arcs_path, solutions_path):
    ## Creation of input vector
    files = os.listdir(directory_path)
    ## calculate the input vector for each solution
    inputEdges = []
    euclidean_avg = []
    for filename in files:
        print(filename)
        # Create the full path to the file
        file_path = os.path.join(directory_path, filename)
        # Check if it's a file (and not a directory)
        if os.path.isfile(file_path):
            # Open the file
            with open(file_path, 'r') as file:
                # Read the contents of the file
                contents = file.readlines()
                nodes = contents[1:len(contents) - 13]

            # calculate the labels for the input
            solutionsName = filename[0:len(filename) - 4]
            labels = create_labels_simple(solutions_path + solutionsName + "/", len(nodes))

            # get infeasible arcs
            infeasibleEdges = []
            file_path_infeasible = os.path.join(infeasible_arcs_path, filename)
            if os.path.isfile(file_path_infeasible):
                # Open the file
                with open(file_path_infeasible, 'r') as file:
                    # Read the contents of the file
                    for line in file:
                        values = line.split(" ")
                        values = [s for s in values if s != '' and '\n' not in s]
                        infeasibleEdges.append(values)

            # extract necessary data
            vertices = util.extract_vertices(nodes)
            startingPoints = util.extract_startingPoints(contents)
            destinationPoints = util.extract_endingPoints(contents)
            chargingStationPoints = util.extract_charging_stations(contents)
            # iterate over each edge
            for i in range(len(vertices)):
                sortedNeighbourhood = calculate_neighbourhood(vertices[i], vertices)
                for j in range(len(vertices)):
                    if i != j:
                        # if infeasibleEdges[i][j] == 'true':
                        #    continue
                        dataPoint = []
                        if vertices[i] in startingPoints or vertices[i] in destinationPoints or vertices[
                            i] in chargingStationPoints:
                            continue
                        if vertices[j] in startingPoints or vertices[j] in destinationPoints or vertices[
                            j] in chargingStationPoints:
                            continue
                        # nearness of nodes. +1 for the later training of knn which uses the minimum and maximum neighborhood.
                        # Because calculate_nearness is based on index the closest neighbor would have the value 0. Which however
                        # would based on the later code mean a k value of 0, which does not make sense.
                        dataPoint.append(calculate_nearness(vertices[j], sortedNeighbourhood) + 1)
                        dataPoint.append(labels[i, j])
                        inputEdges.append(dataPoint)
    return inputEdges


def create_labels(path, length):
    directory_path = path
    files = os.listdir(directory_path)
    labels = np.zeros([length, length])
    objectiveFunValues = []
    weights = np.zeros(math.floor(len(files)/2))
    for filename in files:
        if filename.endswith('.out'):
            file_path = os.path.join(directory_path, filename)

            # Check if it's a file (and not a directory)
            if os.path.isfile(file_path):
                # Open the file
                with open(file_path, 'r') as file:
                    objectiveFunValue = util.extract_objectiveFun(file)
                    objectiveFunValues.append(objectiveFunValue)
    objectiveFunValuesNp = np.array(objectiveFunValues)
    minObjectiveFunValue = np.min(objectiveFunValues)
    exponents = objectiveFunValuesNp / minObjectiveFunValue
    divisor = np.sum(np.exp(exponents))
    weights = 1 - (np.exp(exponents) / divisor)
    # Extract Edges
    fileCounter = 0
    for filename in files:
        if filename.endswith('.sol'):
            file_path = os.path.join(directory_path, filename)

            # Check if it's a file (and not a directory)
            if os.path.isfile(file_path):
                with open(file_path, 'r') as file:
                    # calculate weightings
                    contents = file.readlines()
                    routes = contents[1:len(contents) - 1]
            for route in routes:
                verticeIndices = util.extract_route(route)
                for i in range(len(verticeIndices)-1):
                    #fileCounter stays the same for one file as one file represents one possible solution.
                    labels[verticeIndices[i]-1, verticeIndices[i+1]-1] = labels[verticeIndices[i]-1, verticeIndices[i+1]-1] + weights[fileCounter]
            fileCounter += 1
    labels[labels <= 5] = 0
    labels[labels > 5] = 1
    print(np.sum(labels))
    return labels

"""
The difference for create_labels_simple is that here the quality of the solution is not taken into account.
 Here there is only one solution that is used and a label is 0 if the edge is not part of the solution and
 1 if the edge is part of the solution.
"""
def create_labels_simple(path, length):
    labels = np.zeros([length, length])
    #file = sorted(os.listdir(path))[1]
    #print(file)
    files = os.listdir(path)
    for filename in files:
        if filename.endswith('.sol'):
            # Create the full path to the file
            file_path = os.path.join(path, filename)

            # Check if it's a file (and not a directory)
            if os.path.isfile(file_path):
                # Open the file
                with open(file_path, 'r') as file:
                    contents = file.readlines()
                    routes = contents[1:len(contents)]
                    for route in routes:
                        verticeIndices = util.extract_route(route)
                        for i in range(len(verticeIndices)-1):
                            labels[verticeIndices[i]-1, verticeIndices[i+1]-1] += 1
    print(np.sum(labels > 20))
    return labels > 20


