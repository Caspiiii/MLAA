import util
import numpy as np
import os
import math




def create_input(path):
    ## Creation of input vector
    directory_path = path
    files = os.listdir(directory_path)
    ## calculate the input vector for each solution
    inputEdges = []
    for filename in files:
        # Create the full path to the file
        file_path = os.path.join(directory_path, filename)
        print("Calculating input for:" + filename)
        # Check if it's a file (and not a directory)
        if os.path.isfile(file_path):
            # Open the file
            with open(file_path, 'r') as file:
                # Read the contents of the file
                contents = file.readlines()
                nodes = contents[1:len(contents) - 13]


            # calculate the labels for the input
            solutionsName = filename[0:len(filename) - 4]
            labels = create_labels("out/" + solutionsName + "/", len(nodes) + 1)
            print(labels)
            print(np.sum(labels))
            print("-" * 80)

            # extract necessary data
            vertices = util.extract_vertices(nodes)
            startingPoints = util.extract_startingPoints(contents)
            startingPointsCenter = util.calculate_centroid(startingPoints)
            destinationPoints = util.extract_endingPoints(contents)
            destinationPointsCenter = util.calculate_centroid(destinationPoints)

            # iterate over each edge
            dataPoint = []
            for i in range(len(vertices)):
                sortedNeighbourhood = calculate_neighbourhood(vertices[i], vertices[i+1:len(vertices)])
                for j in range(i + 1, len(vertices)):
                    # edge length
                    dataPoint.append(math.dist(vertices[i], vertices[j]))
                    # nearness of nodes
                    dataPoint.append(calculate_nearness(vertices[j], sortedNeighbourhood))
                    # distance from start
                    # Typically there are more than one starting point
                    # To look at the distance from the start we take the geometric center of all starting points
                    # For the edge we look at the center of the edge
                    edgeCenter = util.calculate_centroid([vertices[i], vertices[j]])
                    dataPoint.append(math.dist(startingPointsCenter, edgeCenter))
                    # distance from end
                    # same principle as for start
                    dataPoint.append(math.dist(destinationPointsCenter, edgeCenter))
                    dataPoint.append(labels[i,j])
                    #print("Edge: (" + str(i) + ", " + str(j) + "):" + str(dataPoint))
                    #print("-" * 80)
            inputEdges.append(dataPoint)
    # distance from center of gravity
    # alpha distance
    # clusterness
    # charging station edge
    return inputEdges



def create_labels(path, length):
    # Extract Solution
    directory_path = path
    files = os.listdir(directory_path)
    labels = np.zeros([length, length])
    objectiveFunValues = []
    weights = np.zeros(math.floor(len(files)/2))
    for filename in files:
        if filename.endswith('.out'):
            # Create the full path to the file
            file_path = os.path.join(directory_path, filename)

            # Check if it's a file (and not a directory)
            if os.path.isfile(file_path):
                # Open the file
                with open(file_path, 'r') as file:
                    objectiveFunValue = util.extract_objectiveFun(file)
                    objectiveFunValues.append(objectiveFunValue)
    objectiveFunValuesNp = np.array(objectiveFunValues)
    minObjectiveFunValue = min(objectiveFunValues)
    exponents = objectiveFunValuesNp / minObjectiveFunValue
    divisor = sum(np.exp(exponents))
    weights = 1 - (np.exp(exponents) / divisor)
    for i in range(math.floor(len(files)/2)):
        divisor = sum(np.exp(exponents))
        weights = 1 - (np.exp(exponents)/divisor)
    # Extract Edges
    for filename in files:
        if filename.endswith('.sol'):
            # Create the full path to the file
            file_path = os.path.join(directory_path, filename)

            # Check if it's a file (and not a directory)
            if os.path.isfile(file_path):
                # Open the file
                with open(file_path, 'r') as file:
                    # calculate weightings
                    contents = file.readlines()
                    routes = contents[1:len(contents) - 1]
            for route in routes:
                verticeIndices = util.extract_route(route)
                for i in range(len(verticeIndices)-1):
                    #print(len(labels))
                    #print(len(verticeIndices))
                    labels[verticeIndices[i],verticeIndices[i+1]] = labels[verticeIndices[i],verticeIndices[i+1]] + weights[i]
    return labels > 0


def calculate_neighbourhood(vertice, neighbourhood):
    v = vertice
    def sortByDist(verticeToCompare):
        return math.dist(v, verticeToCompare)
    return sorted(neighbourhood, key = sortByDist)

def calculate_nearness(neighbour, neighbourhood):
        return neighbourhood.index(neighbour)

