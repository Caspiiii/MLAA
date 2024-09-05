import util
import numpy as np
import os
import math
import matplotlib.pyplot as plt

def create_input(path):
    ## Creation of input vector
    directory_path = path
    files = os.listdir(directory_path)
    ## calculate the input vector for each solution
    inputEdges = []
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
            labels = create_labels_simple("out/" + solutionsName + "/", len(nodes))
            #print(labels[0])
            #print(np.sum(labels))
            #print("-" * 80)



            #get infeasible arcs
            infeasibleEdges = []
            file_path_infeasible = os.path.join("infeasibleArcs/", filename)
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
            timeWindows = util.extract_timeWindow(filename)
            startingPoints = util.extract_startingPoints(contents)
            startingPointsCenter = util.calculate_centroid(startingPoints)
            destinationPoints = util.extract_endingPoints(contents)
            destinationPointsCenter = util.calculate_centroid(destinationPoints)
            chargingStationPoints = util.extract_ChargingStations(contents)
            # iterate over each edge
            for i in range(len(vertices)):
                sortedNeighbourhood = calculate_neighbourhood(vertices[i], vertices)
                for j in range(len(vertices)):
                    if infeasibleEdges[i][j] == 'true':
                        continue
                    dataPoint = []
                    if vertices[i] in startingPoints or vertices[i] in destinationPoints or vertices[i] in chargingStationPoints:
                        continue
                    if vertices[j] in startingPoints or vertices[j] in destinationPoints or vertices[j] in chargingStationPoints:
                        continue
                    print("Calculating Edge Length")
                    print("-----" * 40)
                    if i != j:
                        # edge length
                        if (math.dist(vertices[i], vertices[j])) == 0:
                            dataPoint.append(100)
                        else:
                            dataPoint.append(math.dist(vertices[i], vertices[j]))
                        print("Calculating Nearness of Nodes")
                        print("-----" * 40)
                        # nearness of nodes
                        #dataPoint.append(calculate_nearness(vertices[j], sortedNeighbourhood))
                        # distance from start
                        # Typically there are more than one starting point
                        # To look at the distance from the start we take the geometric center of all starting points
                        # For the edge we look at the center of the edge
                        print("Calculating Distance from Start")
                        print("-----" * 40)
                        """
                        edgeCenter = util.calculate_centroid([vertices[i], vertices[j]])
                        minimumStartingPoint = util.findSmallestDistance(startingPoints, edgeCenter)
                        maximumStartingPoint = util.findBiggestDistance(startingPoints, edgeCenter)
                        minimumDestinationPoint = util.findSmallestDistance(destinationPoints, edgeCenter)
                        maximumDestinationPoint = util.findBiggestDistance(destinationPoints, edgeCenter)
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
                        print("Calculating Clusterness")
                        print("-----" * 40)
                        # density
                        #
                        #dataPoint.append(calculate_clusterness(vertices[i], vertices, 5))
                        #dataPoint.append(math.dist(vertices[i], vertices[j]) <= 5)
                        print("Calculating Charging Station Edge")
                        print("-----" * 40)
                        # charging station edge
                        """
                        dataPoint.append(vertices[i] in chargingStationPoints)
                        dataPoint.append(vertices[j] in chargingStationPoints)
                        dataPoint.append(vertices[j] in chargingStationPoints and vertices[i] in chargingStationPoints)
                        """
                        #print("Calculating Alpha-Nearness")
                        #print("-----" * 40)
                        #alphanearness
                        #dataPoint.append(util.calculatealphaNearness(i, j))
                        #dataPoint.append(util.calculateMinimumSpanningTreeWithEdge(i, j))
                        #dataPoint.append(util.calculateMinimumSpanningTree())

                        print("Calculating Time Windows")
                        print("-----" * 40)
                        #timewindows
                        if (vertices[j] in chargingStationPoints or vertices[j] in startingPoints or vertices[j] in destinationPoints):
                            dataPoint.append(0)
                            #dataPoint.append(0)
                            #dataPoint.append(0)
                            #dataPoint.append(0)
                            #plt.scatter(i, timeWindows[i][0] - timeWindows[j][0], color='black',
                             #           label='Single Point')

                        else:
                            dataPoint.append(
                                ((timeWindows[i][1] - timeWindows[j][1]) + (timeWindows[i][0] - timeWindows[j][0])) / 2)
                            #dataPoint.append(timeWindows[i][1] - timeWindows[j][1])
                            #dataPoint.append(timeWindows[i][0] - timeWindows[j][0])
                            #dataPoint.append(timeWindows[i][1] - timeWindows[j][0])
                        #if (labels[i, j] == 1):
                            #plt.scatter(j, timeWindows[i][0] - timeWindows[j][0],color='red',
                            #        label='Single Point')
                        #else:
                            #plt.scatter(i, timeWindows[i][0] - timeWindows[j][0], color='grey',
                            #            label='Single Point')
                        dataPoint.append(labels[i, j])
                        inputEdges.append(dataPoint)

            inputVector = np.array(inputEdges)
            """
            y = []
            X = []

           
            with open(filename + " output.txt", "w") as txt_file:
                for line in labels:
                    txt_file.write(str(len(line)) + " ")
                    txt_file.write(str(np.sum(line)) + " ")
                    txt_file.write(np.array2string(line) + "\n")
            print(len(inputVector))
            print(len(inputVector[0]))
            for j in range(len(inputVector)):
                for i in range(len(inputVector[j])):
                    if (i + 1) % 22 == 0:
                        y.append(inputVector[j][i])

            for i in range(len(inputVector)):
                dataPoint = []
                for j in range(len(inputVector[i])):
                    if (j + 1) % 22 != 0:
                        dataPoint.append(inputVector[i][j])
                    if ((j + 1) % 22 == 0):
                        X.append(dataPoint)
                        dataPoint = []
            X = np.array(X)
            y = np.array(y)

            # Step 3: Show the plot
            plt.title(filename)
            plt.show()
            """
            """
            plt.scatter(inputVector[:, 19::22], inputVector[:, 21::22], color='black', label='Actual Data')
            plt.xlabel('Feature')
            plt.ylabel('Target')
            plt.title('SVR Predictions vs Actual')
            plt.legend()
            plt.show()
            """
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
    #for i in range(math.floor(len(files)/2)):
     #   divisor = np.sum(np.exp(exponents))
    #  weights = 1 - (np.exp(exponents)/divisor)
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
                    #print(len(labels))
                    #print(len(verticeIndices))
                    #fileCounter stays the same for one file as one file represents one possible solution.
                    labels[verticeIndices[i]-1, verticeIndices[i+1]-1] = labels[verticeIndices[i]-1, verticeIndices[i+1]-1] + weights[fileCounter]
            fileCounter += 1
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
                with open(path[len(path) - 5:len(path) - 1] + " output.txt", "w") as txt_file:
                    for route in routes:
                        verticeIndices = util.extract_route(route)
                        for i in range(len(verticeIndices)-1):
                            labels[verticeIndices[i]-1, verticeIndices[i+1]-1] = 1
                        #labels[verticeIndices[-1]-1, verticeIndices[0]-1] = 1
                        for line in verticeIndices:
                            txt_file.write(str(line) + ", ")
                        txt_file.write("\n")
    return labels
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

