import numpy as np

verticesStored = np.array([])


def calculate_centroid(points):
    # Initialize sums
    sum_x = 0
    sum_y = 0

    # Number of points
    n = len(points)

    # Sum all x and y coordinates
    for i in range(len(points)):
        sum_x += points[i][0]
        sum_y += points[i][1]

    # Calculate the centroid
    centroid_x = sum_x / n
    centroid_y = sum_y / n

    return centroid_x, centroid_y


def extract_startingPoints(instance):
    startingPointIndex = instance[len(instance) - 13]
    startingPointsIndex = [eval(i) for i in instance[len(instance) - 11].split()]
    startingPointsIndex.append(startingPointIndex)
    startingPoints = []
    for i in range(len(startingPointsIndex)):
        startingPoints.append(verticesStored[i])
    return startingPoints


def extract_endingPoints(instance):
    endingPointIndex = instance[len(instance) - 12]
    endingPointsIndex = [eval(i) for i in instance[len(instance) - 10].split()]
    endingPointsIndex.append(endingPointIndex)
    endingPoints = []
    for i in range(len(endingPointsIndex)):
        endingPoints.append(verticesStored[i])
    return endingPoints


def extract_objectiveFun(solution):
    objectiveFunction = find_lines_with_substring(solution, "T best obj:")
    if (len(objectiveFunction) != 1):
        return -1
    return float(objectiveFunction[0].split()[3])


def extract_vertices(instance):
    vertices = []
    for each in instance:
        preCoord = each.split()
        xCoord = float(preCoord[1])
        yCoord = float(preCoord[2])
        vertices.append((xCoord, yCoord))
    global verticesStored
    verticesStored = vertices
    return vertices


def find_lines_with_substring(solution, substring):
    result = []
    for line in solution:
        if substring in line:
            result.append(line)
    return result


def extract_route(route):
    verticeIndeces = []
    splitted = route.split(", ")
    verticeIndeces.append(int(splitted[0][2:]))
    withoutFirst = splitted[1:len(splitted) - 1]
    for s in withoutFirst:
        if "]" in s:
            verticeIndeces.append(int(s[0:len(s) - 2]))
            break
        verticeIndeces.append(int(s))
    return verticeIndeces
