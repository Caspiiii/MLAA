import util
import os
import matplotlib.pyplot as plt
import math
# Example list with gaps (some elements missing)
directories_path = "out/"
directories = os.listdir(directories_path)
max_dist = 0
for directory in directories:
    files_path = os.path.join(directories_path, directory)
    files = os.listdir(files_path)
    for filename in files:
        if filename.endswith("sol"):
            file_path = os.path.join(files_path,filename)
            with open(file_path, 'r') as file:
                # calculate weightings
                contents = file.readlines()
                routes = contents[1:len(contents) - 1]
                routeCounter = 1
            for route in routes:
                verticeIndices = util.extract_route(route)
                instance = "l1/" + directory + ".txt"
                # the origindepots have the lowest ids bigger than the ids of the actual requests. Therefore taking the
                # lowest id of the startingpoints gives the amount of pickup and dropoff locations combined.
                with open(instance, 'r') as file:
                    contents = file.readlines()
                startingPoints = util.extract_first_startingPoints_index(contents)
                n = startingPoints - 1
                # Create pairs between i and i + n//2 (e.g., 1 <-> 6, 2 <-> 7, etc.)
                half_n = math.floor(n/2)
                pairs = [(i, i + half_n) for i in range(1, half_n + 1)]  # Generate pairs like (1, 6), (2, 7), etc.

                # Find distances between pairs
                distances = []

                for pair in pairs:
                    element1, element2 = pair  # e.g. (1, 6), (2, 7), etc.

                    # Check if both elements of the pair are present in the list
                    if element1 in verticeIndices and element2 in verticeIndices:
                        # Find their positions in the list
                        index1 = verticeIndices.index(element1)
                        index2 = verticeIndices.index(element2)

                        # Calculate the number of elements between them
                        dist = abs(index1 - index2) - 1  # Distance between the indices minus 1 (for in-between elements)
                        distances.append(dist)
                    else:
                        # Skip this pair if either element is missing
                        distances.append(None)

                # Output the distances
                print("Distances between pairs (None if missing):", distances)

                # Plotting the distances, excluding missing pairs
                valid_distances = [dist for dist in distances if dist is not None]

                for item in valid_distances:
                    if item > max_dist:
                        max_dist = item

                plt.plot(valid_distances, marker='o')
                plt.title("Number of Elements Between Pick-up/Drop-off Pairs (" + directory + ", routeNr.: "+ str(routeCounter) + ")")
                plt.xlabel("Valid Pair Index")
                plt.ylabel("Number of Elements Between")
                plt.grid(True)
                plt.show()
                routeCounter += 1

        break

print("Max Distance:", max_dist)