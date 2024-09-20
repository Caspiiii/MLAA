import util
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import math

"""
directories_path = "out/"
directories = os.listdir(directories_path)[3:]
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
                    nodes = contents[1:len(contents) - 13]
                startingPoints = util.extract_first_startingPoints_index(contents)
                vertices = util.extract_vertices(nodes)


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
            """



def route_distribution():
    directories_path = "out/"
    directories = os.listdir(directories_path)
    for directory in directories:
        files_path = os.path.join(directories_path, directory)
        files = os.listdir(files_path)
        instance = "l1/" + directory + ".txt"
        for filename in files:
            if filename.endswith("sol"):
                plt.figure(figsize=(10, 10))
                plt.title(f"Instance " + instance.split("/")[1])
                file_path = os.path.join(files_path, filename)
                routes_vertices = util.extract_route_vertice_coordinates(file_path, instance)
                draw_furthest_quadrilateral(routes_vertices, instance)
                plt.grid(True)
                plt.show()
                break


def order_by_closest(furthest_points):
    furthest_points = np.array(furthest_points)

    # Start with the first point (you can choose any point to start with)
    ordered_points = [furthest_points[0]]
    remaining_points = np.copy(furthest_points[1:])

    # Greedily pick the next closest point
    while remaining_points.shape[0] > 0:
        last_point = ordered_points[-1]
        # Compute distances from the last point to all remaining points
        distances = np.linalg.norm(remaining_points - last_point, axis=1)
        # Find the index of the closest point
        closest_index = np.argmin(distances)
        # Add the closest point to the ordered list
        closest_point = remaining_points[closest_index]
        ordered_points.append(closest_point)
        # Remove the closest point from the remaining points
        remaining_points = np.delete(remaining_points, closest_index, axis=0)
    return np.array(ordered_points)


def draw_furthest_quadrilateral(routes_vertices, name):
    cmap = mpl.colormaps['tab10']
    colors = cmap(np.linspace(0, 1, len(routes_vertices)))
    for i, vertices in enumerate(routes_vertices):
        color = colors[i]  # Get the color for this plot
        #plt.subplot(math.ceil(math.sqrt(len(routes_vertices))),
        #            math.floor(math.sqrt(len(routes_vertices) + 2)),
        #            i + 1)
        #plt.scatter(np.array(util.verticesStored)[:, 0], np.array(util.verticesStored)[:, 1], s=5, color="black")
        if len(vertices) < 4:
            print(f"Not enough vertices in ndarray {i + 1} to form a quadrilateral")
            continue

        max_distance_sum = 0
        furthest_points = None

        # Get all combinations of 4 points to form a quadrilateral
        for points in combinations(vertices, 4):
            # Calculate total distance between all pairs of the 4 selected points
            distance_sum = sum(np.linalg.norm(p1 - p2) for p1, p2 in combinations(points, 2))

            # Update if this combination has the largest total distance
            if distance_sum > max_distance_sum:
                max_distance_sum = distance_sum
                furthest_points = np.array(points)

        # Draw the area (quadrilateral) between the four furthest apart points


        # Plot the points of the current route
        plt.scatter(vertices[:, 0], vertices[:, 1], color=color, label="Vertices")

        # Plot the quadrilateral if the furthest points were found
        if furthest_points is not None:
            # Close the polygon by repeating the first point at the end
            furthest_points_ordered = order_by_closest(furthest_points)
            quadrilateral = np.vstack([furthest_points_ordered, furthest_points_ordered[0]])

            # Draw the quadrilateral (polygon)
            plt.plot(quadrilateral[:, 0], quadrilateral[:, 1], color=color, label="Furthest quadrilateral")
        #plt.legend()

"""
    Helper function to additionally generate and plot PC1 and PC2 for all other vertices of 
    the instance. 
"""
def pca_analysis_all_routes(routes_vertices):
    scaler = StandardScaler()
    for i, route_vertices in enumerate(routes_vertices):
        scaled_data = scaler.fit_transform(route_vertices)
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(scaled_data)
        pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
        plt.scatter(pca_df['PC1'], pca_df['PC2'], s=10,  color="black")

        # Step 5: Explained Variance Ratio
        explained_variance = pca.explained_variance_ratio_
"""
    Execute PCA Analysis based on currently 4 features: The first two are the location coordinates.
    The third and fourth are the earliest and latest possible service time. 
"""
def pca_analysis():
    directories_path = "out/"
    directories = os.listdir(directories_path)[3:]

    for directory in directories:
        plt.figure(figsize=(10, 10))
        #
        #plt.xlabel('Principal Component 1')
        #plt.ylabel('Principal Component 2')
        files_path = os.path.join(directories_path, directory)
        files = os.listdir(files_path)
        instance = "l1/" + directory + ".txt"
        for filename in files:
            if filename.endswith("sol"):
                file_path = os.path.join(files_path, filename)
                routes_vertices = util.extract_route_vertice_coordinates_and_time_windows(file_path, instance, directory + ".txt")
                cmap = mpl.colormaps['Set3']
                colors = cmap(np.linspace(0, 1, len(routes_vertices)))
                scaler = StandardScaler()
                for i, route_vertices in enumerate(routes_vertices):
                    scaled_data = scaler.fit_transform(route_vertices)

                    # Step 2: Perform PCA
                    pca = PCA(n_components=2)  # Reduce to 2 dimensions (components)
                    principal_components = pca.fit_transform(scaled_data)

                    # Step 3: Create a DataFrame with the principal components
                    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
                    # TODO: generalize
                    plt.subplot(math.ceil(math.sqrt(len(routes_vertices))), math.floor(math.sqrt(len(routes_vertices)+2)), i+1)
                    plt.title('PCA: ' + directory + "; Route: " + str(i+1))
                    pca_analysis_all_routes(routes_vertices)
                    plt.scatter(pca_df['PC1'], pca_df['PC2'], color=colors[i])
                    plt.grid(True)
                    # Step 5: Explained Variance Ratio
                    explained_variance = pca.explained_variance_ratio_
                    print("Explained Variance Ratio:", explained_variance)
                break
        plt.tight_layout()
        plt.show()



route_distribution()
#pca_analysis()
