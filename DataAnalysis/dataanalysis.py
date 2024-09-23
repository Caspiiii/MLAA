import util
import analysisutil as anu
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import math

########################################################################################################################
##
##                      -- DATA ANALYSIS --
##  Extract and analyse data of both instances and solutions of the EADARP
##
########################################################################################################################



def route_distribution():
    """
       Plot all routes of each instance in the given directory in terms of their
       geographical coordinates.
    """
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
                routes_vertices = anu.extract_route_vertice_coordinates(file_path, instance)
                draw_furthest_quadrilateral(routes_vertices, instance)
                plt.grid(True)
                plt.show()
                break



def order_by_closest(furthest_points):
    """
        Helper function that takes the vertices furthest away from each other and sorts
        them based on their vicinity to the vertice before.
    """
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
    """
        Plots both the geographical location of all vertices for the given instance based on
        their corresponding route and additionally draws a quadrilateral that encases (most)
        vertices of the corresponding route.
    """
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
        #if furthest_points is not None:
        # Close the polygon by repeating the first point at the end
        #    furthest_points_ordered = order_by_closest(furthest_points)
        #    quadrilateral = np.vstack([furthest_points_ordered, furthest_points_ordered[0]])

        # Draw the quadrilateral (polygon)
        #    plt.plot(quadrilateral[:, 0], quadrilateral[:, 1], color=color, label="Furthest quadrilateral")
        #plt.legend()



def pca_analysis_all_routes(routes_vertices):
    """
        Helper function to additionally generate and plot PC1 and PC2 for all other vertices of
        the instance.
    """
    scaler = StandardScaler()
    for i, route_vertices in enumerate(routes_vertices):
        scaled_data = scaler.fit_transform(route_vertices)
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(scaled_data)
        pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
        plt.scatter(pca_df['PC1'], pca_df['PC2'], s=10, color="black")

        # Step 5: Explained Variance Ratio
        explained_variance = pca.explained_variance_ratio_



def pca_analysis():
    """
        Execute PCA Analysis based on currently 4 features: The first two are the location coordinates.
        The third and fourth are the earliest and latest possible service time.
    """
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
                routes_vertices = util.extract_route_vertice_coordinates_and_time_windows(file_path, instance,
                                                                                          directory + ".txt")
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
                    plt.subplot(math.ceil(math.sqrt(len(routes_vertices))),
                                math.floor(math.sqrt(len(routes_vertices) + 2)), i + 1)
                    plt.title('PCA: ' + directory + "; Route: " + str(i + 1))
                    pca_analysis_all_routes(routes_vertices)
                    plt.scatter(pca_df['PC1'], pca_df['PC2'], color=colors[i])
                    plt.grid(True)
                    # Step 5: Explained Variance Ratio
                    explained_variance = pca.explained_variance_ratio_
                    print("Explained Variance Ratio:", explained_variance)
                break
        plt.tight_layout()
        plt.show()


def route_distribution_with_annotations():
    """
    Draws all points of the instance based on the route that they were assigned to. Also assign to each vertice if it
    is a pickup-, dropoff- or other location.
    """
    directories_path = "../out/"
    directories = os.listdir(directories_path)
    for directory in directories:
        files_path = os.path.join(directories_path, directory)
        files = os.listdir(files_path)
        instance_path = "../l1/" + directory + ".txt"
        cmap = mpl.colormaps['tab10']
        colors = cmap(np.linspace(0, 1, 5))
        for filename in files:
            if filename.endswith("sol"):
                solution_path = os.path.join(files_path, filename)
                routes_vertices = anu.extract_route_vertice_coordinates(solution_path, instance_path)
                instance_dims = util.instance_size(instance_path)
                # Number of pickup locations
                n_P = instance_dims["n_P"]
                routes_indices = util.read_solution(solution_path)
                # iterate over all routes
                plt.figure(figsize=(10, 10))
                plt.title(f"Instance " + instance_path.split("/")[2])
                plt.axis("off")
                for i, vertices in enumerate(routes_vertices):
                    route_indices = routes_indices[i]
                    counter = 0
                    plt.subplot(math.ceil(math.sqrt(len(routes_vertices))),
                                math.floor(math.sqrt(len(routes_vertices) + 2)), i+1)
                    plt.grid(True)
                    # iterate over one route of all routes
                    for index in route_indices:
                        if index < n_P:
                            plt.scatter(vertices[counter, 0], vertices[counter, 1], s=100, color="#1f77b4", label="Pickup")
                        elif n_P <= index < 2 * n_P:
                            # plot line between pickup and drop-off location
                            all_vertices = util.verticesStored
                            pickup_location = all_vertices[index - n_P - 1]
                            x_coordinates = [pickup_location[0], vertices[counter][0]]
                            y_coordinates = [pickup_location[1], vertices[counter][1]]
                            plt.plot(x_coordinates, y_coordinates, color='lightgrey', linewidth=5, alpha=0.7)
                            plt.scatter(vertices[counter, 0], vertices[counter, 1], s=100,
                                        color="#ff7f0e", label="Drop-off")
                        else:
                            plt.scatter(vertices[counter:, 0], vertices[counter:, 1], s=100,
                                        color="#2ca02c", label="other")
                        if counter != 0:
                            plt.arrow(vertice_before[0], vertice_before[1], vertices[counter][0] - vertice_before[0],
                                    vertices[counter][1] - vertice_before[1], length_includes_head=1, shape="full",
                                    head_width=0.7, head_length=0.7)

                        vertice_before = vertices[counter]
                        counter += 1
                    #plt.legend()
                plt.tight_layout()
                plt.show()


                break


#route_distribution()
#pca_analysis()
route_distribution_with_annotations()
