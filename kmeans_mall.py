# ==========================================
# TASK 02
# K-Means Clustering (Pure Python)
# ==========================================

import random
import math

# Sample Mall Customer Data
# [Annual Income, Spending Score]
data = [
    [15, 39], [15, 81], [16, 6], [16, 77], [17, 40],
    [18, 76], [18, 6], [19, 94], [19, 3], [20, 72],
    [21, 35], [21, 66], [22, 14], [23, 99], [24, 12],
    [25, 78], [28, 32], [30, 85], [33, 4], [35, 60]
]

# Number of clusters
k = 3

# Randomly initialize centroids
centroids = random.sample(data, k)

def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# Run K-Means
for iteration in range(10):

    clusters = [[] for _ in range(k)]

    # Assign points to nearest centroid
    for point in data:
        distances = [euclidean_distance(point, centroid) for centroid in centroids]
        min_distance_index = distances.index(min(distances))
        clusters[min_distance_index].append(point)

    # Update centroids
    new_centroids = []
    for cluster in clusters:
        if cluster:
            avg_x = sum([point[0] for point in cluster]) / len(cluster)
            avg_y = sum([point[1] for point in cluster]) / len(cluster)
            new_centroids.append([avg_x, avg_y])
        else:
            new_centroids.append(random.choice(data))

    centroids = new_centroids

# Print Results
print("K-Means Clustering Completed!\n")

for i, cluster in enumerate(clusters):
    print(f"Cluster {i+1}:")
    for point in cluster:
        print(point)
    print()
