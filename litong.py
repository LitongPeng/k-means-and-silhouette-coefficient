"""

Litong Peng
02/14/2021

"""
import csv

import matplotlib.pyplot as plt
import random
import numpy as np
from math import inf, sqrt


# load the data from csv file
def load_data(file):
    csv_file = csv.reader(open(file, 'r'))
    data = []
    for row in csv_file:
        # the first line is string
        if csv_file.line_num == 1:
            continue
        temp = []
        for point in row:
            temp.append(float(point))
        data.append(temp)
    # the first line is attributes
    return data[1:]


# generate k random centroids
def random_centroids(data, k):
    # the maximum value from data
    max_v = np.max(data)
    # the minimum value from data
    min_v = np.min(data)
    centroids = []
    for i in range(k):
        temp = []
        for j in range(len(data[0])):
            # choose a random number in the range from minimum to maximum
            temp.append(random.uniform(min_v, max_v))
        centroids.append(temp)
    return centroids


# find the nearest centroid for a point
def nearest(point, centroids):
    min_distance = float(inf)
    for i in range(len(centroids)):
        distance = euclidean(centroids[i], point)
        if distance < min_distance:
            min_distance = distance
            index = i
    return index


# calculate the euclidean distance for two points
def euclidean(centroids, point):
    s = 0
    for i in range(len(centroids)):
        s += pow((centroids[i] - point[i]), 2)
    return sqrt(s)


# calculate the new centroids
def update_centroids(data, clusters):
    new_centroids = []
    # for each centroid
    for cluster in clusters:
        # if this cluster is not empty
        if len(cluster) != 0:
            temp = []
            for i in range(len(cluster[0])):
                sum = 0
                for points in cluster:
                    sum += points[i]
                mean = sum / len(cluster)
                temp.append(mean)
            new_centroids.append(temp)
        # if no item is assigned to this group
        else:
            n = list(random.choice(data))
            new_centroids.append(n)
    return new_centroids


# k-means algorithm
def k_means(data, k):
    # generate random k centroids
    centroids = random_centroids(data, k)
    # determine whether the centroids has changed or not
    centroids_change = True
    # calculate the iteration times
    i = 0
    # initiate the group will be assigned items
    clusters = [[] for _ in range(k)]
    # Assign every item to its nearest centroids
    while centroids_change == True or i < 100:
        centroids_change = False
        for point in data:
            # find the centroids' index of every item
            index = nearest(point, centroids)
            # add the items belong to same index(same centroids) to the cluster group
            clusters[index].append(point)
        # update centroids
        new_centroids = update_centroids(data, clusters)
        # if centroids changed
        if new_centroids == centroids:
            # end loop
            centroids_change = False
        else:
            # next loop
            centroids = new_centroids
        i += 1
    # calculate the sse
    sse = 0
    for cen, clus in zip(centroids, clusters):
        temp = 0
        for p in clus:
            temp += pow(euclidean(p, cen), 2)
        sse += temp
    return sse, clusters


def k_means_sc(data, k):
    sse, clusters = k_means(data, k)
    # calculate the silhouette coefficient
    si = 0
    # for each clusters
    for a in range(len(clusters)):
        # for each points in a cluster
        for pts in range(len(clusters[a])):
            # calculate a(i) value for each point in each cluster
            sum_a = 0
            # all points in same cluster
            for pts_a in range(len(clusters[a])):
                # if it is not the same point
                if pts_a != pts:
                    sum_a += euclidean(clusters[a][pts], clusters[a][pts_a])
            ai = sum_a / len(clusters[a])
            # calculate c(i) value for each point in each cluster
            bi = float(inf)
            temp_sum = 0
            # all clusters
            for c in range(len(clusters)):
                # if it is not the same cluster
                if c != a:
                    for pts_c in range(len(clusters[c])):
                        temp_sum += euclidean(clusters[a][pts], clusters[c][pts_c])
                    ci = temp_sum / len(clusters[c])
                    if ci < bi:
                        bi = ci
            si += (bi - ai) / max(ai, bi)
    sc = si / len(data)
    return sse, sc


# the main program
# which implement k means algorithm,
# and use sse and silouette coefficient to determine cluster quality
def main():
    file = input("please enter your file path")
    # file = '/Users/penglitong/Desktop/Clustering.csv'
    data = load_data(file)
    x = []
    y = []
    sc_list = []
    for k in range(2, 21, 2):
        x.append(k)
        # I found the knee by sse when k = 8
        # and I used Weka to find optimal k is 12
        if k == 8 or k == 12:
            sse, sc = k_means_sc(data, k)
            sc_list.append(sc)
        else:
            sse, unimportant = k_means(data, k)
        y.append(sse)
    # plot the sse for k means
    plt.xlabel('K')
    plt.ylabel('SSE')
    plt.title('Simple k means')
    plt.plot(x, y)
    plt.show()
    # print the silhouette coefficient of k=8 and 12
    print(
        'The silhouette coefficient of k=8 for k means and k=12 for EM are' + str(sc_list[0]) + 'and' + str(sc_list[1]))


if __name__ == '__main__':
    main()
