#TSP_data.py


import numpy as np
import csv
import sys
import time


def TSP_data():
	"""
	return: list with all the city
	names and matrix with all the distances
	"""
	file = open('european_cities.csv', 'r+')
	reader = csv.reader(file, delimiter = ';')
	i = 0
	distance = np.zeros((24,24))
	cities = []
	for row in reader:
   		if i == 0:
   	   		for j in range(len(row)):
   	   			cities.append(row[j])
   		else:
   	   		for j in range(len(distance)):
   	   			distance[i-1][j] = float(row[j])
 		i += 1
 	return cities , distance


def smallcity(size,distance):
	"""
	return: matrix of distances and
	the "number" of each city
	"""
	smallcity = distance[0:size,0:size]
	citieslist = np.zeros(len(smallcity))
	for i in range(len(smallcity)):
		citieslist[i] = i
	return smallcity,citieslist

def TSP_route_2_names(m,cities):
	"""
	return: citynames based on the numer
	sequence m
	"""
	citynames = []
	for k in range(len(m)):
		citynames.append(cities[int(m[k])])
	return citynames





def TSP_route_length(route,smallcity):
	"""
	return: the length of the input route
	"""
	testtrip = np.zeros(len(smallcity))

	for i in range(len(smallcity)-1):
		testtrip[i] = smallcity[route[i],route[i+1]]
	testtrip[-1] = smallcity[route[-1],route[0]]
	return np.sum(testtrip)




