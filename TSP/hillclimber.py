#hillclimber.py
from TSP_data import *
import numpy as np


try:
	size = int(sys.argv[1])

	iterations = int(sys.argv[2])
except:
	print 'USAGE: python <hillclimber.py> <number of cities (minimum 2)> <iterations (minimum 1)>'
	sys.exit(1)

if size <2 or iterations <1:
	print 'USAGE: python <hillclimber.py> <number of cities (minimum 2)> <iterations (minimum 1)>'
	sys.exit(1)



starttime = time.time()

###############################################		DECLARE		###############################################
runs = 20
###############################################		DECLARE DONE		###############################################

cities,distance = TSP_data()
smallcity,citieslist = smallcity(size,distance)



def hill(shufflecity,iterations):
	besttrip = TSP_route_length(shufflecity,smallcity)
	m = shufflecity.copy()
	count = 0
	while count <iterations:
		count2 = 0
		while count2 ==0:
			a = np.random.randint(len(shufflecity))
			b = np.random.randint(len(shufflecity))
			if a!=b:
				count2=1
		value1 = shufflecity[a]
		value2 = shufflecity[b]
		shufflecity[b] = value1
		shufflecity[a] = value2
		testtrip = TSP_route_length(shufflecity,smallcity)
		if testtrip < besttrip:
			m = shufflecity.copy()
			besttrip = testtrip
		else:
			shufflecity = m.copy()
		count +=1
	return besttrip,m

shufflecity = citieslist
np.random.shuffle(shufflecity)

list_besttrip = np.zeros((runs,1))
list_besttour = np.zeros((runs,size))
for run in range(0,runs):

	besttrip,best_tour = hill(shufflecity,iterations)

	list_besttrip[run] = besttrip
	for i in range(size):
		list_besttour[run][i] = best_tour[i]


citynames = TSP_route_2_names(best_tour,cities)
print 'after %d runs: shortest path = \n %s \n length = %.3f km' %(runs,citynames,besttrip)

stoptime=time.time()-starttime
print 'time = ', stoptime


#"""
print 'list_besttour = \n',list_besttour
print 'list_besttrip = \n',list_besttrip

TOTAL_best_trip_position = np.argmin(list_besttrip)
TOTAL_best_trip = list_besttrip[TOTAL_best_trip_position]
print '\nTOTAL_best_trip = \n',TOTAL_best_trip

WORST_of_runs_besttrip_position = np.argmax(list_besttrip)
WORST_of_runs_besttrip = list_besttrip[WORST_of_runs_besttrip_position]
print '\nWORST_of_runs_besttrip = \n', WORST_of_runs_besttrip

average_runs_besttrip = np.mean(list_besttrip, axis=0)
print '\naverage_runs_besttrip = \n', average_runs_besttrip

temp_ST = 0  													#Standard Deviation temp
for i in range(list_besttrip.shape[0]):
	temp_ST += (list_besttrip[i]-average_runs_besttrip)**2
ST = np.sqrt(temp_ST/list_besttrip.shape[0])
print '\nStandard Deviation = \n',ST
#"""
