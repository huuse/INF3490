#exhaustive.py

from TSP_data import *

from itertools import permutations

try:
	size = int(sys.argv[1])
except:
	print 'USAGE: python <exhaustive.py> <number of cities>'
	sys.exit()

starttime= time.time()

cities,distance = TSP_data()
smallcity,citieslist = smallcity(size,distance)

besttrip = 0 								#Creating the first besttrip (shortest trip atempt)
for b in range(0,len(smallcity)-1):

	besttrip += smallcity[b][b+1]
besttrip += smallcity[0][len(smallcity)-1]

m = citieslist




for i in permutations(citieslist):
	testtrip1 = TSP_route_length(i,smallcity)
	if testtrip1<besttrip:
		besttrip = testtrip1
		m = i

citynames = TSP_route_2_names(m,cities)

print 'shortest path = \n %s \n length = %.3f km' %(citynames,besttrip)
stoptime=time.time()-starttime
print 'time = ', stoptime



