#genetic.py
from TSP_data import *
import numpy as np
import random


try:
	size = int(sys.argv[1])

	populationsize = int(sys.argv[2])

	NoG = int(sys.argv[3])
except:
	print 'USAGE: python <hillclimber.py> <number of cities (minimum 2)> \n  \
	<populations size (minimum 1)> <generations (minimum 1)>'
	sys.exit(1)

if size <2 or populationsize <1:
	print 'USAGE: python <hillclimber.py> <number of cities (minimum 2)> \n \
	<populations size (minimum 1)> <generations (minimum 1)>'
	sys.exit(1)

starttime = time.time()

cities,distance = TSP_data()
smallcity,citieslist = smallcity(size,distance)

###############################################		DECLARE		###############################################
tournamentsize = 10
runs = 20
probability = 0.3
###############################################		DECLARE	DONE	###############################################

if not populationsize % tournamentsize == 0 or not tournamentsize>=3:
	print 'change population size (currently %d) or declare a new tournamentsize \n \
	(curently %d), so they can be divided \n (tournamentsize minimum 3)' \
	%(populationsize, tournamentsize)
	sys.exit(1)


def mk_population(size,citieslist):
	"""
	Return: a population dimensioned by the
	variable size
	"""
	population = np.zeros((size, len(citieslist)))
	for i in range(0,size):
		shufflecity = citieslist
		np.random.shuffle(shufflecity)
		population[i] = shufflecity



	return population




def mk_tournament(tournamentsize,population):
	"""
	Returns: Best parents and worst individual
	in each tournament in the current population.
	2 parents and 1 worst individual for each tournament
	"""

	tournamentlist = np.zeros(( population.shape[0]/tournamentsize,tournamentsize))

	temp = np.arange(population.shape[0], dtype=np.int)

	np.random.shuffle(temp)
	count = 0
	for i in range(0,int(population.shape[0]/tournamentsize)):
		for j in range(0,int(tournamentsize)):
			tournamentlist[i][j] = temp[count]
			count +=1


	parents = np.zeros((tournamentlist.shape[0],2))

	temp_deletelist = np.zeros(tournamentlist.shape[0])
	for i in range(0,tournamentlist.shape[0]):
		temp_parentlist = np.zeros(tournamentlist.shape[1])

		for j in range(0,tournamentlist.shape[1]):

			np.put(temp_parentlist, [j], [TSP_route_length(population[tournamentlist[i][j]],smallcity)])



		k = np.argpartition(temp_parentlist,-2)[:2] 			#finding the best parents
		temp_deletelist[i] = np.argpartition(temp_parentlist,-2)[-1:]
		parents[i][0] = tournamentlist[i][k[0]]
		parents[i][1] = tournamentlist[i][k[1]]


	deletelist = np.zeros(temp_deletelist.shape[0])				#CREATING THE FINAL DELETELIST WITH
	for i in range(len(temp_deletelist)): 						#REAL POSITION FROM TOURNAMENT LIST AND SORTING IT

		deletelist[i] = tournamentlist[i][temp_deletelist[i]]

	deletelist = sorted(deletelist, reverse=False)


	return parents,deletelist				####Best two candidatets (parents) and worst (deletelist)


#################################		MK_CROSSOVER		#################################

def mk_crossover(population,parents):
	"""
	Return: Children from the parents from each tournament.
	"""
	count = 0
	children = np.zeros((parents.shape[0], population.shape[1]))
	for i in range(parents.shape[0]):

		p1 = population[parents[i][0]]			 	#parent 1

		p2 = population[parents[i][1]]				#parent 2

		sizep1 = len(p1)							#length of parent 1

		rssize = np.random.randint(sizep1) + 1		#size of segment from parent 1
		posop = sizep1-rssize + 1					#number of optional placements of segment into p2

		posp2 = np.random.randint(posop) 			#position of segment from p1 into p2


		seg = []									#creating the segment list
		for i in range(posp2,posp2+rssize):			#for loop to add the segment parts from parent1
			seg.append(p1[i])
		j=0											#counter for loop counting segment position
		child = p2 									#creating a children to crossover too
		for i in range(posp2,posp2+rssize):			#for loop to insert segment into p2

			if seg[j] in p2:						#checking and moving if elemt already exist in p2 coming from segment
				mt = np.nonzero(p2 == seg[j])
				p2[mt] = p2[i]

			child[i] = seg[j]

			j += 1

		for m in range(len(child)):  				#INSERTING CHILD TO CHILDREN
			children[count][m] = child[m]
		count +=1
	return children

#################################		DONE		#################################






#################################		MUTATION		#################################


def mk_mutation(children,probability):  					#TAKES IN CHILDREN ARRAY AND PROBABILITY FOR MUTATION
 	"""
 	Return: Mutaded child(ren)
 	(with a given probability for each child)

 	"""
	for i in range(children.shape[0]):
		if random.random()<=probability:


			segsize = np.random.randint(children.shape[1])+1
			position = np.random.randint(children.shape[1]-segsize+1)
			endpos = position + segsize
			segment = children[i][position:endpos]
			reverse_segment = segment[::-1]
			children[i][position:endpos] = reverse_segment

	return children



#################################		DONE		#################################



#################################		NEWPOPULATION		#################################


def mk_newpopulation(population, children, deletelist):
	"""
	Return: New population where the new children
	are included and the worst individuals
	are removed
	"""
	newpopulation = np.zeros(population.shape)
	count = 0
	for i in range(newpopulation.shape[0]):
		if i not in deletelist:
			for j in range(newpopulation.shape[1]):
				newpopulation[i][j] = population[i][j]

		if float(i) in deletelist:
			for m in range(population.shape[1]):
				newpopulation[i][m] = children[count][m]
			count +=1


	return newpopulation




#################################		DONE		#################################

#################################		GENETIC ALGORITM RUNNING CODE		#################################


starttime= time.time()
temp_AiEG = np.zeros((runs,NoG))   					#Gathers the best from each generation.
													#Average for Each Generation (across runs)
													#NOT AVERAGED YET


runs_besttrips = np.zeros((runs, 1))   						#Collects the shortes trip at the end of each NoG at each runs

for run in range(0,runs):
	BRiEG = np.zeros((NoG, size)) 							#Best Route in Each Generation
	BLiEG = np.zeros(NoG) 									#Best Length in Each Generation
	population = mk_population(populationsize,citieslist)

	for i in range(NoG):


		parents,deletelist= mk_tournament(tournamentsize,population)
		children=  mk_crossover(population,parents)
		children =  mk_mutation(children,probability)

		population = mk_newpopulation(population, children, deletelist)
		besttrip = TSP_route_length(population[0],smallcity)
		m = population[0]
		for j in range(population.shape[0]):
			testtrip1 = TSP_route_length(population[j],smallcity)
			if testtrip1<besttrip:
				besttrip = testtrip1
				m = population[j]
		for k in range(population.shape[1]):
			BRiEG[i][k] = m[k]

			BLiEG[i] = besttrip

	for i in range(NoG):

		temp_AiEG[run][i] = BLiEG[i]







	besttrip = TSP_route_length(population[0],smallcity)
	for i in range(population.shape[0]):
		testtrip1 = TSP_route_length(population[i],smallcity)
		if testtrip1<besttrip:
			besttrip = testtrip1
			m = population[i]

	BLiEG_min = np.where(BLiEG == BLiEG.min())
	shortest_trip = BLiEG[BLiEG_min]
	shortest_route = BRiEG[BLiEG_min,:]
	citynames_shortest_route = TSP_route_2_names(shortest_route[0][0],cities)





	runs_besttrips[run] = besttrip 							#Collecting all the shortes
																#trip at the end of each NoG
																#at each runs (this doesn't need to be
																#the shortest in the run)

print temp_AiEG
AiEG = np.mean(temp_AiEG, axis=0)
AiEG = np.around(AiEG, decimals=2)
print '###################### \n best fit average for each \n \
generation (across runs)\n ##################### '
print '\nAiEG \n= '#,AiEG
for i in range(len(AiEG)):
	print AiEG[i]



print '\n ################################\n best, worst, average and deviation of last tour \n \
length out of %d runs of the algorithm \n \
(of the best individual of last generation) \n' %runs
print '\nruns_besttrips = \n',runs_besttrips   							#B
TOTAL_best_trip_position = np.argmin(runs_besttrips)
TOTAL_best_trip = runs_besttrips[TOTAL_best_trip_position]
print '\nTOTAL_best_trip = \n',TOTAL_best_trip
WORST_of_runs_besttrips_position = np.argmax(runs_besttrips)
WORST_of_runs_besttrips = runs_besttrips[WORST_of_runs_besttrips_position]
print '\nWORST_of_runs_besttrips = \n', WORST_of_runs_besttrips
average_runs_besttrips = np.mean(runs_besttrips, axis=0)
print '\naverage_runs_besttrips = \n', average_runs_besttrips
temp_ST = 0  													#Standard Deviation temp
for i in range(runs_besttrips.shape[0]):
	temp_ST += (runs_besttrips[i]-average_runs_besttrips)**2
ST = np.sqrt(temp_ST/runs_besttrips.shape[0])
print '\nStandard Deviation = \n',ST
stoptime=time.time()-starttime
print 'time = ', stoptime


"""
print '\n \n'
print 'temp_AiEG = \n',temp_AiEG
print 'BRiEG = \n',BRiEG
print 'BLiEG = \n',BLiEG

print '\nshortest_trip in last BRiEG =',shortest_trip
print 'shortest_route in last BLiEG= ',shortest_route
print 'citylist for the shortest route in last BLiEG/BRiEG = \n',citynames_shortest_route

print '\nthe shortest trip of the last generation of the last run is: ',besttrip
print 'm',m
citynames = TSP_route_2_names(m,cities)
print 'citylist for the last generation in the last run = \n',citynames

"""
#################################		DONE		#################################
#################################		DONE		#################################

