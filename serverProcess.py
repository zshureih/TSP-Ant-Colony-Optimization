import math
import random
import numpy as np
import time
from multiprocessing import Process, Manager, Pool
from multiprocessing.managers import BaseManager, NamespaceProxy
# from threading import Thread
import sys

def d(u, v):
    return round(math.sqrt((u[0]-v[0])*(u[0]-v[0])+(u[1]-v[1])*(u[1]-v[1])), 0)

class graph:
    def __init__(self):
        self.cities = {}
        self.cityDistances = []

class AntManager(BaseManager):
    pass

class AntProxy(NamespaceProxy):
    _exposed_ = ('__getattribute__', '__setattr__', '__delattr__')

class Ant():
    def __init__(self, startCity, availableCities, pheromoneTrail, alpha, beta, getDistance, first_pass=False):
        # have ants run in seperate threads to speed up execution time
        self.first_pass = first_pass
        self.startCity = startCity
        self.availableCities = availableCities.copy()
        self.currentCity = startCity
        self.path = [startCity]
        self.pathLength = 0
        self.pheromoneTrail = pheromoneTrail
        self.tour_complete = False
        self.alpha = alpha
        self.beta = beta
        self.distanceFunc = getDistance

    def run(self):
        numCities = len(self.availableCities)

        # have the ant run a cycle 
        for i in range(numCities):
            nextCity = self._pick_path()
            self._update_path(nextCity)
            self.currentCity = nextCity

        # have that cycle end by returning to the original city
        self._update_path(self.startCity)
        self.currentCity = self.startCity

        self.tour_complete = True

    def _pick_path(self):
        if self.first_pass:
            return random.choice(self.availableCities)

        attractiveness = dict()
        total = 0
        #for each possible location get tau*eta, sum all attractiveness amounts for calculating probability of next move
        for availableCity in self.availableCities:
            pheromoneLevel = float(self.pheromoneTrail[self.currentCity][availableCity])
            distance = float(self.distanceFunc(self.currentCity, availableCity))

            if distance == 0:
                print("error: {}, {}".format(self.currentCity, availableCity))

            attractiveness[availableCity] = (pheromoneLevel ** self.alpha) * ((1/distance) ** self.beta)
            total += attractiveness[availableCity]

        # if the sum total is almost 0 (highly unlikely, but possible) 
        if total == 0.0:
            for key in attractiveness.keys():
                if attractiveness[key] == 0.0:
                    attractiveness[key] = np.nextafter(0, 1) #incrememt by smallest value possible
            total = np.nextafter(0, 1)
        
        # randomly decide based on attractiveness
        tossUp = random.random()
        cumm = 0
        for availableCity in attractiveness:
            weight = attractiveness[availableCity] / total
            if tossUp <= weight + cumm:
                return availableCity
            cumm += weight

    def _update_path(self, nextCity):
        self.path.append(nextCity)

        if nextCity != self.startCity:
            self.availableCities.remove(nextCity)

        self.pathLength += self.distanceFunc(self.currentCity, nextCity)

    def get_path(self):
        if self.tour_complete:
            return self.path
        return None

    def get_distance_traveled(self):
        if self.tour_complete:
            return self.pathLength
        return None

AntManager.register('Ant', Ant, AntProxy)

class AntColony:
    def __init__(self, cities, cityDistances, start=None, ant_count=2, alpha=1.0, beta=1.15,  pheromone_evaporation_rate=.60, pheromone_constant=1000.0, iterations=80):
        self.manager = AntManager()
        self.manager.start()

        self.cities = cities
        
        #if start is none, then initialize to 0, otherwise find the key of the node
        if start is None:
            self.start = 0
        else:
            for key, value in self.id_to_key.items():
                if value == start:
                    self.start = key

        self.first_pass = True #tracks whether this is the initial iteration
        self.cityDistances = self._init_matrix(len(cities))
        self.ant_count = ant_count
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.pheromone_evaporation_rate = float(pheromone_evaporation_rate)
        self.pheromone_constant = float(pheromone_constant)

        self.iterations = iterations
		
        self.ant_updated_pheromone_map = self._init_matrix(len(cities))
        self.pheromone_map = self._init_matrix(len(cities))
        self.id_to_key = self._init_id_to_key(cities)
        self.ants = self._init_ants(self.start)
		#other internal variable init
        self.shortest_distance = None # tracks distance of shrotest path
        self.shortest_path = None

    def _init_matrix(self, size, value=0.0):
        ret = []
        for row in range(size):
            ret.append([float(value) for x in range(size)])
        return ret

    def _init_id_to_key(self, nodes):
        id_to_key = dict()
        
        i = 0
        for key in sorted(nodes.keys()):
            id_to_key[i] = key
            i += 1

        return id_to_key

    def getDistance(self, n1, n2):
        # if distance hasn't already been calculated, calculate it (this should save computation time)
        if not self.cityDistances[n1][n2]:
            distance = d(self.cities[n1], self.cities[n2])
            
            self.cityDistances[n1][n2] = distance 
            return distance
            
        return self.cityDistances[n1][n2]

    def _init_ants(self, start):
        #the ants can only visit the original city at the end of their tour
        availableCities = list(self.cities.keys())
        availableCities.remove(start)

        # if this is the first pass, construct ants
        if self.first_pass:
            pool = Pool(self.ant_count)
            # self.manager.Ant(start, availableCities.copy(), self.pheromone_map.copy(), self.alpha, self.beta, self.getDistance, first_pass=True)
            # ants = [Ant(start, availableCities, self.pheromone_map, self.alpha, self.beta, self.getDistance, first_pass=True) for x in range(self.ant_count)]
            # return ants
        
        # otherwise, reinitialize them
        for ant in self.ants:
            ant.__init__(start, availableCities, self.pheromone_map, self.alpha, self.beta, self.getDistance)		

    def update_pheromone_map(self):
        for start in range(len(self.pheromone_map)):
            for end in range(len(self.pheromone_map)):
                #decay the pheromone value at this point, then add the total added pheromone by the ants
                #tau_xy <- (1-rho)*tau_xy + delta tau_xy_k
                self.pheromone_map[start][end] = (1 - self.pheromone_evaporation_rate)*self.pheromone_map[start][end] + self.ant_updated_pheromone_map[start][end]
    
    def _populate_ant_updated_pheromone_map(self, ant):
        for i in range(len(ant.path) - 1):
            currPheromoneVal = float(self.ant_updated_pheromone_map[ant.path[i]][ant.path[i+1]])

            #  delta tau_xy_k = Q / L_k
            newPheromoneVal = self.pheromone_constant/ant.get_distance_traveled()

            self.ant_updated_pheromone_map[ant.path[i]][ant.path[i+1]] = newPheromoneVal + currPheromoneVal
            self.ant_updated_pheromone_map[ant.path[i+1]][ant.path[i]] = newPheromoneVal + currPheromoneVal

    def run(self):
        for t in range(self.iterations):
            #start multi-threaded ants, calls ant.run() in a new thread
            for ant in self.ants:
                print(ant)

            # wait for ants to finish before updating shared resources (pheromone maps)
            for ant in self.ants:
                ant.thread.join()

            for ant in self.ants:
                #update and_updated_pheromone map with this ant's contribution along its route
                self._populate_ant_updated_pheromone_map(ant)

                # if we haven't run a path yet, the first run for comparisons
                if self.shortest_distance is None:
                    self.shortest_distance = ant.get_distance_traveled()

                if self.shortest_path is None:
                    self.shortest_path = ant.get_path()

                # if we see a shorter path, save it for later
                if ant.get_distance_traveled() < self.shortest_distance:
                    self.shortest_distance = ant.get_distance_traveled()
                    self.shortest_path = ant.get_path()

            # update pheromone map based on previously gathered data (_populate_ant_updated_pheromone_map) and pheromone decay rate
            self.update_pheromone_map()

            # remove first pass flag
            if self.first_pass:
                self.first_pass = False

            # reset all ants for next iteration
            self._init_ants(self.start)

            # reset ant_update_pheromone map for next iteration
            self.ant_updated_pheromone_map = self._init_matrix(len(self.cities))

        # output shortest path after ending of all iterations
        finalPath = []
        for id in self.shortest_path:
            finalPath.append(self.id_to_key[id])

        return finalPath
        
def processInput(graphInput):
    testFileName = sys.argv[1]
    # testFileName = "test-input-2.txt"
    graphInput.testFileName = testFileName          #Save fileName for output file
    #print("This is test file: " + testFileName)

    #Loop through file. " " - cityN - cityX -cityY
    with open(testFileName) as inputFile:
        for line in inputFile:
            readIn = line.strip().split()
            cityVertex = int(readIn[0])
            cityX = int(readIn[1])
            cityY = int(readIn[2])
            #print("CityN--" + str(cityVertex) + "--xCoord--" + str(cityX) + "--yCoord--" + str(cityY))
            graphInput.cities[cityVertex] = [cityX, cityY]
            #print(readIn)

# def moveAnt(i, ant, graph):
#     # from current city, calculate probability to go to each connected city (that isn't in path)
#     a = .5
#     b = 1.2
#     p = [0 for x in range(len(graph.cityN))]  # list of probabilities from i
#     tau = [] # list of pheromone strengths from i to j
#     eta = [] # list of visibilities from i to j (1/distance[i][j])
#     Epsilon = 0
#     for j in graph.cityN:
#         if not j in ant.path and j != i:
#             if graph.cityDistances[i][j] != 0:
#                 e = 1/graph.cityDistances[i][j] ** a
#                 t = graph.pheromones[i][j] ** b
#                 Epsilon += e * t

#     for j in graph.cityN:
#         if not j in ant.path and j != i:
#             if graph.cityDistances[i][j] != 0:
#                 e = 1/graph.cityDistances[i][j] ** a
#                 t = graph.pheromones[i][j] ** b
#                 p[j] = ((e*t) / Epsilon)

#     return np.random.choice(graph.cityN, 1, p=p)[0]

# def updatePheromones(graph, colony):
#     updateDelta(graph, colony)
#     p = 0.4
#     for i in range(len(graph.pheromones)):
#         for j in range(len(graph.pheromones)):
#             graph.pheromones[i][j] = (1-p)*graph.pheromones[i][j] + graph.delta[i][j]
#             graph.pheromones[j][i] = (1-p)*graph.pheromones[i][j] + graph.delta[i][j]


# def updateDelta(graph, colony):
#     Q = 1
#     for ant in colony.ants:
#         for i in range(1, len(ant.path)):
#             currentCity = ant.path[i]
#             prevCity = ant.path[i-1]
#             graph.delta[prevCity][currentCity] += (Q / ant.pathLength)
#             graph.delta[currentCity][prevCity] += (Q / ant.pathLength)

# def getPathLength(graph, ant):
#     L = 0
#     for i in range(1, len(ant.path)):
#         currentCity = ant.path[i]
#         prevCity = ant.path[i-1]
#         L += graph.cityDistances[prevCity][currentCity]
#     ant.pathLength = L


def main():
    tspGraph = graph()#Instantiate InputGraph
    processInput(tspGraph)
    
    #Calculate distance between all city Vertices

    startTime = time.time()

    T = 200
    m = 25
    colony = AntColony(tspGraph.cities, tspGraph.cityDistances, None, m, alpha=1.0, beta=1.0, pheromone_evaporation_rate=0.6, pheromone_constant=10000, iterations=T)
    bestPath = colony.run()
    print(colony.shortest_path)
    print(colony.shortest_distance)
    # for t in range(0, T):
    #     pathLengths = []
    #     for ant in colony.ants:
    #         firstCity = random.randrange(0, len(tspGraph.cityN))
    #         ant.path = [firstCity]
    #         currentCity = firstCity

    #         for move_count in range(0, len(tspGraph.cityN) - 1):
    #             #determine where ant goes next with probFunction
    #             currentCity = moveAnt(currentCity, ant, tspGraph)
    #             ant.path.append(currentCity)

    #         ant.path.append(firstCity)
    #         getPathLength(tspGraph, ant)

    #     updatePheromones(tspGraph, colony)

    #     if t == T-1:
    #         print(colony.ants[len(colony.ants) - 1].path)
    #         print(colony.ants[len(colony.ants) - 1].pathLength)

    runTime = time.time() - startTime
    print(runTime)




if __name__ == "__main__":
    main()