
from collections import defaultdict
from .UnionFind import UnionFind
import random

class EdgeInfo:
    def __init__(self, cost, u, v, initialPheromone):
        self.cost = cost
        self.u = u  # vertice 1
        self.v = v  # vertice 2 #TODO watch out for vertice order dependent bugs
        self.initialPheromone = initialPheromone  # Initial Level
        self.pheromoneLevel = initialPheromone  # Current level
        self.updates = 0

    def __repr__(self):
        return 'EdgeInfo([{},{}] cost={:.1f}, ip={:.1f}, l={:.1f}, u={})'.format(self.u, self.v, self.cost,
                self.initialPheromone, self.pheromoneLevel, self.updates)

# Functions for sorting the edge infos
def geteiCost(ei):
    return ei.cost

def geteiPheromoneLevel(ei):
    return ei.pheromoneLevel

def getTreeCost(edgeInfos):
    '''Returns the total edge cost of a tree (A list of edgeInfo objects)'''
    return sum(ei.cost for ei in edgeInfos)

def get_nodes_chain_when_degree_is_2(node_num, edgeInfos):
    class Node:
        def __init__(self, left=None, right=None, pos=None):
            self.pos = pos
            self.left = left
            self.right = right

    node_list = [Node() for i in range(node_num)]
    # create path
    for edgeInfo in edgeInfos:
        if node_list[edgeInfo.u].left is None:
            node_list[edgeInfo.u].left = edgeInfo.v
        else:
            node_list[edgeInfo.u].right = edgeInfo.v
        if node_list[edgeInfo.v].left is None:
            node_list[edgeInfo.v].left = edgeInfo.u
        else:
            node_list[edgeInfo.v].right = edgeInfo.u
    # find a end of the chain
    begin_pos = None
    for i in range(len(node_list)):
        if node_list[i].right is None:
            begin_pos = i
            break
    # output the chain
    pos, pre = node_list[begin_pos].left, begin_pos
    nodes_chain = [begin_pos, pos]
    while len(nodes_chain) <= 2 or node_list[pos].right is not None:
        if node_list[pos].left != pre:
            nodes_chain.append(node_list[pos].left)
            pre = pos
            pos = nodes_chain[-1]
        else:
            nodes_chain.append(node_list[pos].right)
            pre = pos
            pos = nodes_chain[-1]
    print(f'The path is: {nodes_chain}')
    return nodes_chain

class Ant:
    '''Represents an ant in our model'''
    def __init__(self, initialVertice):
        '''Initialize with starting vertice position'''
        self.position = initialVertice
        # List of visited vertices
        self.visited = set()

    def __repr__(self):
        return 'Ant({}, {})'.format(self.position, self.visited)

class AB_DCMST:
    maxCycles = 1000  # 10000  # Maximum allowed cycles
    s = 75  # Steps: number of edges an ant traverses each cycle
    H = 0.5  # Initial pheromone evaporation factor
    Y = 1.5  # Initial pheromone enhancement factor
    #DH = 0.95  # Update constant applied to η
    #DY = 1.05  # Update constant applied to γ
    #updateCycles = 500  # Number of cycles between updating η and γ
    updateSteps = s / 3  # Number of steps between applying pheromone updates
    escapeCycles = 100  # Number of cycles without improvement before escaping
    stopCycles = 2500  # Number of cycles without improvement before stopping

    def __init__(self, edges, verbose=False):

        # Edge data
        edges.sort()
        self.m = float(edges[ 0][0])  # min edge cost
        self.M = float(edges[-1][0])  # max edge cost
        costDiff = self.M - self.m
        self.minP = costDiff / 3.0  # max pheromone level of edges
        self.maxP = 1000.0 * (costDiff + self.minP)  # min pheromone level of edges

        self.verbose = verbose
        self.d = 0  # Degree constraint

        self.graph = defaultdict(dict)
        #Set the static edge data and edgeInfo retrieval data structures
        self.edgeInfos = []
        for cost, u, v in edges:
            cost = float(cost)
            # Calculate pheromone level of current edge
            initialPheromone = (self.M - cost) + self.minP
            ei = EdgeInfo(cost, u, v, initialPheromone)
            # To get edge info from two vertices easily
            self.graph[u][v] = ei
            self.graph[v][u] = ei
            # To iterate easily on edges
            self.edgeInfos.append(ei)

        self.ants = []
        self.n = len(self.graph) # Number of vertices
        self.nCandiates = 5 * self.n # Candidate set size

    # the sum Could be calculated after each pheromone update
    def __getNextVertice(self, startVertice):
        '''Returns a neighbor vertice according to probability of pheromone levels'''
        # Calculate the sum of the pheromone levels neighbor edges
        neighborPheromes = (ei.pheromoneLevel for ei in self.graph[startVertice].values())
        pheromoneSum = sum(neighborPheromes)

        # Pick a number between 0 and the sum of pheromone levels
        target = random.uniform(0.0, pheromoneSum)
        pheromoneSum = 0.0

        # Find the vertice that has the target number in it's pheromone range
        for v2, ei in self.graph[startVertice].items():
            pheromoneSum += ei.pheromoneLevel
            if pheromoneSum >= target:
                return v2
        return v2

    def __moveAnt(self, ant):
        '''Move an ant'''
        nAttempts = 0
        i = ant.position
        while nAttempts < 5:
            # select an edge (i, j) at random and proportional to its pheromone level
            j = self.__getNextVertice(i)
            if j not in ant.visited:
                # mark edge (i, j) for pheromone update
                self.graph[i][j].updates += 1
                # move ant to vertice j
                ant.position = j
                # mark j visited
                ant.visited.add(j)
                break
            else:
                nAttempts += 1

    def __clipEdgeInfoPheromoneLevel(self, ei):
        '''Put pheromone level in acceptable range'''
        if ei.pheromoneLevel > self.maxP:
                ei.pheromoneLevel = self.maxP - ei.initialPheromone
        elif ei.pheromoneLevel < self.minP:
            ei.pheromoneLevel = self.minP + ei.initialPheromone

    def __updatePheromones(self):
        '''Applies evaporation and reinforcement of pheromone levels on all edges'''
        for ei in self.edgeInfos:
            ei.pheromoneLevel = (1 - self.H) * ei.pheromoneLevel + \
                                ei.updates * ei.initialPheromone
            self.__clipEdgeInfoPheromoneLevel(ei)
            # Clear ei.updates
            ei.updates = 0

    def __pheromoneEnhancement(self, edgeInfos):
        '''Enhances the pheromone level of passed edges'''
        for ei in edgeInfos:
            ei.pheromoneLevel *= self.Y
            self.__clipEdgeInfoPheromoneLevel(ei)

    def __getTree(self):
        # Sort edges by decreasing pheromone levels
        self.edgeInfos.sort(key=geteiPheromoneLevel, reverse=True)
        start = 0
        # Current edges that will be inspected
        C = []
        # Data structure for Kruskal's algorithm
        subtrees = UnionFind()
        solution = []
        degrees = defaultdict(int) # Map of degrees for each vertice

        # While we don't have n-1 edges in our solution
        while len(solution) != self.n - 1:
            # Pick the the next nCanditates edges
            C = self.edgeInfos[start : start + self.nCandiates]

            # Sort the edge selection by increasing cost
            C.sort(key=geteiCost)
            start += self.nCandiates

            for ei in C:
                u, v = ei.u, ei.v
                # Test if edge can be added to the current solution
                if subtrees[u] != subtrees[v] and \
                   degrees[u] < self.d and degrees[v] < self.d:  # Check constraint
                    solution.append(ei)
                    # Increment degrees of vertices of the added edge
                    degrees[u] += 1
                    degrees[v] += 1
                    # Update subtrees
                    subtrees.union(u, v)
        return solution

    def getSolution(self, degreeConstraint):
        # Initialization State
        self.d = degreeConstraint
        # assign one ant to each vertex of the graph
        self.ants = [Ant(i) for i in self.graph.keys()]
        # initialize pheromone level of each edge
        for ei in self.edgeInfos:
            ei.pheromoneLevel = ei.initialPheromone
            ei.updates = 0

        B = self.__getTree()  # best tree
        minCost = getTreeCost(B)
        if self.verbose: print('Initial minCost =', minCost)
        lastImprovementCycle = 0
        evaporationTimer = 0

        for cycle in range(1, self.maxCycles + 1):  # stopping criteria not met:
            if self.verbose and cycle%100 == 0:
                print('cycle =', cycle)

            if cycle - lastImprovementCycle > self.stopCycles:
                break

            # Exploration Stage
            for step in range(1, self.s + 1):
                # Move all ants along one edge
                for ant in self.ants:
                    self.__moveAnt(ant)
                if step % self.updateSteps == 0:
                    # Update pheromone levels for all edges
                    self.__updatePheromones()

            # Remove visited vertices constraint from ants
            for ant in self.ants:
                ant.visited.clear()

            # Tree Construction Stage
            T = self.__getTree()
            newCost = getTreeCost(T)

            if newCost < minCost:
                B = T  # Update best tree
                minCost = newCost
                if self.verbose: print('New min Cost', minCost)
                lastImprovementCycle = cycle
                evaporationTimer = cycle

            # enhance pheromone levels for edges in the best tree B
            self.__pheromoneEnhancement(B)

            # if no improvement in escapeCycles cycles
            if cycle - evaporationTimer > self.escapeCycles:
                if self.verbose: print('No improvement in', self.escapeCycles, 'cycles')
                # evaporate pheromone levels from edges of the best tree B
                for ei in B:
                    ei.pheromoneLevel *= (1 - self.H)
                # Reset evaporationTimer
                evaporationTimer = cycle
        return B

