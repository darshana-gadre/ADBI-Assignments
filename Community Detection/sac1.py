import sys
import pandas as pd
import numpy as np
from igraph import *
from scipy import spatial

# get_similarity() function used to calculate similarity between all pairs of vertices of the graph
def get_similarity() :
	global num_vertices, cosine_similarity
	
	# Initialize cosine_similarity[] to 0
	cosine_similarity = [[0 for x in range(num_vertices)] for x in range(num_vertices)]
	
	# Calculate similarity for all pair of vertices
	for i in range(0, num_vertices) :
		vertex_1 = g.vs.select(i)[0].attributes().values()
		
		for j in range(i, num_vertices) :
			vertex_2 = g.vs.select(j)[0].attributes().values()
			
			# Using inbuilt cosine() function for calculating cosine similarity between attributes of vartices
			distance = spatial.distance.cosine(list(vertex_1), list(vertex_2)) + 1.0
			
			cosine_similarity[i][j] = 1.0 / (distance)
			cosine_similarity[j][i] = cosine_similarity[i][j]


# get_modularityGain() function is used to calculate modulairty gain based on the formula given in the paper
def get_modularityGain(vertex, community):
	x = 0
	deg = 0
	num_edges = len(g.es)
	community = list(set(community))

	for node_community in community:
		if g.are_connected(vertex, node_community):
			ind = g.get_eid(vertex, node_community)
			x += g.es["weight"][ind]
	
	# Calculating value of g_neuman based on formula given in paper
	g_neuman = x - sum(g.degree(community)) * g.degree(vertex) / (2 * num_edges)
	
	g_neuman = g_neuman / (2.0 * num_edges)
	
	# Calculating value of g_attribute based on formula given in paper
	g_attribute = 0.0
	
	for i in community:
		g_attribute = g_attribute + cosine_similarity[i][vertex]
		
	g_attribute = g_attribute / len(community) / len(community)
	
	# Return value of modulairty
	return alpha * g_neuman + (1 - alpha) * g_attribute


# Creating clusters of vertices with similar attributes
def clustering(graph, communities):
    count = 0
    
    # Check best cluster for each vertex of the graph
    for vertex_graph in range(num_vertices):
    	
        gains = []
        vertex_community = []
        
        # Get the community that the vertex currently belongs to
        for community in communities:
        	if vertex_graph in community:
        		vertex_community = community
        		break
		
		# Iniialize maximum gain to -1
        max_gain = -1
        max_community = []
        
        # Check if current vertex is a better fit in other communities
        for community in communities:
        	
        	# Get the modularity gain of current vertex with current community
            gain = get_modularityGain(vertex_graph, community)
            
            if gain > 0:
            	
            	# Check if vertex has better modularity gain than current gain
            	if gain > max_gain:
            	    max_gain = gain
            	    max_community = community
                
        # Check if vertex has better modularity gain from a community other than current community
        if set(vertex_community) != set(max_community):
        	if max_gain > 0:
        		
        		# Remove the vertex from current community
        		vertex_community.remove(vertex_graph)
        		
        	# Add the vertex to the community with maximum modularity gain
        	max_community.append(vertex_graph)
        	
        	# Increment count because of a change in communities
        	count += 1
        	
        	# If community is empty after removing current vertex then delete the empty community
        	if len(vertex_community) == 0:
        		communities.remove([])
    
    # Return the count of updated communities  		
    return count


# phaseOne() function implements the phase 1 specified in the paper
def phaseOne(g, cosine_similarity, communities):
	
	# Get the matrix cosine_similarity containing similarity between all existing vertices
	cosine_similarity = get_similarity()
	
	# Get the count i.e number of updated communities (communities with a change in their vertices). For the first call to phaseOne() count will be equal to the number of original vertices.
	count = clustering(g, communities)
	
	num_iterations = 0
	
	# Loop will iterate till their is a change in the distribution of vertices in the communities of till 15 iterations complete, whatever occurs first.
	while count > 0 and num_iterations < 15:
		
		num_iterations+=1
		
		# Get the count i.e number of updated communities (communities with a change in their vertices)
		count = clustering(g, communities)


# phaseTwo() function implements the phase 2 specified in the paper. Phase 2 takes output of phase 1 as its input
def phaseTwo(g, cosine_similarity, groupped_communities, groupped_vertices):
	global num_vertices
	
	# index is used to count the number of existing communities
	index = 0
	
	# groupped_vertices[] contains a unique index assigned to vertices of each community
	for community in groupped_communities :
	
		for vertex in community :
		
			groupped_vertices[vertex] = index
			
		index += 1
	
	# contract_vertices() is an inbuilt fruntion from the igraph package. It replaces groups of vertices with single vertices. The edges are not affected
	g.contract_vertices(groupped_vertices, combine_attrs = "mean")
	
	# simplify() is an inbuilt fruntion from the igraph package. It simplifies a graph by removing self-loops and/or multiple edges.
	g.simplify(multiple = True, loops = True)
	
	# Updating number of vertices after contracting vertices from same cluster into one vertex
	num_vertices = index
	
	groupped_communities = [[vertex] for vertex in range(num_vertices)]
	
	# Initializing weights of edges of the graph to 0
	g.es["weight"] = [0 for edge in range(len(g.es))]
	
	for edge in edges :
	
		community_1, community_2 = groupped_vertices[edge[0]], groupped_vertices[edge[1]]
		
		if community_1 != community_2:
			
			# Updating weights of edges after contracting vertices
			id = g.get_eid(community_1, community_2)
			g.es["weight"][id] += 1
	
	# Calculating cosine simmilarity between vertices of the updated graph after contracting vertices of same community into one
	cosine_similarity = get_similarity()
	
	# Calling phaseOne() of the algorihtm - Computing groups on the updated graph vertices
	phaseOne(g, cosine_similarity, groupped_communities)
	
	

# main

# Check if alpha value is specified. If not exit
if len(sys.argv) != 2:
	print ("Invalid Input")
	sys.exit()

alpha = float(sys.argv[1])

# processing input data

attributes = pd.read_csv('./data/fb_caltech_small_attrlist.csv')
num_vertices = len(attributes)

# Parse fb_caltech_small_edgelist.txt into edges
edgelist_file = open('./data/fb_caltech_small_edgelist.txt')
edge_list = edgelist_file.read().split("\n")

edges = []
for edge in edge_list:
	v = edge.split(' ')
	if v[0] != '' and v[1] != '':
		edges.append((int(v[0]),int(v[1])))

# Create graph using Graph() from igraph package
g = Graph()

# Add vertices, edges and weights to the graph
g.add_vertices(num_vertices)
g.add_edges(edges)

g.es["weight"] = [1 for x in range(len(edges))]

for col in attributes.keys():
	g.vs[col] = attributes[col]

# Initialize cosine_similarity, communities, groupped_vertices arrays
cosine_similarity = [[0 for x in range(num_vertices)] for x in range(num_vertices)] 

communities = [[x] for x in range(num_vertices)]
groupped_vertices = [0 for x in range(num_vertices)]

# Call to the Phase 1 of SAC-1 Algoruthm
phaseOne(g, cosine_similarity, communities)

# Call to the Phase 2 of the SAC-1 Algorithm
phaseTwo(g, cosine_similarity, communities, groupped_vertices)


# Store the communities formed by above implementation into output_file 
output_file = open("./communities.txt", "w")
print('Writing communities in file communities.txt')
for community in communities:
    for i in range(len(community)):
        if i != 0:
            output_file.write(",")
        output_file.write(str(community[i]))
    output_file.write("\n")
output_file.close()
