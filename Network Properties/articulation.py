import sys
import time
import networkx as nx
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import functions
from graphframes import *
from copy import deepcopy

sc=SparkContext("local", "degree.py")
sqlContext = SQLContext(sc)

def numComponents(graph, node):
    g = deepcopy(graph)
    g.remove_node(node)
    return(nx.number_connected_components(g))
    

def articulations(g, usegraphframe=False):
    # Get the starting count of connected components
    # YOUR CODE HERE
    
    initialCount = g.connectedComponents().select('component').distinct().count()

    # Default version sparkifies the connected components process 
    # and serializes node iteration.
    if usegraphframe:
        # Get vertex list for serial iteration
        # YOUR CODE HERE
        
        vertexList = g.vertices.map(lambda row: row.id).collect()
        
        # For each vertex, generate a new graphframe missing that vertex
        # and calculate connected component count. Then append count to
        # the output
        # YOUR CODE HERE
        
        list_aPoints = []
        
        for vertex in vertexList:
            temp_vertices = g.vertices.filter('id != "'+vertex+'"')
            temp_edges = g.edges.filter('src != "'+vertex+'"').filter('dst != "'+vertex+'"')
            temp_graph = GraphFrame(temp_vertices, temp_edges)
            
            tempCount = temp_graph.connectedComponents().select('component').distinct().count()
            
            if tempCount > initialCount:
                articulationPoint = 1
            else:
                articulationPoint = 0
            
            list_aPoints.append((vertex, articulationPoint))
        
        result_df = sqlContext.createDataFrame(sc.parallelize(list_aPoints), ['id','articulation'])
        
        return result_df
            
        
    # Non-default version sparkifies node iteration and uses networkx 
    # for connected components count.
    else:
        # YOUR CODE HERE
        graphX = nx.Graph()
        
        edgeList = g.edges.map(lambda row: (row.src, row.dst)).collect()
        graphX.add_edges_from(edgeList)
        
        vertexList = g.vertices.map(lambda row: row.id).collect()
        graphX.add_nodes_from(vertexList)
        
        result_df = sqlContext.createDataFrame(g.vertices.map(lambda row: (row.id, 1 if numComponents(graphX, row.id) > initialCount else 0)),['id','articulation'])
        
        return result_df
        

filename = sys.argv[1]
lines = sc.textFile(filename)

pairs = lines.map(lambda s: s.split(","))
e = sqlContext.createDataFrame(pairs,['src','dst'])
e = e.unionAll(e.selectExpr('src as dst','dst as src')).distinct() # Ensure undirectedness 	

# Extract all endpoints from input file and make a single column frame.
v = e.selectExpr('src as id').unionAll(e.selectExpr('dst as id')).distinct()	

# Create graphframe from the vertices and edges.
g = GraphFrame(v,e)

#Runtime approximately 5 minutes
print("---------------------------")
print("Processing graph using Spark iteration over nodes and serial (networkx) connectedness calculations")
init = time.time()
df = articulations(g, False)
print("Execution time: %s seconds" % (time.time() - init))
print("Articulation points:")
df.filter('articulation = 1').show(truncate=False)
print("---------------------------")

df.toPandas().to_csv("articulations_out.csv")

#Runtime for below is more than 2 hours
print("Processing graph using serial iteration over nodes and GraphFrame connectedness calculations")
init = time.time()
df = articulations(g, True)
print("Execution time: %s seconds" % (time.time() - init))
print("Articulation points:")
df.filter('articulation = 1').show(truncate=False)
