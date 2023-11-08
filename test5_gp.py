import operator, random
from deap import algorithms, base, creator, tools, gp
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

def volume(v):
    result = v ** 3
    return result

def mysin(x):

    return np.sin(x * np.pi /180. )

def mytan(x):

    return np.tan(x * np.pi /180. )

def mycos(x):
    return np.cos(x * np.pi /180. )

def safe_pow(x,y):
    if y > 0:
        return np.power(x,y)

def evaluateRegression(individual, points, pset,label):
    func,code = gp.compile(expr=individual, pset=pset)
    
    created_feature = func(*(points.T))
    if np.size(created_feature) == 1:
        created_feature = np.repeat(created_feature, points.shape[0],axis=0)
    
    if sum(np.isnan(created_feature))>0 or sum(np.isinf(created_feature))>0:
        cost = np.inf
        return cost,
    
    # the Evaluation part should be changed based on given problem , reshape , ...
    model = LinearRegression()
    y_pred = cross_val_predict(model, created_feature.reshape(-1,1), label, cv=10)
    mse = mean_squared_error(y_pred,label)   
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_pred,label)
    cost = mae
    return cost,

def evaluateRegressionMultiGene(genes, points, pset,label):
    num_of_points = points.shape[0]
    created_features = np.zeros((num_of_points,len(genes)))
    i = 0
    for gene in genes:
        
        func = gp.compile(expr=gene, pset=pset)
        cf = func(*(points.T))       
        if np.size(cf) == 1:
            cf = np.repeat(cf, num_of_points,axis=0)
        
        if sum(np.isnan(cf))>0 or sum(np.isinf(cf))>0:
            cost = np.inf
            return cost,
        created_features[:,i] = cf
        i +=1 
    

    # the Evaluation part should be changed based on given problem , reshape , ...
    model = LinearRegression()
    y_pred = cross_val_predict(model, created_features, label, cv=10)
    mse = mean_squared_error(y_pred,label)   
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_pred,label)
    cost = mae
    return cost,

# Loading the data
data = pd.read_csv("/home/zeiss/Downloads/test.csv")
num_of_vars = data.shape[1] - 1 
points = data.iloc[:,:num_of_vars]
label = data.iloc[:,num_of_vars:]
names = list(points.columns.values)
points = np.array(points)
label = np.array(label)
max_genes = 5 # random.randint(2, num_of_genes)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
creator.create("Multigene", list, fitness=creator.FitnessMin)


# Defining the primitive set
pset = gp.PrimitiveSet(name="MAIN", arity=num_of_vars)


pset.addPrimitive(operator.add, arity=2)
pset.addPrimitive(operator.sub, arity=2)
pset.addPrimitive(operator.mul, arity=2)
pset.addPrimitive(operator.abs, arity=1)
pset.addPrimitive(operator.truediv, arity=2)
pset.addPrimitive(np.sqrt, arity=1)
pset.addPrimitive(np.square, arity=1)
pset.addPrimitive(operator.neg, arity=1)
pset.addPrimitive(volume, arity=1)
pset.addPrimitive(mysin, arity=1)
pset.addPrimitive(mycos, arity=1)
pset.addPrimitive(mytan, arity=1)
pset.addPrimitive(np.log, arity=1)
pset.addPrimitive(np.log10, arity=1)
pset.addPrimitive(np.log2, arity=1)
pset.addPrimitive(np.exp, arity=1)
# pset.addPrimitive(safe_pow, arity=2)
pset.addEphemeralConstant('rand101', lambda: random.uniform(-10, 10))



# # ************************Defining a toolbox for single gene config*********************************** 
# toolbox = base.Toolbox()
# toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
# toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
# toolbox.register("population", tools.initRepeat, list, toolbox.individual)
# toolbox.register("evaluate", evaluateRegression, points=points, pset=pset,label=label) # single gene cost function
# toolbox.register("mate", gp.cxOnePoint)
# toolbox.register("expr_mut", gp.genHalfAndHalf, min_=0, max_=2) # 
# toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
# toolbox.register("select", tools.selTournament, tournsize=3)

# # ************************Defining a toolbox for multi gene config*********************************** 
toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("multigene", tools.initRepeat, creator.Multigene, toolbox.individual,max_genes) # multi gene individual
toolbox.register("population", tools.initRepeat, list, toolbox.multigene)
toolbox.register("evaluate", evaluateRegressionMultiGene, points=points, pset=pset,label=label) # multi gene cost function
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genHalfAndHalf, min_=0, max_=2) # 
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.register("select", tools.selTournament, tournsize=10)


# Testing the program
if __name__ == "__main__":
    pop = toolbox.population(n=300)
    
    #def eaSimple(population, toolbox, cxpb, mutpb, ngen,mgcrpb,max_num_of_genes, stats=None, halloffame=None, verbose=__debug__):
    algorithms.eaSimple(pop, toolbox, cxpb=0.7, mutpb=0.1, mgcrpb=0.8,max_num_of_genes=max_genes, ngen=100, verbose=True)    
    bests = tools.selBest(pop, k=5)
    print(len(bests[0]))
    for i in range(0,len(bests[0])):
        print(str(bests[0][i]))
    
    print(bests[0].fitness)
   

# for i in range(0,num_of_vars):
#     pset.arguments[i] = names[i]

import pygraphviz #/home/zeiss/anaconda3/envs/spyder-env/lib/python3.11/site-packages/graphviz
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
for i in range(0,len(bests[0])):
    nodes, edges, labels = gp.graph(bests[0][i])
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    
    pos = graphviz_layout(graph)
    
    plt.figure(figsize=(15,15),dpi=900)
    nx.draw_networkx_nodes(graph, pos, node_size=4000,cmap=plt.cm.Wistia, node_color=range(len(graph)))
    nx.draw_networkx_edges(graph, pos)
    nx.draw_networkx_labels(graph, pos, labels)
    plt.axis("off")
    plt.show()     