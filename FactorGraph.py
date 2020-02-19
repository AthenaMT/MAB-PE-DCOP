from xml.dom import minidom
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import bipartite
from Nodes import VNode, FNode
from fglib import rv


#initialize necessary variables
myCOP = minidom.parse('input/Rnd5-5-4.xml')
agents = myCOP.getElementsByTagName('agent')
domains = myCOP.getElementsByTagName('domain')
variables = myCOP.getElementsByTagName('variable')
relations = myCOP.getElementsByTagName('relation')
constraints = myCOP.getElementsByTagName('constraint')

#Necessary Functions to Parse Through an XML file
def getAgentName(agentArray, index): #gets the name of the agent
    return agents[index].attributes['name'].value

def getAgent(agentName): #gets the actual agent object
    for agent in agents:
        if agent.attributes['name'].value == agentName:
            return agent

def getDomain(domainName): #gets a Domain corresponding to a Domain Name
    for domain in domains:
        if domain.attributes['name'].value == domainName:
            return domain

def getAgentDomain(agentName): #gets the Domain associated with an agent
    for variable in variables:
        if (variable.attributes['agent'].value == agentName):
            agentDomain = variable.attributes['domain'].value
            return getDomain(agentDomain)

def getDomainRange(agentName): #gets the Domain Range associated with an agent
    agentDomain = getAgentDomain(agentName)
    return (agentDomain.childNodes[0].wholeText)

#FactorGraph Functions
def constructConstraintTable(relationName): #constructs Constraint Table Graph for a given relation
    for relation in relations:
        if relation.attributes['name'].value == relationName:
            constraintTableData = relation.childNodes[0].wholeText #string formatted values
            constraintTableArray = constraintTableData.split('|')
            rowCounter = 0
            tempArray = []
            factorGraphArray = []
            for constraint in constraintTableArray:
                value_index_pair = constraint.split(':')
                value = value_index_pair[0]
                indexes = value_index_pair[1].rstrip().split(' ')
                indexes = [int(num)-1 for num in indexes]
                if rowCounter != indexes[0]:
                    rowCounter = indexes[0]
                    factorGraphArray.append(tempArray)
                    tempArray = [value]
                else:
                    tempArray.append(value)
            factorGraphArray.append(tempArray)
            return (tuple(factorGraphArray))

def getRelationAgentNames(constraint): #gets the Agents associated with each constraint
    agentNames = str(constraint.attributes['scope'].value).split(' ')
    agentNames = [int(scope) for scope in agentNames]
    return (agentNames)

def constructVariableNodes(): #constructs list of Variable Nodes
    variableNodeList = []
    for i in range(len(agents)):
        variablenode = VNode(getAgentName(agents, i), rv.Discrete)
        variableNodeList.append(variablenode)

    return (variableNodeList)

def constructFactorNodes(): #constructs list of Factor Nodes
    factorNodeList = []
    for constraint in constraints:
        constraintName = constraint.attributes['name'].value
        relationName = constraint.attributes['reference'].value
        factorNodeTable = constructConstraintTable(relationName)
        relationAgentNames = getRelationAgentNames(constraint)
        factorNode = FNode(relationName, factorNodeTable)
        factorNodeList.append(factorNode)

    return factorNodeList

#initializing variables and factors
variableNodes = constructVariableNodes()
factorNodes = constructFactorNodes()

#creating Graph
fg = nx.Graph()
fg.add_nodes_from(variableNodes, bipartite = 0)
fg.add_nodes_from(factorNodes, bipartite = 1)

for constraint in constraints:
    agentNames = getRelationAgentNames(constraint)
    factorNode = [value for index,value in enumerate(factorNodes) if value.label == constraint.attributes['reference'].value][0]
    if len(agentNames) > 1:
        variableNode1 = [value for index,value in enumerate(variableNodes) if value.label == str(agentNames[0])][0]
        variableNode2 = [value for index,value in enumerate(variableNodes) if value.label == str(agentNames[1])][0]
        fg.add_edge(variableNode1, factorNode)
        fg.add_edge(factorNode, variableNode2)
    else:
        variableNode1 = [value for index,value in enumerate(variableNodes) if value.label == str(agentNames[0])][0]
        fg.add_edge(variableNode1, factorNode)

#initializing variables and factors label
variableNodesLabel = [value.label for value in constructVariableNodes()]
factorNodesLabel = [value.label for value in constructFactorNodes()]

#creating Label Graph
fg_label = nx.Graph()
fg_label.add_nodes_from(variableNodesLabel, bipartite = 0)
fg_label.add_nodes_from(factorNodesLabel, bipartite = 1)

for constraint in constraints:
    agentNames = getRelationAgentNames(constraint)
    factorNode = [value for index,value in enumerate(factorNodesLabel) if value == constraint.attributes['reference'].value][0]
    if len(agentNames) > 1:
        variableNode1 = [value for index,value in enumerate(variableNodesLabel) if value == str(agentNames[0])][0]
        variableNode2 = [value for index,value in enumerate(variableNodesLabel) if value == str(agentNames[1])][0]
        fg_label.add_edge(variableNode1, factorNode)
        fg_label.add_edge(factorNode, variableNode2)
    else:
        variableNode1 = [value for index,value in enumerate(variableNodes) if value == str(agentNames[0])][0]
        fg_label.add_edge(variableNode1, factorNode)


nx.draw_networkx(fg_label)
plt.savefig('test.png')
