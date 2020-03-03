from xml.dom import minidom
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import bipartite
from Nodes import VNode, FNode
from fglib import rv

class FactorGraph:
    #initialize necessary variables
    def __init__(self, filename):
        self.myCOP = minidom.parse(filename)
        self.agents = self.myCOP.getElementsByTagName('agent')
        self.domains = self.myCOP.getElementsByTagName('domain')
        self.variables = self.myCOP.getElementsByTagName('variable')
        self.relations = self.myCOP.getElementsByTagName('relation')
        self.constraints = self.myCOP.getElementsByTagName('constraint')
        self.variableNodeList = []
        self.factorNodeList = []
        self.fg = nx.Graph()
        self.fg_label = nx.Graph()
        self.constructVariableNodes()
        self.constructFactorNodes()
        self.createGraph()
        self.createLabelGraph()

    #Necessary Functions to Parse Through an XML file
    def getAgentName(self, agentArray, index): #gets the name of the agent
        return self.agents[index].attributes['name'].value

    def getAgent(self, agentName): #gets the actual agent object
        for agent in self.agents:
            if agent.attributes['name'].value == agentName:
                return agent

    def getDomain(self, domainName): #gets a Domain corresponding to a Domain Name
        for domain in self.domains:
            if domain.attributes['name'].value == domainName:
                return domain

    def getAgentDomain(self, agentName): #gets the Domain associated with an agent
        for variable in self.variables:
            if (variable.attributes['agent'].value == agentName):
                agentDomain = variable.attributes['domain'].value
                return self.getDomain(agentDomain)

    def getDomainRange(self, agentName): #gets the Domain Range associated with an agent
        agentDomain = self.getAgentDomain(agentName)
        return (agentDomain.childNodes[0].wholeText)

    #FactorGraph Functions
    def constructConstraintTable(self, relationName): #constructs Constraint Table Graph for a given relation
        for relation in self.relations:
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

    def getRelationAgentNames(self, constraint): #gets the Agents associated with each constraint
        agentNames = str(constraint.attributes['scope'].value).split(' ')
        agentNames = [int(scope) for scope in agentNames]
        return (agentNames)

    def constructVariableNodes(self): #constructs list of Variable Nodes
        for i in range(len(self.agents)):
            variablenode = VNode(self.getAgentName(self.agents, i), rv.Discrete)
            self.variableNodeList.append(variablenode)

    def constructFactorNodes(self): #constructs list of Factor Nodes
        for constraint in self.constraints:
            constraintName = constraint.attributes['name'].value
            relationName = constraint.attributes['reference'].value
            factorNodeTable = self.constructConstraintTable(relationName)
            relationAgentNames = self.getRelationAgentNames(constraint)
            factorNode = FNode(relationName, factorNodeTable)
            self.factorNodeList.append(factorNode)

    #creating Graph
    def createGraph(self):
        self.fg.add_nodes_from(self.variableNodeList, bipartite = 0)
        self.fg.add_nodes_from(self.factorNodeList, bipartite = 1)

        for constraint in self.constraints:
            agentNames = self.getRelationAgentNames(constraint)
            factorNode = [value for index,value in enumerate(self.factorNodeList) if value.label == constraint.attributes['reference'].value][0]
            if len(agentNames) > 1:
                variableNode1 = [value for index,value in enumerate(self.variableNodeList) if value.label == str(agentNames[0])][0]
                variableNode2 = [value for index,value in enumerate(self.variableNodeList) if value.label == str(agentNames[1])][0]
                self.fg.add_edge(variableNode1, factorNode)
                self.fg.add_edge(factorNode, variableNode2)
            else:
                variableNode1 = [value for index,value in enumerate(self.variableNodeList) if value.label == str(agentNames[0])][0]
                self.fg.add_edge(variableNode1, factorNode)

        #initializing variables and factors label
    def createLabelGraph(self):
        variableNodesLabel = [value.label for value in self.variableNodeList]
        factorNodesLabel = [value.label for value in self.factorNodeList]

        #creating Label Graph
        self.fg_label.add_nodes_from(variableNodesLabel, bipartite = 0)
        self.fg_label.add_nodes_from(factorNodesLabel, bipartite = 1)

        for constraint in self.constraints:
            agentNames = self.getRelationAgentNames(constraint)
            factorNode = [value for index,value in enumerate(factorNodesLabel) if value == constraint.attributes['reference'].value][0]
            if len(agentNames) > 1:
                variableNode1 = [value for index,value in enumerate(variableNodesLabel) if value == str(agentNames[0])][0]
                variableNode2 = [value for index,value in enumerate(variableNodesLabel) if value == str(agentNames[1])][0]
                self.fg_label.add_edge(variableNode1, factorNode)
                self.fg_label.add_edge(factorNode, variableNode2)
            else:
                variableNode1 = [value for index,value in enumerate(variableNodes) if value == str(agentNames[0])][0]
                self.fg_label.add_edge(variableNode1, factorNode)


    def drawNetwork(self):
        nx.draw_networkx(self.fg_label)
        plt.savefig('test1.png')

def main():
    factorGraph = FactorGraph('input/Rnd5-5-4.xml')
    factorGraph.drawNetwork()

if __name__ == "__main__":
    main()
