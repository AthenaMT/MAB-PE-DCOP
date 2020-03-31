from collections import defaultdict
from collections import Iterable as CollectionIterable
from typing import Dict, Iterable, Union, List
import numpy as np
from queue import Queue, Empty
import threading
from threading import Thread
import math



from xml.dom import minidom
import yaml

from pyDcop.pydcop.computations_graph import factor_graph
from pyDcop.pydcop.dcop.dcop import DCOP
from pyDcop.pydcop.dcop.objects import (
    VariableDomain,
    Variable,
    ExternalVariable,
    VariableWithCostFunc,
    VariableNoisyCostFunc,
    AgentDef,
)
from pydcop.dcop.relations import (
    relation_from_str,
    RelationProtocol,
    NAryMatrixRelation,
    assignment_matrix,
    generate_assignment_as_dict,
)
from pyDcop.pydcop.commands._utils import build_algo_def, _load_modules
from pydcop.infrastructure.run import run_local_thread_dcop

def load_dcop(filenames: str) -> DCOP:
    content = ""
    if isinstance(filenames, CollectionIterable):
        filename = filenames[0]
    else:
        filename = filenames
    xmlFile = minidom.parse(filename)
    xmlPresentation = xmlFile.getElementsByTagName('presentation')
    xmlName = xmlPresentation[0].attributes['name'].value
    xmlObjective = 'max' if xmlPresentation[0].attributes['maximize'].value == 'true' else 'min'
    dcop = DCOP(
        xmlName,
        xmlObjective
    )

    dcop.domains = xml_build_domains(xmlFile)
    dcop.variables = xml_build_variables(xmlFile, dcop)
    dcop._constraints = xml_build_constraints(xmlFile, dcop)
    dcop._agents_def = xml_build_agents(xmlFile)
    dcop.dist_hints = None

    return dcop

def xml_build_domains(xmlFile) -> Dict[str, VariableDomain]:
    domains_dict = {}
    domains = xmlFile.getElementsByTagName('domain')
    for domain in domains:
        domain_name = domain.attributes['name'].value
        domain_type = 'd'
        domain_values = str_2_domain_values(domain.childNodes[0].wholeText)
        new_domain = VariableDomain(domain_name, domain_type, domain_values)
        domains_dict[domain_name] = new_domain
    return domains_dict

def xml_build_variables(xmlFile, dcop) -> Dict[str, Variable]:
    variables_dict = {}
    variables = xmlFile.getElementsByTagName('variable')
    for variable in variables:
        variable_name = variable.attributes['name'].value
        variable_domain = dcop.domain(variable.attributes['domain'].value)
        initial_value = None
        variables_dict[variable_name] = Variable(variable_name, variable_domain, initial_value)

    return variables_dict

def xml_build_constraints(xmlFile, dcop) -> Dict[str, RelationProtocol]:
    constraints_dict = {}
    constraint_normaldistribution_dict = {}
    constraints = xmlFile.getElementsByTagName('constraint')

    for constraint in constraints:
        assignment_dict = {}
        default = None
        constraint_name = constraint.attributes['name'].value
        #the first value is mu_max, and the second is the variance
        constraint_normaldistribution_dict[constraint_name] = [np.random.uniform(10), np.random.uniform(1)]
        constraint_variables = constraint.attributes['scope'].value.split(' ')
        constraint_variables = [dcop.variable(v) for v in constraint_variables]
        values = assignment_matrix(constraint_variables, default)

        relation_name = constraint.attributes['reference'].value
        #fix later by making a dictionary of relations
        for relation in xmlFile.getElementsByTagName('relation'):
            if relation.attributes['name'].value == relation_name:
                constraintTableData = relation.childNodes[0].wholeText #string formatted values
                constraintTableArray = constraintTableData.split('|')
                for assignment in constraintTableArray:
                    value_state_pair = assignment.split(':')
                    #first parameter is mu and second paramter is sigma
                    value_state_pair[0] = np.random.normal(constraint_normaldistribution_dict[constraint_name][0],
                                                            math.sqrt(constraint_normaldistribution_dict[constraint_name][1]))
                    if value_state_pair[0] not in assignment_dict:
                        assignment_dict[value_state_pair[0]] = value_state_pair[1]
                    else:
                        assignment_dict[value_state_pair[0]] = str(assignment_dict[value_state_pair[0]]) + '| ' + str(value_state_pair[1])

        #constructing relation protocol
        #similar code to yamlDCOP
        for value, assignments_def in assignment_dict.items():
            # can be a str like "1 2 3" or "1 2 3 | 1 3 4"
            # several assignment for the same value are separated with |
            assignments_def = assignments_def.split("|")
            for ass_def in assignments_def:
                val_position = values
                vals_def = ass_def.split()
                for i, val_def in enumerate(vals_def[:-1]):
                    iv, _ = constraint_variables[i].domain.to_domain_value(val_def.strip())
                    val_position = val_position[iv]

                # value for the last variable of the assignment
                val_def = vals_def[-1]
                iv, _ = constraint_variables[-1].domain.to_domain_value(val_def.strip())
                val_position[iv] = int(value)
        constraints_dict[constraint_name] = NAryMatrixRelation(constraint_variables, values, name=constraint_name)

    return constraints_dict

def xml_build_agents(xmlFile) -> Dict[str, AgentDef]:
    agents_dict = {}
    agents_list = {}
    relations = xmlFile.getElementsByTagName('relation')
    variables = xmlFile.getElementsByTagName('variable')
    numNodes = len(relations) + len(variables)
    for agent in range(numNodes):
        agent_name = str(agent)
        agents_list[agent_name] = {}
        agents_dict[agent_name] = AgentDef(
            agent_name,
            default_hosting_cost=0,
            hosting_costs={},
            default_route=1,
            routes={},
            **agents_list[agent_name]
        )
    return agents_dict

def str_2_domain_values(domain_str):
    """
    Deserialize a domain expressed as a string.

    If all variable in the domain can be interpreted as a int, the list is a
    list of int, otherwise it is a list of strings.

    :param domain_str: a string like 0..5 of A, B, C, D

    :return: the list of values in the domain
    """
    try:
        sep_index = domain_str.index("..")
        # Domain str is : [0..5]
        min_d = int(domain_str[0:sep_index])
        max_d = int(domain_str[sep_index + 2 :])
        return list(range(min_d, max_d + 1))
    except ValueError:
        values = [v.strip() for v in domain_str[1:].split(",")]
        try:
            return [int(v) for v in values]
        except ValueError:
            return values
