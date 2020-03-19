#Things To Do
#import correct modules (complete)

#Load File Correctly (complete)
#build DCOP objects (complete)
#build Domains (complete)
#build Variables
#build ExternalVariables?
#build Constraints
#build Agents
#build dist_hints (may not even be necessary)

from collections import defaultdict
from collections import Iterable as CollectionIterable
from typing import Dict, Iterable, Union, List
import numpy as np

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


#things for load_dcop
# dcop.domains = _build_domains(loaded)
# dcop.variables = _build_variables(loaded, dcop)
# dcop.external_variables = _build_external_variables(loaded, dcop)
# dcop._constraints = _build_constraints(loaded, dcop)
# dcop._agents_def = _build_agents(loaded)
# dcop.dist_hints = _build_dist_hints(loaded, dcop)

def load_dcop(filename: str) -> DCOP:
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
    constraints = xmlFile.getElementsByTagName('constraint')

    for constraint in constraints:
        assignment_dict = {}
        default = None
        constraint_name = constraint.attributes['name'].value
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
                val_position[iv] = value
        constraints_dict[constraint_name] = NAryMatrixRelation(constraint_variables, values, name=constraint_name)

    return constraints_dict



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

load_dcop('input/Rnd5-5-3.xml')


#!-----TO TEST YAML VS XML FORMAT AND OUTPUT-----!

# def load_yaml_dcop(dcop_str: str) -> DCOP:
#     loaded = yaml.load(dcop_str, Loader=yaml.FullLoader)
#     if "name" not in loaded:
#         raise ValueError("Missing name in dcop string")
#     if "objective" not in loaded or loaded["objective"] not in ["min", "max"]:
#         raise ValueError("Objective is mandatory and must be min or max")
#
#     dcop = DCOP(
#         loaded["name"],
#         loaded["objective"],
#         loaded["description"] if "description" in loaded else "",
#     )
#
#     dcop.domains = _build_domains(loaded)
#     dcop.variables = _build_variables(loaded, dcop)
#     dcop._constraints = _build_constraints(loaded, dcop)
#     return dcop
#
# def _build_domains(loaded) -> Dict[str, VariableDomain]:
#     domains = {}
#     if "domains" in loaded:
#         for d_name in loaded["domains"]:
#             d = loaded["domains"][d_name]
#             values = d["values"]
#
#             if len(values) == 1 and ".." in values[0]:
#                 values = str_2_domain_values(d["values"][0])
#             d_type = d["type"] if "type" in d else ""
#             domains[d_name] = VariableDomain(d_name, d_type, values)
#
#     return domains
#
# def _build_variables(loaded, dcop) -> Dict[str, Variable]:
#     variables = {}
#     if "variables" in loaded:
#         for v_name in loaded["variables"]:
#             v = loaded["variables"][v_name]
#             domain = dcop.domain(v["domain"])
#             initial_value = v["initial_value"] if "initial_value" in v else None
#             if initial_value and initial_value not in domain.values:
#                 raise ValueError(
#                     "initial value {} is not in the domain {} "
#                     "of the variable {}".format(initial_value, domain.name, v_name)
#                 )
#
#             if "cost_function" in v:
#                 cost_expression = v["cost_function"]
#                 cost_func = ExpressionFunction(cost_expression)
#                 if "noise_level" in v:
#                     variables[v_name] = VariableNoisyCostFunc(
#                         v_name,
#                         domain,
#                         cost_func,
#                         initial_value,
#                         noise_level=v["noise_level"],
#                     )
#                 else:
#                     variables[v_name] = VariableWithCostFunc(
#                         v_name, domain, cost_func, initial_value
#                     )
#
#             else:
#                 variables[v_name] = Variable(v_name, domain, initial_value)
#     return variables
#
# def _build_constraints(loaded, dcop) -> Dict[str, RelationProtocol]:
#     constraints = {}
#     if "constraints" in loaded:
#         for c_name in loaded["constraints"]:
#             c = loaded["constraints"][c_name]
#             if "type" not in c:
#                 raise ValueError(
#                     "Error in contraints {} definition: type is "
#                     'mandatory and only "intention" is '
#                     "supported for now".format(c_name)
#                 )
#             elif c["type"] == "intention":
#                 constraints[c_name] = relation_from_str(
#                     c_name, c["function"], dcop.all_variables
#                 )
#             elif c["type"] == "extensional":
#                 values_def = c["values"]
#                 #print (values_def)
#                 default = None if "default" not in c else c["default"]
#                 if type(c["variables"]) != list:
#                     # specific case for constraint with a single variable
#                     v = dcop.variable(c["variables"].strip())
#                     values = [default] * len(v.domain)
#                     for value, assignments_def in values_def.items():
#                         if isinstance(assignments_def, str):
#                             for ass_def in assignments_def.split("|"):
#                                 iv, _ = v.domain.to_domain_value(ass_def.strip())
#                                 values[iv] = value
#                         else:
#                             values[v.domain.index(assignments_def)] = value
#
#                     constraints[c_name] = NAryMatrixRelation([v], values, name=c_name)
#                     continue
#
#                 # For constraints that depends on several variables
#                 vars = [dcop.variable(v) for v in c["variables"]]
#                 values = assignment_matrix(vars, default)
#                 print (values)
#                 for value, assignments_def in values_def.items():
#                     # can be a str like "1 2 3" or "1 2 3 | 1 3 4"
#                     # several assignment for the same value are separated with |
#                     assignments_def = assignments_def.split("|")
#                     #print (value, assignments_def)
#                     for ass_def in assignments_def:
#                         val_position = values
#                         vals_def = ass_def.split()
#                         #print(vals_def)
#                         for i, val_def in enumerate(vals_def[:-1]):
#                             iv, _ = vars[i].domain.to_domain_value(val_def.strip())
#                             val_position = val_position[iv]
#                         # value for the last variable of the assignment
#                         val_def = vals_def[-1]
#                         iv, _ = vars[-1].domain.to_domain_value(val_def.strip())
#                         val_position[iv] = value
#
#                 constraints[c_name] = NAryMatrixRelation(vars, values, name=c_name)
#
#             else:
#                 raise ValueError(
#                     "Error in contraints {} definition: type is  mandatory "
#                     'and must be "intention" or "intensional"'.format(c_name)
#                 )
#
#     return constraints

#with open('input/graph_coloring_50.yaml') as file:
    #load_yaml_dcop(file)
