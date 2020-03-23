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
from queue import Queue, Empty
import threading
from threading import Thread



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

# import pyDcop.pydcop.algorithms.maxsum as ms
# import pyDcop.pydcop.computations_graph.factor_graph


#things for load_dcop
# dcop.domains = _build_domains(loaded)
# dcop.variables = _build_variables(loaded, dcop)
# dcop.external_variables = _build_external_variables(loaded, dcop)
# dcop._constraints = _build_constraints(loaded, dcop)
# dcop._agents_def = _build_agents(loaded)
# dcop.dist_hints = _build_dist_hints(loaded, dcop)

def load_dcop(filenames: str) -> DCOP:
    content = ""
    if isinstance(filenames, CollectionIterable):
        filename = filenames[0]
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

#dcop = load_dcop('input/Rnd5-5-3.xml')

#!-----TO TEST YAML VS XML FORMAT AND OUTPUT-----!
def load_yaml_dcop_from_file(filenames: Union[str, Iterable[str]]):
    """
    load a dcop from one or several files

    Parameters
    ----------
    filenames: str or iterable of str
        The dcop can the given as a single file or as several files. When
        passing an iterable of file names, their content is concatenated
        before parsing. This can be usefull when you want to define the
        agents in a separate file.

    Returns
    -------
    A DCOP object built by parsing the files

    """
    content = ""
    if isinstance(filenames, CollectionIterable):
        for filename in filenames:
            with open(filename, mode="r", encoding="utf-8") as f:
                content += f.read()
    else:
        with open(filenames, mode="r", encoding="utf-8") as f:
            content += f.read()

    if content:
        return load_yaml_dcop(content)

def load_yaml_dcop(dcop_str: str) -> DCOP:
    loaded = yaml.load(dcop_str, Loader=yaml.FullLoader)
    if "name" not in loaded:
        raise ValueError("Missing name in dcop string")
    if "objective" not in loaded or loaded["objective"] not in ["min", "max"]:
        raise ValueError("Objective is mandatory and must be min or max")

    dcop = DCOP(
        loaded["name"],
        loaded["objective"],
        loaded["description"] if "description" in loaded else "",
    )

    dcop.domains = _build_domains(loaded)
    dcop.variables = _build_variables(loaded, dcop)
    dcop._constraints = _build_constraints(loaded, dcop)
    dcop._agents_def = _build_agents(loaded)
    dcop.dist_hints = _build_dist_hints(loaded, dcop)
    return dcop

def _build_domains(loaded) -> Dict[str, VariableDomain]:
    domains = {}
    if "domains" in loaded:
        for d_name in loaded["domains"]:
            d = loaded["domains"][d_name]
            values = d["values"]

            if len(values) == 1 and ".." in values[0]:
                values = str_2_domain_values(d["values"][0])
            d_type = d["type"] if "type" in d else ""
            domains[d_name] = VariableDomain(d_name, d_type, values)

    return domains

def _build_variables(loaded, dcop) -> Dict[str, Variable]:
    variables = {}
    if "variables" in loaded:
        for v_name in loaded["variables"]:
            v = loaded["variables"][v_name]
            domain = dcop.domain(v["domain"])
            initial_value = v["initial_value"] if "initial_value" in v else None
            if initial_value and initial_value not in domain.values:
                raise ValueError(
                    "initial value {} is not in the domain {} "
                    "of the variable {}".format(initial_value, domain.name, v_name)
                )

            if "cost_function" in v:
                cost_expression = v["cost_function"]
                cost_func = ExpressionFunction(cost_expression)
                if "noise_level" in v:
                    variables[v_name] = VariableNoisyCostFunc(
                        v_name,
                        domain,
                        cost_func,
                        initial_value,
                        noise_level=v["noise_level"],
                    )
                else:
                    variables[v_name] = VariableWithCostFunc(
                        v_name, domain, cost_func, initial_value
                    )

            else:
                variables[v_name] = Variable(v_name, domain, initial_value)
    return variables

def _build_constraints(loaded, dcop) -> Dict[str, RelationProtocol]:
    constraints = {}
    if "constraints" in loaded:
        for c_name in loaded["constraints"]:
            c = loaded["constraints"][c_name]
            if "type" not in c:
                raise ValueError(
                    "Error in contraints {} definition: type is "
                    'mandatory and only "intention" is '
                    "supported for now".format(c_name)
                )
            elif c["type"] == "intention":
                constraints[c_name] = relation_from_str(
                    c_name, c["function"], dcop.all_variables
                )
            elif c["type"] == "extensional":
                values_def = c["values"]
                default = None if "default" not in c else c["default"]
                if type(c["variables"]) != list:
                    # specific case for constraint with a single variable
                    v = dcop.variable(c["variables"].strip())
                    values = [default] * len(v.domain)
                    for value, assignments_def in values_def.items():
                        if isinstance(assignments_def, str):
                            for ass_def in assignments_def.split("|"):
                                iv, _ = v.domain.to_domain_value(ass_def.strip())
                                values[iv] = value
                        else:
                            values[v.domain.index(assignments_def)] = value

                    constraints[c_name] = NAryMatrixRelation([v], values, name=c_name)
                    continue

                # For constraints that depends on several variables
                vars = [dcop.variable(v) for v in c["variables"]]
                values = assignment_matrix(vars, default)
                for value, assignments_def in values_def.items():
                    # can be a str like "1 2 3" or "1 2 3 | 1 3 4"
                    # several assignment for the same value are separated with |
                    assignments_def = assignments_def.split("|")
                    for ass_def in assignments_def:
                        val_position = values
                        vals_def = ass_def.split()
                        for i, val_def in enumerate(vals_def[:-1]):
                            iv, _ = vars[i].domain.to_domain_value(val_def.strip())
                            val_position = val_position[iv]
                        # value for the last variable of the assignment
                        val_def = vals_def[-1]
                        iv, _ = vars[-1].domain.to_domain_value(val_def.strip())
                        val_position[iv] = value

                constraints[c_name] = NAryMatrixRelation(vars, values, name=c_name)

            else:
                raise ValueError(
                    "Error in contraints {} definition: type is  mandatory "
                    'and must be "intention" or "intensional"'.format(c_name)
                )

    return constraints

def _build_agents(loaded) -> Dict[str, AgentDef]:

    # Read agents list, without creating AgentDef object yet.
    # We need the preferences to create the AgentDef objects
    agents_list = {}
    if "agents" in loaded:
        for a_name in loaded["agents"]:
            try:
                kw = loaded["agents"][a_name]
                # we accept any attribute for the agent
                # Most of the time it will be capacity and also preference but
                # any named value is valid:
                agents_list[a_name] = kw if kw else {}
            except TypeError:
                # means agents are given as a list and not a map:
                agents_list[a_name] = {}

    routes = {}
    default_route = 1
    if "routes" in loaded:
        for a1 in loaded["routes"]:
            if a1 == "default":
                default_route = loaded["routes"]["default"]
                continue
            if a1 not in agents_list:
                raise DcopInvalidFormatError("Route for unknown " "agent " + a1)
            a1_routes = loaded["routes"][a1]
            for a2 in a1_routes:
                if a2 not in agents_list:
                    raise DcopInvalidFormatError("Route for unknown " "agent " + a2)
                if (a2, a1) in routes or (a1, a2) in routes:
                    if routes[(a2, a1)] != a1_routes[a2]:
                        raise DcopInvalidFormatError(
                            "Multiple route definition r{} = {}"
                            " != r({}) = {}".format(
                                (a2, a1), routes[(a2, a1)], (a1, a2), a1_routes[a2]
                            )
                        )
                routes[(a1, a2)] = a1_routes[a2]

    hosting_costs = {}
    default_cost = 0
    default_agt_costs = {}
    if "hosting_costs" in loaded:
        costs = loaded["hosting_costs"]
        for a in costs:
            if a == "default":
                default_cost = costs["default"]
                continue
            if a not in agents_list:
                raise DcopInvalidFormatError("hosting_costs for unknown " "agent " + a)
            a_costs = costs[a]
            if "default" in a_costs:
                default_agt_costs[a] = a_costs["default"]
            if "computations" in a_costs:
                for c in a_costs["computations"]:
                    hosting_costs[(a, c)] = a_costs["computations"][c]

    # Now that we parsed all agents info, we can build the objects:
    agents = {}
    for a in agents_list:
        d = default_cost
        if a in default_agt_costs:
            d = default_agt_costs[a]
        p = {c: hosting_costs[b, c] for (b, c) in hosting_costs if b == a}

        routes_a = {a2: v for (a1, a2), v in routes.items() if a1 == a}
        routes_a.update({a1: v for (a1, a2), v in routes.items() if a2 == a})

        agents[a] = AgentDef(
            a,
            default_hosting_cost=d,
            hosting_costs=p,
            default_route=default_route,
            routes=routes_a,
            **agents_list[a]
        )
    #print (agents)

    return agents

def _build_dist_hints(loaded, dcop):
    if "distribution_hints" not in loaded:
        return None
    loaded = loaded["distribution_hints"]

    must_host, host_with = None, None
    if "must_host" in loaded:
        for a in loaded["must_host"]:
            if a not in dcop.agents:
                raise ValueError(
                    "Cannot use must_host with unknown agent " "{}".format(a)
                )
            for c in loaded["must_host"][a]:
                if c not in dcop.variables and c not in dcop.constraints:
                    raise ValueError(
                        "Cannot use must_host with unknown "
                        "variable or constraint {}".format(c)
                    )

        must_host = loaded["must_host"]

    if "host_with" in loaded:
        host_with = defaultdict(lambda: set())
        for i in loaded["host_with"]:
            host_with[i].update(loaded["host_with"][i])
            for j in loaded["host_with"][i]:
                s = {i}.union(loaded["host_with"][i])
                s.remove(j)
                host_with[j].update(s)

    return DistributionHints(
        must_host, dict(host_with) if host_with is not None else {}
    )

# with open('input/graph_coloring_3agts.yaml') as file:
#     load_yaml_dcop(file)
# def _results(status):
#     """
#     Outputs results and metrics on stdout and trace last metrics in csv
#     files if requested.
#
#     :param status:
#     :return:
#     """
#
#     metrics = orchestrator.end_metrics()
#     metrics["status"] = status
#     global end_metrics, run_metrics
#     if end_metrics is not None:
#         add_csvline(end_metrics, collect_on, metrics)
#     if run_metrics is not None:
#         add_csvline(run_metrics, collect_on, metrics)
#
#     if output_file:
#         with open(output_file, encoding="utf-8", mode="w") as fo:
#             fo.write(json.dumps(metrics, sort_keys=True, indent="  ", cls=NumpyEncoder))
#
#     print(json.dumps(metrics, sort_keys=True, indent="  ", cls=NumpyEncoder))
#
#
# dcop = None
# orchestrator = None
# INFINITY = None
#
# # Files for logging metrics
# columns = {
#     "cycle_change": [
#         "cycle",
#         "time",
#         "cost",
#         "violation",
#         "msg_count",
#         "msg_size",
#         "status",
#     ],
#     "value_change": [
#         "time",
#         "cycle",
#         "cost",
#         "violation",
#         "msg_count",
#         "msg_size",
#         "status",
#     ],
#     "period": ["time", "cycle", "cost", "violation", "msg_count", "msg_size", "status"],
# }
#
# collect_on = None
# run_metrics = None
# end_metrics = None
#
# timeout_stopped = False
# output_file = None
#
# with open('input/graph_coloring_3agts.yaml') as file:
#     dcop = load_yaml_dcop(file)
# distribution = 'oneagent'
# algo_name = 'maxsum'
# dist_module, algo_module, graph_module = _load_modules(distribution, algo_name)
# cg = graph_module.build_computation_graph(dcop)
# if dist_module is not None:
#
#     if not hasattr(algo_module, "computation_memory"):
#         algo_module.computation_memory = lambda *v, **k: 0
#     if not hasattr(algo_module, "communication_load"):
#         algo_module.communication_load = lambda *v, **k: 0
#
#     distribution = dist_module.distribute(
#         cg,
#         dcop.agents.values(),
#         hints=dcop.dist_hints,
#         computation_memory=algo_module.computation_memory,
#         communication_load=algo_module.communication_load,
#     )
# else:
#     distribution = load_dist_from_file(args.distribution)
#
# collector_queue = Queue()
# collect_t = Thread(
#     target=collect_tread, args=[collector_queue, csv_cb], daemon=True
# )
# collect_t.start()
#
# period = None
# algo = build_algo_def(algo_module, algo_name, dcop.objective, None)
# #orchestrator = run_local_thread_dcop(algo, cg, distribution, dcop, None)
# orchestrator = run_local_thread_dcop(
#     algo,
#     cg,
#     distribution,
#     dcop,
#     INFINITY,
#     collector=collector_queue,
#     collect_moment=None,
#     period=period,
#     delay=None,
#     uiport=None,
# )
# try:
#     orchestrator.deploy_computations()
#     orchestrator.run(timeout=timeout)
#     if timer:
#         timer.cancel()
#     if not timeout_stopped:
#         if orchestrator.status == "TIMEOUT":
#             _results("TIMEOUT")
#             sys.exit(0)
#         elif orchestrator.status != "STOPPED":
#             _results("FINISHED")
#             sys.exit(0)
#
#     # in case it did not stop, dump remaining threads
#
# except Exception as e:
#     logger.error(e, exc_info=1)
#     orchestrator.stop_agents(5)
#     orchestrator.stop()
#     _results("ERROR")
