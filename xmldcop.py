#Things To Do
#import

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

    return dcop

def xml_build_domains(xmlFile) -> Dict[str, VariableDomain]:
    domains_dict = {}
    domains = xmlFile.getElementsByTagName('domain')
    for domain in domains:
        domain_name = domain.attributes['name'].value
        domain_type = 'd'
        domain_values = str_2_domain_values(domain.childNodes[0].wholeText)
        domains_dict[domain_name] = VariableDomain(domain_name, domain_type, domain_values)

    return domains_dict

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


load_dcop('input/Rnd3-2-1.xml')
