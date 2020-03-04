from fglib import rv

class VNode:
    def __init__(self, label, rv_type):
        self.label = label
        self.rv_type = rv_type
class FNode:
    def __init__(self, label, constraintTable):
        self.label = label
        self.constraintTable = constraintTable
