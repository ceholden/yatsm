from collections import namedtuple

_fields = ['method', 'index', 'score', 'process', 'pvalue', 'signif']

#: namedtuple: Structural break detection results
StructuralBreakResult = namedtuple('StructuralBreak', _fields)
