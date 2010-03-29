"""
This subpackage contains several standard datasets.
"""


from datasets import (narma30, mackey_glass, analog_speech, timit_tiny, mso)
from grammars import (elman_grammar, simple_pcfg)

# clean up namespace
del datasets
del grammars

__all__=['narma30','mackey_glass','analog_speech','timit_tiny',' mso','elman_grammar','simple_pcfg']

