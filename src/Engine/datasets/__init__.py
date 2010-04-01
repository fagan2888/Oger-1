"""
This subpackage contains several standard datasets.
"""


from datasets import (narma10, narma30, memtest,)
from speech_datasets import (analog_speech, timit_tiny,)
from timeseries_datasets import (mackey_glass, mso, lorentz,)
from grammars import (simple_pcfg,)

# clean up namespace
del datasets
del speech_datasets
del timeseries_datasets
del grammars

__all__=['narma10','narma30','mackey_glass','analog_speech','timit_tiny',' mso','simple_pcfg','lorentz','memtest']

