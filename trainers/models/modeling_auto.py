from collections import OrderedDict

from transformers.models.auto.auto_factory import _BaseAutoModelClass, auto_class_update

# from transformers.models.bert import BertForMultipleChoice
# from transformers.models.deberta_v2 import DebertaV2ForMultipleChoice

from models.modeling_bert import BertConfig, BertForSpanClassification
from models.modeling_distilbert import DistilBertConfig, DistilBertForSpanClassification
from models.modeling_roberta import RobertaConfig, RobertaForSpanClassification
from models.modeling_xlm_roberta import XLMRobertaConfig, XLMRobertaForSpanClassification
from models.modeling_deberta import DebertaConfig, DebertaForSpanClassification
from models.modeling_deberta_v2 import DebertaV2Config, DebertaV2ForSpanClassification

CONFIG_MAPPING = OrderedDict([
    ("bert", BertConfig),
    ("deberta", DebertaConfig),
    ("deberta-v2", DebertaV2Config),
    ("distilbert", DistilBertConfig),
    ("roberta", RobertaConfig),
    ("xlm-roberta", XLMRobertaConfig),
])

MODEL_FOR_TOKEN_PAIR_CLASSIFICATION_MAPPING = OrderedDict([
    ("bert", BertForSpanClassification),
    ("deberta", DebertaForSpanClassification),
    ("deberta-v2", DebertaV2ForSpanClassification),
    ("distilbert", DistilBertForSpanClassification),
    ("roberta", RobertaForSpanClassification),
    ("xlm-roberta", XLMRobertaForSpanClassification),
])


class AutoMapping:
    def __init__(self, config_mapping, model_mapping):
        self._config_mapping = config_mapping
        self._model_mapping = model_mapping
        self._reverse_config_mapping = {v: k for k, v in config_mapping.items()}

    def __getitem__(self, key):
        key = self._reverse_config_mapping[key]
        return self._model_mapping[key]

    def keys(self):
        return self._config_mapping.values()

class AutoModelForSpanClassification(_BaseAutoModelClass):
    _model_mapping = AutoMapping(CONFIG_MAPPING, MODEL_FOR_TOKEN_PAIR_CLASSIFICATION_MAPPING)

AutoModelForSpanClassification = auto_class_update(AutoModelForSpanClassification, head_doc="token pair classification")
