
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass

from transformers.activations import get_activation
from transformers.modeling_outputs import ModelOutput, MultipleChoiceModelOutput

from transformers.models.xlm_roberta.modeling_xlm_roberta import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    XLM_ROBERTA_START_DOCSTRING,
    XLM_ROBERTA_INPUTS_DOCSTRING,
    XLMRobertaPreTrainedModel,
    XLMRobertaConfig,
    XLMRobertaModel,
)

import torch
from torch import nn

@dataclass
class SpanClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@add_start_docstrings(
    """
    RoBERTa Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    XLM_ROBERTA_START_DOCSTRING,
)
class XLMRobertaForSpanClassification(XLMRobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.roberta = XLMRobertaModel(config, add_pooling_layer=False)
        classifier_dropout = (config.classifier_dropout
                              if config.classifier_dropout is not None else config.hidden_dropout_prob)
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(XLM_ROBERTA_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        index1: Optional[torch.Tensor] = None,
        index2: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SpanClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        assert index1 is not None and index2 is not None
        index1 = index1.view(-1, 1, 1).repeat(1, 1, outputs[0].size(2))
        index2 = index2.view(-1, 1, 1).repeat(1, 1, outputs[0].size(2))
        pooled_output = outputs[0].gather(1, index1).squeeze(1) + outputs[0].gather(1, index2).squeeze(1)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SpanClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# class XLMRobertaForMultipleChoice(XLMRobertaPreTrainedModel):
#     _keys_to_ignore_on_load_missing = [r"position_ids"]

#     def __init__(self, config):
#         super().__init__(config)

#         self.roberta = XLMRobertaModel(config, add_pooling_layer=False)
#         self.dense = nn.Linear(config.hidden_size, config.hidden_size)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.classifier = nn.Linear(config.hidden_size, 1)

#         # Initialize weights and apply final processing
#         self.post_init()

#     @add_start_docstrings_to_model_forward(
#         XLM_ROBERTA_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
#     )
#     def forward(
#         self,
#         input_ids: Optional[torch.LongTensor] = None,
#         token_type_ids: Optional[torch.LongTensor] = None,
#         attention_mask: Optional[torch.FloatTensor] = None,
#         labels: Optional[torch.LongTensor] = None,
#         position_ids: Optional[torch.LongTensor] = None,
#         head_mask: Optional[torch.FloatTensor] = None,
#         inputs_embeds: Optional[torch.FloatTensor] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[Tuple[torch.Tensor], MultipleChoiceModelOutput]:
#         r"""
#         labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
#             Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
#             num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
#             `input_ids` above)
#         """
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict
#         num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]  # type: ignore

#         flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
#         flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
#         flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
#         flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
#         flat_inputs_embeds = (
#             inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
#             if inputs_embeds is not None
#             else None
#         )

#         outputs = self.roberta(
#             flat_input_ids,
#             position_ids=flat_position_ids,
#             token_type_ids=flat_token_type_ids,
#             attention_mask=flat_attention_mask,
#             head_mask=head_mask,
#             inputs_embeds=flat_inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )
#         sequence_output = outputs[0]

#         pooled_output = sequence_output[:, 0, :]  # take <s> token (equiv. to [CLS])
#         pooled_output = self.dropout(pooled_output)
#         pooled_output = self.dense(pooled_output)
#         pooled_output = get_activation("gelu")(pooled_output)
#         pooled_output = self.dropout(pooled_output)

#         logits = self.classifier(pooled_output)
#         reshaped_logits = logits.view(-1, num_choices)

#         loss = None
#         if labels is not None:
#             loss_fct = nn.CrossEntropyLoss()
#             loss = loss_fct(reshaped_logits, labels)

#         if not return_dict:
#             output = (reshaped_logits,) + outputs[2:]
#             return ((loss,) + output) if loss is not None else output

#         return MultipleChoiceModelOutput(
#             loss=loss,
#             logits=reshaped_logits,
#             hidden_states=outputs.hidden_states,
#             attentions=outputs.attentions,
#         )
