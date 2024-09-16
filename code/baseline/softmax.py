import torch
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification,
)



class SoftmaxModule(nn.Module):
    """
    Add a softmax layer on top the base model. Note this does not necessarily
    mean the output score is a well-calibrated probability.
    """

    def __init__(self, model_path: str):
        super().__init__()
        self.model_path = model_path
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        scores = nn.functional.softmax(outputs.logits, dim=-1)
        return scores



#usage
# softmax_module = SoftmaxModule(model_checkpoint).to(device)
# softmax_module.eval()
# print('# of parameters: ', sum(p.numel() for p in softmax_module.parameters() if p.requires_grad)) #num params