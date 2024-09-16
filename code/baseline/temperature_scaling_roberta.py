import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    
)
from tqdm import trange



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TemperatureScalingCalibrationModule(nn.Module):

    def __init__(self, model_path: str, tokenizer):
        super().__init__()
        self.model_path = model_path
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = tokenizer

        # the single temperature scaling parameter
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, input_ids, attention_mask): #, token_type_ids
        """forward method that returns softmax-ed confidence scores."""
        outputs = self.forward_logit(input_ids, attention_mask) #, token_type_ids
        scores = nn.functional.softmax(outputs, dim=-1)
        return scores

    def forward_logit(self, input_ids, attention_mask): #, token_type_ids
        """forward method that returns logits, to be used with cross entropy loss."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask #token_type_ids=token_type_ids
            
        ).logits
        return outputs / self.temperature

    def fit(self, dataset_tokenized, n_epochs: int = 6, batch_size: int = 64, lr: float = 0.01):
        """fits the temperature scaling parameter."""
        data_collator = DataCollatorWithPadding(self.tokenizer, padding=True)
        data_loader = DataLoader(dataset_tokenized, collate_fn=data_collator, batch_size=batch_size)

        self.freeze_base_model()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=lr)

        for epoch in trange(n_epochs):
            for examples in data_loader:
                labels = examples['labels'].long().to(device)
                input_ids = examples['input_ids'].to(device)
                attention_mask = examples['attention_mask'].to(device)
                #token_type_ids = examples['token_type_ids'].to(device)

                # standard step to perform the forward and backward step
                self.zero_grad()
                predict_proba = self.forward_logit(input_ids, attention_mask) #, token_type_ids
                loss = criterion(predict_proba, labels)
                loss.backward()
                optimizer.step()

        return self

    def freeze_base_model(self):
        """remember to freeze base model's parameters when training temperature scaler"""
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

        return self
