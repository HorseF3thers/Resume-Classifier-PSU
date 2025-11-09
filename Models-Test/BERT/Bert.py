# -*- coding: utf-8 -*-
"""
Created on Fri Nov  7 19:45:54 2025

@author: Josh
"""
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
#===================================================================================================================================
#open data
df = pd.read_csv("../../../CMPSC 445/final project test/Resume.csv", encoding='utf-8', engine='python')
texts = df['Resume_str'].astype(str).tolist()
labels = df['Category'].astype(str).tolist()

#===================================================================================================================================
#encode labels which is just assigning 0,1,2,3.... for labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
num_labels = len(label_encoder.classes_)

#===================================================================================================================================
#split into train + (validation+test)
train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    texts, encoded_labels, test_size=0.2, random_state=42
)

#===================================================================================================================================
#split into validation and test
val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts, temp_labels, test_size=0.5, random_state=42
)

#===================================================================================================================================
#initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

#===================================================================================================================================
#dataset class
class ResumeDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

#===================================================================================================================================
#create dataLoaders
train_dataset = ResumeDataset(train_texts, train_labels, tokenizer)
val_dataset = ResumeDataset(val_texts, val_labels, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)

#===================================================================================================================================
#model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels).to(device)

optimizer = optim.AdamW(model.parameters(), lr=2e-5)
loss_fn = torch.nn.CrossEntropyLoss()

#===================================================================================================================================
#training loop
for epoch in range(5):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} - Training Loss: {total_loss / len(train_loader):.4f}")

#===================================================================================================================================
#create test dataLoader
test_dataset = ResumeDataset(test_texts, test_labels, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=8)

#===================================================================================================================================
#evaluation mode on test set
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

#===================================================================================================================================
#compute metrics
accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average='weighted')  # weighted for multi-class

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test F1 Score: {f1:.4f}")

#model.eval()

sample_text = "sample resume text."
encoding = tokenizer(
    sample_text,
    truncation=True,
    padding='max_length',
    max_length=256,
    return_tensors='pt'
)

input_ids = encoding['input_ids'].to(device)
attention_mask = encoding['attention_mask'].to(device)

import torch.onnx

onnx_model_path = "../../../../CMPSC 445/final project test/bert_resume_classifier.onnx"

torch.onnx.export(
    model,
    (input_ids, attention_mask),
    onnx_model_path,
    input_names=['input_ids', 'attention_mask'],
    output_names=['logits'],
    dynamic_axes={
        'input_ids': {0: 'batch_size', 1: 'sequence_length'},
        'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
        'logits': {0: 'batch_size'}
    },
    opset_version=17
)

print(f"Model saved to {onnx_model_path}")
