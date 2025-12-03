#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
BERT model wrapped in a function for comparison to CNN
Mitchell Readinger
CMPSC 445
Final Project
Resume Classifier
"""


# In[ ]:


# Imports

import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

import platform
import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE

import plotly.graph_objects as go
import plotly.express as px


# In[5]:


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


# In[9]:


def run_bert(
    csv_path,
    max_len=256,
    batch_size=8,
    num_epochs=5,
    lr=2e-5,
    random_state=42,
    onnx_model_path=None # Team: Point to a directory if you want the ONNX file, but know that it will be overwritten each call during comparison
):
    # Load CSV
    df = pd.read_csv(
        csv_path,
        encoding="utf-8",
        engine="python",
    )
    texts = df['Resume_str'].astype(str).tolist()
    labels_str = df['Category'].astype(str).tolist()

    # Encode labels and split
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels_str)
    num_labels = len(label_encoder.classes_)

    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts, encoded_labels, test_size=0.2, random_state=random_state
    )

    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts, temp_labels, test_size=0.5, random_state=random_state
    )

    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Datasets + loaders 
    train_dataset = ResumeDataset(train_texts, train_labels, tokenizer, max_len=max_len)
    val_dataset   = ResumeDataset(val_texts,  val_labels,  tokenizer, max_len=max_len)
    test_dataset  = ResumeDataset(test_texts, test_labels, tokenizer, max_len=max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size)

    # Device + model + optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased', num_labels=num_labels
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # Training loop with timing 
    print("=== Training Environment Info ===")
    print(f"Python version: {platform.python_version()}")
    print(f"OS: {platform.platform()}")

    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        gpu_props = torch.cuda.get_device_properties(current_device)
        print(f"Compute device: CUDA GPU (device {current_device})")
        print(f"GPU name: {gpu_props.name}")
        print(f"Total VRAM: {gpu_props.total_memory / (1024 ** 3):.2f} GB")
    else:
        print("Compute device: CPU")
        print(f"CPU: {platform.processor()}")

    print(f"\nTorch device in use: {device}")
    print("================================\n")

    history = []
    epoch_times = []
    training_start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        epoch_start = time.time()

        for batch in train_loader:
            optimizer.zero_grad()

            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_batch   = batch['labels'].to(device)

            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                labels=labels_batch
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        epoch_duration = time.time() - epoch_start
        epoch_times.append(epoch_duration)

        avg_loss = total_loss / len(train_loader)
        history.append(avg_loss)
        print(
            f"[BERT] Epoch {epoch + 1}/{num_epochs} - "
            f"[BERT] Training Loss: {avg_loss:.4f} - "
            f"[BERT] Epoch Time: {epoch_duration:.2f} sec"
        )

    total_training_time = time.time() - training_start_time
    print(f"\nTotal training time: {total_training_time:.2f} sec "
          f"({total_training_time/60:.2f} min)\n")

    # Evaluate on test set
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_batch   = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test F1 Score: {f1:.4f}")

    # Per-class report for later visualizations/comparisons
    labels_idx = np.arange(num_labels)
    cls_report = classification_report(
        all_labels,
        all_preds,
        labels=labels_idx,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0
    )

    # ONNX export 
    if onnx_model_path is not None:
        sample_text = "sample resume text."
        encoding = tokenizer(
            sample_text,
            truncation=True,
            padding='max_length',
            max_length=max_len,
            return_tensors='pt'
        )

        input_ids_ex      = encoding['input_ids'].to(device)
        attention_mask_ex = encoding['attention_mask'].to(device)

        torch.onnx.export(
            model,
            (input_ids_ex, attention_mask_ex),
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

    # Return everything 
    return {
        "model_name": "bert",
        "model": model,
        "tokenizer": tokenizer,
        "label_encoder": label_encoder,
        "y_true": all_labels,
        "y_pred": all_preds,
        "accuracy": accuracy,
        "f1": f1,
        "epoch_times": epoch_times,
        "total_training_time": total_training_time,
        "report": cls_report,
        "max_len": max_len,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "history": history,
    }


# In[ ]:


# Test function call
# Make sure to comment this our before calling from the comparison!!!

# csv_path = r"C:\Users\mshar\Desktop\School Fall 2025\CMPSC 445\Final Project\models\data\Resume.csv"
# bert_result = run_bert(csv_path=csv_path)


# In[ ]:




