#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Comparison imports and runs both CNN and BERT models
Mitch Readinger
CMPSC 445
Final Project
Resume Classifier
"""


# In[2]:


# Imports and config

import os
import sys
import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

# Specifically for the t-SNE visual
import torch
from sklearn.model_selection import train_test_split
from BERT_for_comparison import ResumeDataset



# If the .py files are in a different folder, add that folder to sys.path here.
# Example if they live in ../models:
# sys.path.append(os.path.abspath("../models"))

from CNN_for_comparison import run_cnn
from BERT_for_comparison import run_bert

# Path to the CSV (team: please update to your file structure)
csv_path = r"C:\Users\mshar\Desktop\School Fall 2025\CMPSC 445\Final Project\models\data\Resume.csv"

# Global hyperparameters for this comparison run
# Meant to represent the best case training scenarios from each model
CNN_PARAMS = {
    "csv_path": csv_path,
    "max_len": 500,
    "batch_size": 64,
    "num_epochs": 15,
}

BERT_PARAMS = {
    "csv_path": csv_path,
    "max_len": 256,
    "batch_size": 8,
    "num_epochs": 5,
}


# In[3]:


# Peek at the CSV

df = pd.read_csv(csv_path)
print("Shape:", df.shape)
display(df.head())

print("\nColumns:", df.columns.tolist())

print("\nCategory value counts:")
display(df['Category'].value_counts())

# Build a clean counts DataFrame
counts = df['Category'].value_counts().reset_index()
counts.columns = ['Category', 'count']

fig_eda = px.bar(
    counts,
    x='Category',
    y='count',
    labels={'Category': 'Category', 'count': 'Count'},
    title='Class Distribution in Resume Dataset'
)
fig_eda.update_layout(xaxis_tickangle=-45)
fig_eda.show()


# In[ ]:


# Run CNN and BERT with our final chosen hyperparameters
# These are calls to the run methods in BERT_for_comparison.py and CNN_for_comparison.py
# Get your computer ready, things start heating up here...

print("=== Running CNN ===")
cnn_result = run_cnn(**CNN_PARAMS)

print("\n=== Running BERT ===")
bert_result = run_bert(**BERT_PARAMS)


# In[4]:


# Summary metrics table for CNN vs BERT

summary_rows = [
    {
        "model": "CNN",
        "max_len": cnn_result["max_len"],
        "batch_size": cnn_result["batch_size"],
        "num_epochs": cnn_result["num_epochs"],
        "accuracy": cnn_result["accuracy"],
        "f1": cnn_result["f1"],
        "total_training_time_sec": cnn_result["total_training_time"],
    },
    {
        "model": "BERT",
        "max_len": bert_result["max_len"],
        "batch_size": bert_result["batch_size"],
        "num_epochs": bert_result["num_epochs"],
        "accuracy": bert_result["accuracy"],
        "f1": bert_result["f1"],
        "total_training_time_sec": bert_result["total_training_time"],
    },
]

summary_df = pd.DataFrame(summary_rows)
summary_df


# In[5]:


# Accuracy / F1 and training time comparison

# Melt accuracy/F1 for plotting
metrics_melted = summary_df.melt(
    id_vars=["model"],
    value_vars=["accuracy", "f1"],
    var_name="metric",
    value_name="value"
)

fig_metrics = px.bar(
    metrics_melted,
    x="model",
    y="value",
    color="metric",
    barmode="group",
    title="CNN vs BERT: Accuracy and F1",
    text_auto=".3f"
)
fig_metrics.update_layout(yaxis_title="Score")
fig_metrics.show()

# Training time
fig_time = px.bar(
    summary_df,
    x="model",
    y="total_training_time_sec",
    title="CNN vs BERT: Total Training Time (seconds)",
    text_auto=".1f"
)
fig_time.update_layout(yaxis_title="Time (sec)")
fig_time.show()


# In[7]:


# Epoch loss and timing comparison

# Epoch indices (1..N)
cnn_epochs = list(range(1, len(cnn_result["history"]) + 1))
bert_epochs = list(range(1, len(bert_result["history"]) + 1))

# Loss line chart
loss_df = pd.DataFrame({
    "epoch": cnn_epochs + bert_epochs,
    "loss": cnn_result["history"] + bert_result["history"],
    "model": ["CNN"] * len(cnn_epochs) + ["BERT"] * len(bert_epochs),
})

fig_loss = px.line(
    loss_df,
    x="epoch",
    y="loss",
    color="model",
    markers=True,
    title="Training Loss per Epoch: CNN vs BERT"
)
fig_loss.update_layout(xaxis_title="Epoch", yaxis_title="Training Loss")
fig_loss.show()

# Epoch time line chart
cnn_et = cnn_result["epoch_times"]
bert_et = bert_result["epoch_times"]

time_df = pd.DataFrame({
    "epoch": list(range(1, len(cnn_et) + 1)) + list(range(1, len(bert_et) + 1)),
    "epoch_time_sec": cnn_et + bert_et,
    "model": ["CNN"] * len(cnn_et) + ["BERT"] * len(bert_et),
})

fig_time_epochs = px.line(
    time_df,
    x="epoch",
    y="epoch_time_sec",
    color="model",
    markers=True,
    title="Epoch Duration: CNN vs BERT"
)
fig_time_epochs.update_layout(xaxis_title="Epoch", yaxis_title="Time (sec)")
fig_time_epochs.show()


# In[11]:


# Precision / Recall / F1 heatmaps for CNN and BERT

def per_class_metrics_df(result):
    report = result["report"]
    classes = result["label_encoder"].classes_
    rows = []
    for cls in classes:
        if cls in report:
            rows.append({
                "class": cls,
                "precision": report[cls]["precision"],
                "recall": report[cls]["recall"],
                "f1": report[cls]["f1-score"],
            })
    return pd.DataFrame(rows)

cnn_metrics_df  = per_class_metrics_df(cnn_result)
bert_metrics_df = per_class_metrics_df(bert_result)

# CNN heatmap
fig_cnn_metrics = px.imshow(
    cnn_metrics_df.set_index("class")[["precision", "recall", "f1"]],
    text_auto=".2f",
    aspect="auto",
    color_continuous_scale="Viridis",
    labels=dict(x="Metric", y="Class", color="Score"),
    title="CNN Per-class Precision / Recall / F1"
)
fig_cnn_metrics.update_layout(width=900, height=900)
fig_cnn_metrics.show()

# BERT heatmap
fig_bert_metrics = px.imshow(
    bert_metrics_df.set_index("class")[["precision", "recall", "f1"]],
    text_auto=".2f",
    aspect="auto",
    color_continuous_scale="Viridis",
    labels=dict(x="Metric", y="Class", color="Score"),
    title="BERT Per-class Precision / Recall / F1"
)
fig_bert_metrics.update_layout(width=900, height=900)
fig_bert_metrics.show()


# In[12]:


# Misclassification Sankey diagrams for CNN and BERT

from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go

classes = list(cnn_result["label_encoder"].classes_)
num_classes = len(classes)
labels_idx = list(range(num_classes))

def make_misclassification_sankey(y_true, y_pred, classes, title):
    cm = confusion_matrix(y_true, y_pred, labels=labels_idx)

    # Node labels: True: X (0..n-1), Pred: X (n..2n-1)
    node_labels = [f"True: {c}" for c in classes] + [f"Pred: {c}" for c in classes]

    sources = []
    targets = []
    values  = []

    for i in range(num_classes):
        for j in range(num_classes):
            if i == j:
                continue  # skip correct classifications; only show misclassifications
            count = cm[i, j]
            if count > 0:
                sources.append(i)            # True node index
                targets.append(num_classes + j)  # Pred node index
                values.append(count)

    if not values:
        print(f"No misclassifications for {title} – Sankey will be empty.")
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            label=node_labels
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values
        )
    )])

    fig.update_layout(
        title_text=title,
        font_size=10,
        width=1000,
        height=600
    )
    fig.show()

# CNN
make_misclassification_sankey(
    cnn_result["y_true"],
    cnn_result["y_pred"],
    classes,
    title="CNN Misclassification Flow (True → Pred)"
)

# BERT
make_misclassification_sankey(
    bert_result["y_true"],
    bert_result["y_pred"],
    classes,
    title="BERT Misclassification Flow (True → Pred)"
)


# In[6]:


# Confusion matrices for CNN and BERT

# Get class names from both encoders and make sure they match as sets
classes_cnn  = list(cnn_result["label_encoder"].classes_)
classes_bert = list(bert_result["label_encoder"].classes_)

assert set(classes_cnn) == set(classes_bert), "CNN and BERT classes don't match."

# Choose an order (here I'm using CNN's order)
classes = classes_cnn
num_classes = len(classes)

# The integer labels are 0..num_classes-1
labels_idx = list(range(num_classes))

# Force confusion_matrix to use all labels in this order
cnn_cm = confusion_matrix(
    cnn_result["y_true"],
    cnn_result["y_pred"],
    labels=labels_idx
)

bert_cm = confusion_matrix(
    bert_result["y_true"],
    bert_result["y_pred"],
    labels=labels_idx
)

fig_cnn_cm = px.imshow(
    cnn_cm,
    x=classes,
    y=classes,
    color_continuous_scale="Blues",
    text_auto=True,
    labels=dict(x="Predicted", y="True", color="Count"),
    title="CNN Confusion Matrix",
    width=900,   # adjust size
    height=900   
)
fig_cnn_cm.update_xaxes(side="top")
fig_cnn_cm.show()

fig_bert_cm = px.imshow(
    bert_cm,
    x=classes,
    y=classes,
    color_continuous_scale="Blues",
    text_auto=True,
    labels=dict(x="Predicted", y="True", color="Count"),
    title="BERT Confusion Matrix",
    width=900,   # adjust size
    height=900   
)
fig_bert_cm.update_xaxes(side="top")
fig_bert_cm.show()



# In[15]:


# t-SNE on BERT [CLS] embeddings (test set)
# Had to bring over even though I can't do this easily on CNN
# Looks too cool to skip...

from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

device = next(bert_result["model"].parameters()).device
model_bert = bert_result["model"].bert   # underlying BertModel
tokenizer  = bert_result["tokenizer"]
label_encoder = bert_result["label_encoder"]

# Rebuild texts and encoded labels using the same encoder
df_full = pd.read_csv(csv_path)
texts_full  = df_full['Resume_str'].astype(str).tolist()
labels_full = df_full['Category'].astype(str).tolist()

encoded_labels_full = label_encoder.transform(labels_full)

# Recreate the same splits as in run_bert (train/temp, then val/test)
train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    texts_full, encoded_labels_full, test_size=0.2, random_state=42
)
val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts, temp_labels, test_size=0.5, random_state=42
)

test_dataset_tsne = ResumeDataset(test_texts, test_labels, tokenizer, max_len=bert_result["max_len"])
test_loader_tsne  = DataLoader(test_dataset_tsne, batch_size=16, shuffle=False)

# Collect CLS embeddings
model_bert.eval()
cls_embeddings = []
y_test = []

with torch.no_grad():
    for batch in test_loader_tsne:
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels_batch   = batch['labels'].to(device)

        outputs = model_bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # [batch_size, seq_len, hidden_dim]
        cls_emb = last_hidden_state[:, 0, :]           # [CLS] token embedding
        cls_embeddings.append(cls_emb.cpu().numpy())
        y_test.extend(labels_batch.cpu().numpy())

cls_embeddings = np.vstack(cls_embeddings)
y_test = np.array(y_test)

print("CLS embeddings shape:", cls_embeddings.shape)

# t-SNE to 2D
tsne = TSNE(n_components=2, random_state=42, init="random", perplexity=30)
tsne_coords = tsne.fit_transform(cls_embeddings)

tsne_df = pd.DataFrame({
    "tsne_x": tsne_coords[:, 0],
    "tsne_y": tsne_coords[:, 1],
    "label_idx": y_test,
})
tsne_df["label_name"] = label_encoder.inverse_transform(tsne_df["label_idx"])

fig_tsne = px.scatter(
    tsne_df,
    x="tsne_x",
    y="tsne_y",
    color="label_name",
    title="BERT t-SNE of [CLS] Embeddings (Test Set)",
    hover_data=["label_name"],
)
fig_tsne.update_layout(width=900, height=800)
fig_tsne.show()


# In[1]:


# DANGER CELL
# MAY TAKE A LONG TIME TO RUN THESE TESTS

# Hyperparameter sweep experiments
# Passing some test parameters to the ..._for_comparison models 
# Show the results in a chart
# Runs both models multiple times with different max_len and num_epochs

experiments = []

for max_len in [128, 256]:
    for num_epochs in [5, 10]:
        print(f"\n=== Running CNN (max_len={max_len}, num_epochs={num_epochs}) ===")
        cnn_res = run_cnn(
            csv_path=csv_path,
            max_len=max_len,
            num_epochs=num_epochs,
            batch_size=CNN_PARAMS["batch_size"]
        )
        experiments.append(cnn_res)

        print(f"\n=== Running BERT (max_len={max_len}, num_epochs={num_epochs}) ===")
        bert_res = run_bert(
            csv_path=csv_path,
            max_len=max_len,
            num_epochs=num_epochs,
            batch_size=BERT_PARAMS["batch_size"]
        )
        experiments.append(bert_res)

# Build a summary DataFrame of all experiment runs
rows = []
for r in experiments:
    rows.append({
        "model": r["model_name"],
        "max_len": r["max_len"],
        "num_epochs": r["num_epochs"],
        "accuracy": r["accuracy"],
        "f1": r["f1"],
    })

exp_results_df = pd.DataFrame(rows)
exp_results_df


# In[ ]:




