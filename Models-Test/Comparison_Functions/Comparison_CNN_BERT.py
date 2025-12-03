#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Comparison imports and runs both CNN and BERT models
Visualizations using various data investigation tools
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

import nltk
from IPython.display import display

from sklearn.metrics import (
    classification_report,
    confusion_matrix
)

# Specifically for the t-SNE visual
import torch
from sklearn.model_selection import train_test_split
from BERT_for_comparison import ResumeDataset



# TEAM: If the .py files are in a different folder, add that folder to sys.path here.
# Example if they live in ../models:
# sys.path.append(os.path.abspath("../models"))

from CNN_for_comparison import run_cnn
from BERT_for_comparison import run_bert

# Path to the CSV (TEAM: please update to your file structure)
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

# Class Distribution Bar Chart
# Library: Plotly Express (px.bar)
# Title: "Class Distribution in Resume Dataset"
# What it shows: Count of resumes in each Category from Resume.csv (x = category, y = count).
# Purpose: Show class imbalance before modeling.

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


# In[4]:


# Example Resume Parse Tree with NLTK

# Library: NLTK (tokenizers, POS tagger, RegexpParser)
# What it shows: A shallow chunk parse tree (NP / VP / PP / CLAUSE) for a small window
# of tokens taken from the first sentence of a single resume.
# Data: One row from df (index example_idx) using the "Resume_str" text and "Category"
# label; uses sent_tokenize : word_tokenize : pos_tag on the first sentence, then
# applies a custom chunk grammar and prints both ASCII and bracketed tree output.


# Ensure required NLTK resources are available (handles old + new names)
needed_resources = [
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),  # newer punkt tables
    ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
    ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
]

for path, name in needed_resources:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(name)

# Picks a single example resume from the dataset
example_idx = 54  # or any other row 
example_text = str(df.loc[example_idx, "Resume_str"])
example_label = df.loc[example_idx, "Category"]

print("=== Example Resume Parse Tree (NLTK) ===")
print(f"Row index: {example_idx}")
print(f"Category : {example_label}\n")

print("Raw text snippet:")
print(example_text[:500] + ("..." if len(example_text) > 500 else ""))
print("\n---\n")

# Use only the first sentence for readability
sentences = nltk.sent_tokenize(example_text)
if not sentences:
    raise ValueError("No sentences found in this resume text.")
sentence = sentences[0]
print("Sentence used for parse:")
print(sentence)
print("\n---\n")

# Tokenize and POS-tag
tokens = nltk.word_tokenize(sentence)
tagged_tokens = nltk.pos_tag(tokens)

print("POS-tagged tokens:")
print(tagged_tokens)
print("\n---\n")

# Chooses a small window of tokens to keep the tree narrow
start = 3          # where to start in tagged_tokens
window_size = 10   # how many tokens to include

tagged_for_tree = tagged_tokens[start:start + window_size]

print(f"Building parse tree from tokens {start} to {start + window_size - 1}:\n")
print(tagged_for_tree)
print("\n---\n")

# Simple chunk grammar for a shallow parse tree (NP / VP / PP / CLAUSE)
grammar = r"""
  NP:     {<DT|PRP\$>?<JJ.*>*<NN.*>+}    # determiner/possessive + adjectives + noun(s)
  PP:     {<IN><NP>}                     # preposition + NP
  VP:     {<VB.*><NP|PP|CLAUSE>+}        # verb + NP/PP/CLAUSE
  CLAUSE: {<NP><VP>}                     # NP + VP
"""

cp = nltk.RegexpParser(grammar)
tree = cp.parse(tagged_for_tree)

print("Chunk parse tree (ASCII, compact slice):\n")
tree.pretty_print()

# Bracketed representation too:
print(tree)


# In[5]:


# Get your computer ready, things start heating up here...
# Run CNN and BERT with our final chosen hyperparameters
# These are calls to the run methods in BERT_for_comparison.py and CNN_for_comparison.py

print("=== Running CNN ===")
cnn_result = run_cnn(**CNN_PARAMS)

print("\n=== Running BERT ===")
bert_result = run_bert(**BERT_PARAMS)


# In[6]:


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


# In[7]:


# Accuracy / F1 and training time comparison
# Library: Plotly Express (px.bar)
# Title: "CNN vs BERT: Accuracy and F1"
# What it shows: Side-by-side bars for accuracy and F1 for CNN vs BERT.
# Data: summary_df melted into metrics_melted with columns model, metric, value.
#
# Total Training Time Bar Chart
# Library: Plotly Express (px.bar)
# Title: "CNN vs BERT: Total Training Time (seconds)"
# What it shows: One bar per model for total training time in seconds.
# Data: summary_df["total_training_time_sec"].

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


# In[8]:


# Training Loss per Epoch Line Chart
# Library: Plotly Express (px.line)
# Title: "Training Loss per Epoch: CNN vs BERT"
# What it shows: 
# X-axis: Epoch number
# Y-axis: Training loss
# Color: Model (CNN vs BERT)
# Data: loss_df built from cnn_result["history"] and bert_result["history"].
#
# Epoch Duration per Epoch Line Chart
# Library: Plotly Express (px.line)
# Title: "Epoch Duration: CNN vs BERT"
# What it shows:
# X-axis: Epoch number
# Y-axis: Seconds per epoch
# Color: Model
# Data: time_df from cnn_result["epoch_times"] and bert_result["epoch_times"].

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


# In[9]:


# Precision / Recall / F1 heatmaps for CNN and BERT

# CNN Per-class Precision/Recall/F1 Heatmap
# Library: Plotly Express (px.imshow)
# Title: "CNN Per-class Precision / Recall / F1"
# What it shows: Heatmap where:
# Rows: class labels
# Columns: precision, recall, f1
# Color: metric value
# Data: cnn_metrics_df derived from cnn_result["report"].

# BERT Per-class Precision/Recall/F1 Heatmap
# Library: Plotly Express (px.imshow)
# Title: "BERT Per-class Precision / Recall / F1"
# What it shows: Same structure as above but for BERT.
# Data: bert_metrics_df from bert_result["report"].

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


# In[10]:


# Misclassification Sankey diagrams for CNN and BERT

# CNN Misclassification Sankey Diagram
# Library: Plotly Graph Objects (go.Sankey)
# Title: "CNN Misclassification Flow (True → Pred)"
# What it shows:
# Left set of nodes: True: <class>
# Right set of nodes: Pred: <class>
# Links: Only misclassified flows (correct predictions are removed).
# Link thickness: Number of misclassified samples moving from true class i to predicted class j.
# Data: CNN confusion matrix (cm) computed inside make_misclassification_sankey.

# BERT Misclassification Sankey Diagram
# Library: Plotly Graph Objects (go.Sankey)
# Title: "BERT Misclassification Flow (True → Pred)"
# What it shows: Same structure as CNN Sankey but using bert_result["y_true"] / ["y_pred"].


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


# In[11]:


# Confusion matrices for CNN and BERT

# CNN Confusion Matrix Heatmap
# Library: Plotly Express (px.imshow)
# Title: "CNN Confusion Matrix"
# What it shows:
# X-axis: Predicted class
# Y-axis: True class
# Cell text: counts
# Data: cnn_cm = confusion_matrix(cnn_result["y_true"], cnn_result["y_pred"], labels=labels_idx).

# BERT Confusion Matrix Heatmap
# Library: Plotly Express (px.imshow)
# Title: "BERT Confusion Matrix"
# What it shows: Same as above but for BERT (bert_cm).


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



# In[12]:


# t-SNE on BERT [CLS] embeddings (test set)

# Library:
# scikit-learn TSNE for dimensionality reduction
# Plotly Express (px.scatter) for plotting
# Title: "BERT t-SNE of [CLS] Embeddings (Test Set)"
# What it shows:
# Each point = one test-set resume’s [CLS] embedding reduced to 2D.
# X/Y: t-SNE coordinates (tsne_x, tsne_y)
# Color: label_name (class/category)
# Data: CLS vectors extracted from bert_result["model"].bert on the test set built with ResumeDataset.


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


# In[ ]:




