#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
CNN model wrapped in a function for comparison to BERT
Mitchell Readinger
CMPSC 445
Final Project
Resume Classifier
"""


# In[14]:


# Imports

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.data import DataLoader, Dataset
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk, re
import platform
import time


# In[11]:


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, kernel_sizes=[3,4,5], num_filters=100):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(embed_dim, num_filters, k) for k in kernel_sizes])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(len(kernel_sizes)*num_filters, num_classes)
    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0,2,1)
        x = [torch.relu(conv(x)).max(dim=2)[0] for conv in self.convs]
        x = torch.cat(x, dim=1)
        x = self.dropout(x)
        return self.fc(x)


# In[15]:


def run_cnn(
    csv_path,
    max_len=500,
    batch_size=64,
    num_epochs=20,
    embed_dim=128,
    lr=1e-3,
    random_state=42
):
    # Load data
    df = pd.read_csv(csv_path)
    df['text']  = df['Resume_str'].astype(str)
    df['label'] = df['Category'].astype(str)

    # Clean text
    def clean_text(text):
        text = re.sub(r"http\S+|www\S+", '', text)
        text = re.sub(r"[^a-zA-Z\s]", '', text)
        text = text.lower()
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    df['clean_text'] = df['text'].apply(clean_text)

    # Labels and split
    label_encoder = LabelEncoder()
    df['label_encoded'] = label_encoder.fit_transform(df['label'])

    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'],
        df['label_encoded'],
        test_size=0.2,
        random_state=random_state,
        stratify=df['label_encoded']
    )

    num_classes = len(label_encoder.classes_)

    # Tokenize + vocab
    nltk.download('stopwords')
    nltk.download('wordnet')
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def tokenize(text):
        return text.split()

    all_tokens = [token for text in df['clean_text'] for token in tokenize(text)]
    all_tokens = [lemmatizer.lemmatize(w) for w in all_tokens if w not in stop_words]

    vocab_counter = Counter(all_tokens)
    vocab = {word: i + 1 for i, (word, _) in enumerate(vocab_counter.most_common())}
    vocab_size = len(vocab) + 1

    def encode(text):
        return [vocab.get(token, 0) for token in tokenize(text)]

    # Encode & pad
    X_train_seq = [torch.tensor(encode(text), dtype=torch.long) for text in X_train]
    X_test_seq  = [torch.tensor(encode(text), dtype=torch.long) for text in X_test]

    X_train_seq = pad_sequence([x[:max_len] for x in X_train_seq], batch_first=True, padding_value=0)
    X_test_seq  = pad_sequence([x[:max_len] for x in X_test_seq],  batch_first=True, padding_value=0)

    y_train_tensor_seq = torch.tensor(y_train.values, dtype=torch.long)
    y_test_tensor_seq  = torch.tensor(y_test.values,  dtype=torch.long)

    # Dataset & loaders 
    class ResumeDataset(Dataset):
        def __init__(self, sequences, labels):
            self.sequences = sequences
            self.labels = labels
        def __len__(self):
            return len(self.labels)
        def __getitem__(self, idx):
            return self.sequences[idx], self.labels[idx]

    train_dataset_seq = ResumeDataset(X_train_seq, y_train_tensor_seq)
    test_dataset_seq  = ResumeDataset(X_test_seq,  y_test_tensor_seq)

    train_loader_seq = DataLoader(train_dataset_seq, batch_size=batch_size, shuffle=True)
    test_loader_seq  = DataLoader(test_dataset_seq,  batch_size=batch_size)

    # Training environment info 
    print("=== Training Environment Info (CNN) ===")
    print(f"Python version: {platform.python_version()}")
    print(f"OS: {platform.platform()}")

    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        gpu_props = torch.cuda.get_device_properties(current_device)
        print(f"Compute device: CUDA GPU (device {current_device})")
        print(f"GPU name: {gpu_props.name}")
        print(f"Total VRAM: {gpu_props.total_memory / (1024 ** 3):.2f} GB")
        device = torch.device("cuda")
    else:
        print("Compute device: CPU")
        print(f"CPU: {platform.processor()}")
        device = torch.device("cpu")

    print(f"\nTorch device in use: {device}")
    print("======================================\n")

    # Model / loss / optimizer
    cnn_model = TextCNN(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_classes=num_classes
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn_model.parameters(), lr=lr)

    # Training loop with epoch timings
    history = []
    epoch_times = []
    training_start_time = time.time()

    for epoch in range(num_epochs):
        cnn_model.train()
        total_loss = 0.0
        epoch_start = time.time()

        for batch_x, batch_y in train_loader_seq:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = cnn_model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        epoch_duration = time.time() - epoch_start
        epoch_times.append(epoch_duration)

        avg_loss = total_loss / len(train_loader_seq)
        history.append(avg_loss)
        print(
            f"[CNN] Epoch {epoch + 1}/{num_epochs} - "
            f"Training Loss: {avg_loss:.4f} - "
            f"Epoch Time: {epoch_duration:.2f} sec"
        )

    total_training_time = time.time() - training_start_time
    print(f"\n[CNN] Total training time: {total_training_time:.2f} sec "
          f"({total_training_time/60:.2f} min)\n")

    # Evaluation
    cnn_model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader_seq:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            outputs = cnn_model(batch_x)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_true.extend(batch_y.cpu().numpy())

    cnn_acc = accuracy_score(all_true, all_preds)
    cnn_f1  = f1_score(all_true, all_preds, average='weighted')

    print(f"[CNN] Test Accuracy: {cnn_acc:.4f}")
    print(f"[CNN] Test F1 Score: {cnn_f1:.4f}")

    # Per-class classification report 
    labels_idx = list(range(num_classes))
    cls_report = classification_report(
        all_true,
        all_preds,
        labels=labels_idx,
        target_names=label_encoder.classes_,
        output_dict=True,
        zero_division=0
    )

    # Return structure
    return {
        "model_name": "cnn",
        "model": cnn_model,
        "label_encoder": label_encoder,
        "y_true": all_true,
        "y_pred": all_preds,
        "accuracy": cnn_acc,
        "f1": cnn_f1,
        "history": history,
        "epoch_times": epoch_times,
        "total_training_time": total_training_time,
        "report": cls_report,
        "max_len": max_len,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
    }


# In[17]:


# Test function call
# Make sure to comment this out before calling from the comparison script!!!

# csv_path = r"C:\Users\mshar\Desktop\School Fall 2025\CMPSC 445\Final Project\models\data\Resume.csv"

# cnn_result = run_cnn(
#     csv_path=csv_path,
#     max_len=500,
#     batch_size=64,
#     num_epochs=15
# )


# In[ ]:




