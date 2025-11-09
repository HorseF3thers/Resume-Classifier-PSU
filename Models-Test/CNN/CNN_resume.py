import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk, re

#===================================================================================================================================
#load and preprocess data
df = pd.read_csv("Resume.csv")
df['text'] = df['Resume_str'].astype(str)
df['label'] = df['Category'].astype(str)

def clean_text(text):
    text = re.sub(r"http\S+|www\S+", '', text)
    text = re.sub(r"[^a-zA-Z\s]", '', text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_text'] = df['text'].apply(clean_text)

#===================================================================================================================================
#encode labels
label_encoder = LabelEncoder()
df['label_encoded'] = label_encoder.fit_transform(df['label'])

#train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['label_encoded'],
    test_size=0.2, random_state=42, stratify=df['label_encoded']
)

#===================================================================================================================================
#CNN prep
num_classes = len(label_encoder.classes_)

# Tokenization, stopwords, lemmatization
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def tokenize(text):
    return text.split()

all_tokens = [token for text in df['clean_text'] for token in tokenize(text)]
all_tokens = [lemmatizer.lemmatize(w) for w in all_tokens if w not in stop_words]

#build vocabulary
vocab_counter = Counter(all_tokens)
vocab = {word: i+1 for i, (word, _) in enumerate(vocab_counter.most_common())}  # 0 reserved for padding
vocab_size = len(vocab) + 1

#encode text sequences
def encode(text):
    return [vocab.get(token, 0) for token in tokenize(text)]

X_train_seq = [torch.tensor(encode(text), dtype=torch.long) for text in X_train]
X_test_seq = [torch.tensor(encode(text), dtype=torch.long) for text in X_test]

#pad sequences to fixed length
max_len = 500
X_train_seq = pad_sequence([x[:max_len] for x in X_train_seq], batch_first=True, padding_value=0)
X_test_seq = pad_sequence([x[:max_len] for x in X_test_seq], batch_first=True, padding_value=0)

y_train_tensor_seq = torch.tensor(y_train.values, dtype=torch.long)
y_test_tensor_seq = torch.tensor(y_test.values, dtype=torch.long)

#===================================================================================================================================
#custom PyTorch dataset
class ResumeDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

train_dataset_seq = ResumeDataset(X_train_seq, y_train_tensor_seq)
test_dataset_seq = ResumeDataset(X_test_seq, y_test_tensor_seq)

#dataloaders
batch_size = 64
train_loader_seq = DataLoader(train_dataset_seq, batch_size=batch_size, shuffle=True)
test_loader_seq = DataLoader(test_dataset_seq, batch_size=batch_size)

#===================================================================================================================================
#CNN Model
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

cnn_model = TextCNN(vocab_size=vocab_size, embed_dim=128, num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)

#===================================================================================================================================
#training loop
epochs = 20
for epoch in range(epochs):
    cnn_model.train()
    total_loss = 0

    for batch_x, batch_y in train_loader_seq:
        optimizer.zero_grad()
        outputs = cnn_model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader_seq)
    print(f"Epoch {epoch+1} - Training Loss: {avg_loss:.4f}")

#===================================================================================================================================
#final evaluation
cnn_model.eval()
all_preds = []
all_true = []

with torch.no_grad():
    for batch_x, batch_y in test_loader_seq:
        outputs = cnn_model(batch_x)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.numpy())
        all_true.extend(batch_y.numpy())

cnn_acc = accuracy_score(all_true, all_preds)
cnn_f1 = f1_score(all_true, all_preds, average='weighted')

print(f"Accuracy: {cnn_acc:.4f}")
print(f"F1 Score: {cnn_f1:.4f}")
