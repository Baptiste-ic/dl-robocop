import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_scheduler
from datasets import load_dataset
from sklearn.metrics import accuracy_score

dataset = load_dataset("ucberkeley-dlab/measuring-hate-speech")

tokenizer = AutoTokenizer.from_pretrained("facebook/roberta-hate-speech-dynabench-r4-target")

def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)


train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(10000))  # Selecting a subset for training
val_dataset = tokenized_datasets["validation"].shuffle(seed=42).select(range(2000))  # Selecting a subset for validation

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=8)
val_loader = DataLoader(val_dataset, batch_size=8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSequenceClassification.from_pretrained("facebook/roberta-hate-speech-dynabench-r4-target")
model.to(device)


optimizer = AdamW(model.parameters(), lr=5e-6)
num_epochs = 3
num_training_steps = num_epochs * len(train_loader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Training loss: {total_loss / len(train_loader)}")

    model.eval()
    total_eval_loss = 0
    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        total_eval_loss += loss.item()
    print(f"Epoch {epoch+1}, Validation loss: {total_eval_loss / len(val_loader)}")

