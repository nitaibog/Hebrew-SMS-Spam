import joblib
import pandas as pd
import torch
from transformers import  BertTokenizerFast, BertModel, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader
from transformers import logging
logging.set_verbosity_error()
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

MODEL_NAME = "onlplab/alephbert-base"

df = pd.DataFrame(columns=['Message', 'Category'])

sms_spam_data_file = open("Sms_Data/spam_data.txt", "r+", encoding='utf-8')
sms_ham_data_file = open("Sms_Data/ham_data.txt", "r+", encoding='utf-8')
spam_content = sms_spam_data_file.read()
ham_content = sms_ham_data_file.read()
sms_ham_data_list = ham_content.split(";")
sms_spam_data_list = spam_content.split(";")

for i, msg in enumerate(sms_spam_data_list):
    current_msg_df = pd.DataFrame({'Message': msg ,'Category': 1}, index=[i])
    df = pd.concat([df, current_msg_df], ignore_index=True)

for i, msg in enumerate(sms_ham_data_list):
    current_msg_df = pd.DataFrame({'Message': msg ,'Category': 0}, index=[i])
    df = pd.concat([df, current_msg_df], ignore_index=True)

X = df["Message"]
y = df["Category"]

train_texts, val_texts, train_labels, val_labels = train_test_split(list(X), list(y), test_size=0.3, random_state=42)
#val_texts, test_texts, val_labels, test_labels = train_test_split(val_texts, val_labels, test_size=0.33, random_state=42)

alephbert_tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
alephbert = BertModel.from_pretrained(MODEL_NAME)

train_encodings = alephbert_tokenizer(list(train_texts), truncation=True, padding=True)
val_encodings = alephbert_tokenizer(list(val_texts), truncation=True, padding=True)
#test_encoding = alephbert_tokenizer(list(test_texts), truncation=True, padding=True)

print("train size: ", len(train_texts))
print("validation size: ", len(val_texts))
#print("test size: ", len(test_texts))


class HebrewSMSDDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = HebrewSMSDDataset(train_encodings, train_labels)
val_dataset = HebrewSMSDDataset(val_encodings, val_labels)
#test_dataset = HebrewSMSDDataset(test_encoding, test_labels)

model = BertForSequenceClassification.from_pretrained(MODEL_NAME)
model.to(device)
model.train()


train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

optim = torch.optim.AdamW(model.parameters(), lr=5e-5)


for epoch in range(3):
    for batch in tqdm(train_loader, total=len(train_loader)):
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        loss.backward()
        optim.step()

model.eval()

val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
val_loss = 0.0
val_preds = []
val_labels = []

with torch.no_grad():
    for batch in tqdm(val_loader,total=len(val_loader)):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss, logits = outputs[:2] # get the loss and logits from the model outputs
        val_loss += loss.item()
        preds = torch.argmax(logits, axis=1).detach().cpu().numpy()
        val_preds.extend(preds)
        val_labels.extend(labels.detach().cpu().numpy())

val_accuracy = accuracy_score(val_labels, val_preds)

print("Validation Loss: {:.4f}".format(val_loss))
print("Validation Accuracy: {:.2f}%".format(val_accuracy * 100))

fileName = "Completed_model.joblib"
joblib.dump(model, fileName)
trained_model = joblib.load(fileName)

new_sms = "קוד האימות שלך מצהל הינו 699672"
encoded_sms = alephbert_tokenizer.encode_plus(new_sms, max_length=128, padding='max_length', truncation=True, return_tensors='pt')
input_ids = encoded_sms['input_ids'].to(device)
attention_mask = encoded_sms['attention_mask'].to(device)
with torch.no_grad():
    outputs = trained_model(input_ids, attention_mask=attention_mask)
    logits = outputs[0]
    predicted_label = torch.argmax(logits, axis=1).item()
    if predicted_label == 1:
        print("The SMS is spam")
    else:
        print("The SMS is not spam")