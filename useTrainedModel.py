import joblib
import pandas as pd
import torch
from transformers import BertTokenizerFast, BertModel, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader
from transformers import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
logging.set_verbosity_error()


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
trained_model = joblib.load("SmsSpamProject\Completed_model.joblib")
alephbert_tokenizer = BertTokenizerFast.from_pretrained(trained_model)

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