# Hebrew SMS Spam Prediction

**Hebrew SMS Spam Detector**

This project is a Hebrew SMS spam detector, using a fine-tuned Aleph-Bert model. The model was trained on a dataset consisting of 171 legitimate messages (ham) and 100 spam messages, for a total of 271 SMS messages.

**Aleph-Bert**
Aleph-Bert is a pre-trained Hebrew language model based on the BERT architecture. The Aleph-Bert model was fine-tuned on a large corpus of Hebrew text.

**Fine-tuning**
To fine-tune the Aleph-Bert model for the SMS spam detection task, i first built a dataset by labeling my own SMS messages as either ham or spam. i then fine-tuned the model on this dataset using the PyTorch framework.

**Test Accuracy**
The trained model achieved a test accuracy of 96%, which suggests that it can accurately distinguish between legitimate and spam SMS messages in Hebrew.
