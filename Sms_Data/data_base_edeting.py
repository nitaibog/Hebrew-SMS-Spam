sms_spam_data_file = open("spam_data.txt", "r+", encoding='utf-8')
content = sms_spam_data_file.read()
content = content.replace("]", ";")
sms_spam_data_file.seek(0)
sms_spam_data_file.write(content)
list_spam = content.split(";")
for i in list_spam[:10]:
    print(i)


