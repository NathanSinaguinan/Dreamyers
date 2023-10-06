import chardet

# Detect the encoding of the CSV file
with open('SMSDataset.csv', 'rb') as f:
    result = chardet.detect(f.read())

print(result['encoding'])
