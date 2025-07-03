from datasets import load_dataset
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score

ds = load_dataset("SetFit/enron_spam")
df_train = ds["train"].to_pandas()
df_test = ds["test"].to_pandas()

df_train = df_train[['text', 'label']]
df_test = df_test[['text', 'label']]

nltk.download("stopwords")
stop_keys = set(stopwords.words("english"))

def ref_text(t):
    t = re.sub(r'<.*?>', '', t)
    t = re.sub(r'[^a-zA-Z\s]', '', t)
    t = t.lower()
    t = " ".join(word for word in t.split() if word not in stop_keys)
    return t

df_train["clean_text"] = df_train["text"].apply(ref_text)
df_test["clean_text"] = df_test["text"].apply(ref_text)

vec = TfidfVectorizer(max_features=5000)
X_train = vec.fit_transform(df_train['clean_text'])
X_test = vec.transform(df_test['clean_text'])

model = LogisticRegression()
model.fit(X_train, df_train['label'])

y_pred = model.predict(X_test)

accuracy = accuracy_score(df_test['label'], y_pred)
precision = precision_score(df_test['label'], y_pred)
recall = recall_score(df_test['label'], y_pred)
f1 = f1_score(df_test['label'], y_pred)

print(f"Accuracy: {accuracy:.3f}")
print(f"precision: {precision:.3f}")
print(f"recall: {recall:.3f}")
print(f"f1: {f1:.3f}")
