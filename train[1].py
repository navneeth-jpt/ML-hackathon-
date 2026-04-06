
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
import joblib

df = pd.read_csv("../dataset/news_dataset.csv")
df["labels"] = df["labels"].apply(lambda x: x.split(","))

mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df["labels"])

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df["text"])

model = MultiOutputClassifier(LogisticRegression())
model.fit(X,y)

joblib.dump(model,"../model/model.pkl")
joblib.dump(tfidf,"../model/vectorizer.pkl")
joblib.dump(mlb,"../model/mlb.pkl")

print("training completed")
