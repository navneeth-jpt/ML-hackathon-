
import joblib

model = joblib.load("../model/model.pkl")
tfidf = joblib.load("../model/vectorizer.pkl")
mlb = joblib.load("../model/mlb.pkl")

text = "New AI smartphone released"
X = tfidf.transform([text])

pred = model.predict(X)
labels = mlb.inverse_transform(pred)

print(labels)
