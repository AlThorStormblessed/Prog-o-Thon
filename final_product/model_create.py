import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import pickle

df = pd.read_csv("data/278k_song_labelled.csv")
#df = df.sort_values(by = "popularity", ascending=False).head(10000)

X = df[["valence", "acousticness", "danceability", "energy", "instrumentalness", "liveness", "loudness", "speechiness", "tempo"]]
y = df["labels"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 7)

rfc = RandomForestClassifier(n_estimators=50)
rfc.fit(X_train, y_train)
predictions = rfc.predict(X_test)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

pickle.dump(rfc, open("Song_classifier.pkl", "wb"))