import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from imblearn.over_sampling import RandomOverSampler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv1D, GlobalMaxPooling1D
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import confusion_matrix
import seaborn as sns

data = pd.read_csv("spam.csv")

X = data["Message"]
y = data["Category"]

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X).toarray()

rus = RandomOverSampler(random_state=42)
X, y = rus.fit_resample(X, y)

X, y = shuffle(X, y, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(Dense(64, activation="relu", input_dim=X_train.shape[1]))
model.add(Dense(1, activation="sigmoid"))

# model = Sequential()
# model.add(LSTM(64, activation="relu", input_shape=(X_train.shape[1], 1)))
# model.add(Dense(1, activation='sigmoid'))
#
# model = Sequential()
# model.add(Conv1D(128, 5, activation='relu'))
# model.add(GlobalMaxPooling1D())
# model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

start_time = time.time()
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
end_time = time.time()

train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print("Точність на тренувальних даних: %.2f%%" % (train_acc * 100))
print("Точність на тестових даних: %.2f%%" % (test_acc * 100))

print("Час роботи моделі: %.2f секунд" % (end_time - start_time))

y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype("int32")

y_pred_labels = label_encoder.inverse_transform(y_pred.flatten())

cm = confusion_matrix(label_encoder.inverse_transform(y_test), y_pred_labels)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Матриця помилок")
plt.xlabel("Передбачена категорія")
plt.ylabel("Справжня категорія")
plt.show()

model.save("spam_classifier_model.h5")
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

