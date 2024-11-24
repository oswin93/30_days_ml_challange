# problem filter spam from collection of emails
# it is again a classification type problem
# but this is text classification problem

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# step 1 create the data

data = {
    'label': ['spam', 'ham', 'spam', 'ham', 'ham', 'spam', 'spam', 'ham', 'ham', 'spam',
              'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'spam', 'ham', 'spam', 'ham'],
    'email': [
        'Win a $1000 Walmart gift card! Click here to claim now!',
        'Hey, are you coming to the party tonight?',
        'Congratulations! You have won a free vacation to the Bahamas!',
        'Can we reschedule our meeting to 3 PM?',
        'Your Amazon order has been shipped.',
        'You have been selected for a cash prize! Call now to claim.',
        'Urgent! Your account has been compromised, please reset your password.',
        'Don’t forget about the doctor’s appointment tomorrow.',
        'Your package is out for delivery.',
        'Get rich quick by investing in this opportunity. Don’t miss out!',
        'Can you send me the latest project report?',
        'Exclusive offer! Buy one, get one free on all items.',
        'Are you free for lunch tomorrow?',
        'Claim your free iPhone now by clicking this link!',
        'I’ll call you back in 5 minutes.',
        'Get a $500 loan approved instantly. No credit check required!',
        'Hurry! Limited-time offer, act now to win a $1000 gift card.',
        'Let’s catch up over coffee this weekend.',
        'You’ve been pre-approved for a personal loan. Apply today!',
        'Meeting reminder for Monday at 10 AM.'
    ]
}

# step 2 preprocessing , convert label spam =1 and ham =0

data_df = pd.DataFrame(data)
data_df["label"] = data_df["label"].map({"ham": 0, "spam": 1})


# step 3 convert the text data into numerical features that machine learning understands using CountVectorizer

vectorizer = CountVectorizer(stop_words="english")
X = vectorizer.fit_transform(data_df["email"]) # convert the text into bag of words

# print("vocabulary: \n", vectorizer.get_feature_names_out())
# print("count matrix: \n", X.toarray())

# step 4: Split the data we are splitting into 80-20 ratio

X_train, X_test, y_train, y_test = train_test_split(X, data_df["label"], test_size=0.2, random_state=42)


# step 5 create and train model
model = MultinomialNB()
model.fit(X_train, y_train)

# step 6 make predictions and evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)


print("accuracy is :", accuracy)
print("confusion matrix: \n",conf_matrix)

# step 7 Visualize

plt.figure(figsize=(10,7)) #window size
sns.heatmap(conf_matrix, annot= True, fmt="d", cmap="Blues", xticklabels=["ham", "spam"], yticklabels=["ham", "spam"])
plt.title("confusion matrix")
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
plt.show()