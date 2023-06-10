# Spam emails classification
## Project Description:
The given project is a spam message classification application that uses machine learning techniques to classify email messages as either spam or non-spam. The project consists of three main components: data preprocessing and feature generation, model training, and a web application for interacting with the trained model and retrieving email messages.
## Prerequisites:
To run this project, you need the following prerequisites:
1. Python installed on your system.
2. Python libraries: 'imaplib', 'email', 'tensorflow', 'flask', 'pandas', 'scikit-learn', 'imblearn', 'keras', 'matplotlib', 'pickle', 'seaborn'.
3. Access to an email account that supports IMAP (e.g., Gmail).
## Installing:
1. Clone the repository by executing the command: 
```python
git clone https://github.com/flaerys/spam_email_clasification.git
```
2. Install the required packages
## Training the model:
To train the model, open the 'nn.py' and execute it. The file contains the code for training the model, and once training is complete, the trained model will be saved as 'spam_classifier_model.h5' in the project directory.
## Running the API:
1. Execute the 'app.py'. 
2. After running the app, you can access the application by visiting http://127.0.0.1:5000/ in your browser.
## Expected results for this application:
1. Accessing the index page ('GET' request): Renders the index.html template with no messages displayed initially.
2. Submitting the form on the index page ('POST' request):
  + Clicking the "All" button: Retrieves all messages (spam and non-spam) from the email account, classifies them, and renders the index.html template with the retrieved messages displayed, including sender, date, classification, and text.
  + Clicking the "Spam" button: Retrieves spam messages from the email account, renders the index.html template with the retrieved messages displayed, including sender, date, classification, and text. 
  + Clicking the "Non-spam" button: Retrieves non-spam messages from the email account, renders the index.html template with the retrieved messages displayed, including sender, date, classification, and text.
  + Clicking the "New" button: Retrieves only unseen messages from the email account, classifies them, and renders the index.html template with the retrieved messages displayed, including sender, date, classification, and text.
## Authors:
- Tetiana Pyuryk and Vitalii Ivashchenko
