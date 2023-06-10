import imaplib
import email
from email.header import decode_header
from tensorflow.keras.models import load_model
import pickle
from flask import Flask, render_template, request

app = Flask(__name__)

model = load_model('spam_classifier_model.h5')


def preprocess_text(text):
    if text is None or not text.strip():
        return None

    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    sequences = vectorizer.transform([text]).toarray()

    return sequences


def classify_message(text):
    preprocessed_text = preprocess_text(text)

    if preprocessed_text is None:
        return 'Empty or Stop Words Only'

    predictions = model.predict(preprocessed_text)
    print(predictions)

    if predictions > 0.5:
        return 'Spam'
    else:
        return 'Non-spam'


def decode_text(text, encoding):
    if text is None:
        return None
    try:
        decoded_text = text.decode(encoding)
    except UnicodeDecodeError:
        decoded_text = text.decode('utf-8', 'ignore')
    return decoded_text


def decode_ukrainian_text(text):
    if isinstance(text, bytes):
        for encoding in ['utf-8', 'iso-8859-1', 'cp1251', 'koi8-u']:
            try:
                decoded_text = text.decode(encoding)
                break
            except UnicodeDecodeError:
                pass
        else:
            decoded_text = decode_text(text, 'utf-8')
        return decoded_text
    else:
        return text


def get_message_text(message):
    if message.is_multipart():
        for part in message.walk():
            content_type = part.get_content_type()
            if content_type == 'text/plain':
                return part.get_payload(decode=True)
    else:
        return message.get_payload(decode=True)


def retrieve_messages(imap_server, message_type=None, unseen_only=False):
    mailbox = 'INBOX'
    imap_server.select(mailbox)

    if unseen_only:
        status, data = imap_server.search(None, 'UNSEEN')
    else:
        status, data = imap_server.search(None, 'ALL')

    message_ids = data[0].split()[::-1]

    max_messages = 10

    messages = []

    for i, message_id in enumerate(message_ids):
        if i >= max_messages:
            break

        status, data = imap_server.fetch(message_id, '(RFC822)')
        raw_message = data[0][1]
        msg = email.message_from_bytes(raw_message)

        message_text = get_message_text(msg)
        decoded_text = decode_text(message_text, 'utf-8')

        classification = classify_message(decoded_text)

        if message_type is None or classification == message_type:
            sender = decode_header(msg['From'])[0][0]
            subject = decode_header(msg['Subject'])[0][0]
            date = msg['Date']

            sender = decode_ukrainian_text(sender)
            subject = decode_ukrainian_text(subject)
            decoded_text = decode_ukrainian_text(decoded_text)

            message = {
                'sender': sender,
                'subject': subject,
                'date': date,
                'classification': classification,
                'text': decoded_text
            }
            messages.append(message)

    return messages


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        imap_server = imaplib.IMAP4_SSL("imap.gmail.com")
        imap_server.login("vitalii.ivashchenko.knm.2020@lpnu.ua", "13.02.2003")

        if 'spam_button' in request.form:
            messages = retrieve_messages(imap_server, 'Spam')
        elif 'non_spam_button' in request.form:
            messages = retrieve_messages(imap_server, 'Non-spam')
        elif 'all_button' in request.form:
            messages = retrieve_messages(imap_server)
        elif 'new_button' in request.form:
            messages = retrieve_messages(imap_server, unseen_only=True)

        imap_server.logout()

        return render_template('index.html', messages=messages)

    return render_template('index.html')


if __name__ == '__main__':
    app.run()
