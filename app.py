import sqlite3
import cv2
import os
from flask import Flask, request, render_template, redirect, session, url_for
from datetime import date, datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication

#### Defining Flask App
app = Flask(__name__)

#### Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

#### Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
try:
    cap = cv2.VideoCapture(0)
except:
    cap = cv2.VideoCapture(1)

#### If these directories don't exist, create them
for directory in ['Record', 'static', 'static/faces']:
    if not os.path.isdir(directory):
        os.makedirs(directory)

Record_file_path = f'Record/Record-{datetoday}.csv'

if Record_file_path not in os.listdir('Record'):
    with open(Record_file_path, 'w') as f:
        f.write('Name,Roll,Time,Date')

#### get a number of total registered users
def totalreg():
    return len(os.listdir('static/faces'))

#### extract the face from an image
def extract_faces(img):
    if img.any():
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.3, 5)
        return face_points
    else:
        return []

def send_email_with_csv(csv_file_path, receiver_email):
    sender_email = "samidalashankar@gmail.com"  # Replace with your Gmail email address
    sender_password = "egjeucbtruziimwo"  # Replace with your Gmail password

    # Create the MIME object
    message = MIMEMultipart()
    message['From'] = sender_email
    message['To'] = receiver_email
    message['Subject'] = 'Record Report'

    # Attach CSV file
    with open(csv_file_path, "rb") as f:
        attach = MIMEApplication(f.read(), _subtype="csv")
        attach.add_header('Content-Disposition', 'attachment', filename=f.name)
        message.attach(attach)

    # Set the email body
    body = MIMEText("Please find the attached Record report.")
    message.attach(body)

    # Establish a connection with the SMTP server
    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login(sender_email, sender_password)

        # Send the email
        server.send_message(message)

    print("Email sent successfully!")

#### Identify face using ML model
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)

#### A function which trains the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')

#### Extract info from today's Record file in Record folder
def extract_Record():
    df = pd.read_csv(Record_file_path)
    names, rolls, times, date = df['Name'], df['Roll'], df['Time'], df['Date']
    l = len(df)
    return names, rolls, times, date, l

#### Add Record of a specific user
"""def add_Record(name):
    username, userid = name.split('_')[0], name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")

    # Check if the current time is within the specified range (9:30 AM to 5:30 PM)
    start_time, end_time = datetime.strptime("09:30:00", "%H:%M:%S"), datetime.strptime("21:30:00", "%H:%M:%S")
    current_datetime = datetime.strptime(current_time, "%H:%M:%S")

    # Check if the userid is not already present in the existing Record data
    df = pd.read_csv(Record_file_path)
    if str(userid) not in list(df['Roll']):
        # Append a new line to the CSV file with username, userid, current time, and date
        with open(Record_file_path, 'a') as f:
            f.write(f'\n{username},{userid},{current_time},{datetoday}')
    else:
        print(f"Record recorded for user {username}")"""
        
def add_Record(name):
    # Check if name is not empty and contains at least one underscore
    if name and '_' in name:
        username, userid = name.split('_', 1)  # Split only once to avoid issues with multiple underscores
        current_time = datetime.now().strftime("%H:%M:%S")

        # Check if the current time is within the specified range (9:30 AM to 5:30 PM)
        start_time, end_time = datetime.strptime("09:30:00", "%H:%M:%S"), datetime.strptime("21:30:00", "%H:%M:%S")
        current_datetime = datetime.strptime(current_time, "%H:%M:%S")

        # Check if the userid is not already present in the existing Record data
        df = pd.read_csv(Record_file_path)
        if str(userid) not in list(df['Roll']):
            # Append a new line to the CSV file with username, userid, current time, and date
            with open(Record_file_path, 'a') as f:
                f.write(f'\n{username},{userid},{current_time},{datetoday}')
        else:
            print(f"Record recorded for user {username}")
    else:
        print("Invalid name format: ", name)


################## ROUTING FUNCTIONS ##############################

#### Our main page
@app.route('/')
def index():
    names, rolls, times, date, l = extract_Record()
    return render_template('index.html', names=names, rolls=rolls, times=times, l=l, date=date,
                           totalreg=totalreg(), datetoday2=datetoday2)


Record_file_path = f'Record/Record-{datetoday}.csv'
roll_numbers_file_path = 'Record/data.csv'  # Replace with the actual path


def check_absent_users():
    Record_df = pd.read_csv(Record_file_path)
    roll_numbers_df = pd.read_csv(roll_numbers_file_path)

    absent_users = roll_numbers_df[~roll_numbers_df['Roll'].isin(Record_df['Roll'])]

    return absent_users['Roll'].tolist()

import time
#### This function will run when we click on Take Record Button
@app.route('/start', methods=['GET'])
def start():
    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('index.html', totalreg=totalreg(), datetoday2=datetoday2,
                               mess='There is no trained model in the static folder. Please add a new face to continue.')

    cap = cv2.VideoCapture(0)
    ret = True
    messages_printed = False  # Flag to track whether the messages have been printed

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        identified_person = "Unknown"
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            face = cv2.resize(frame[y:y + h, x:x + w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            cv2.putText(frame, f'{identified_person}', (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2)

        cv2.imshow('Record Check', frame)
        cv2.putText(frame, 'hello', (30, 30), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255))

        if not messages_printed:
            current_time = datetime.now().strftime("%H:%M:%S")
            start_time, end_time = datetime.strptime("09:30:00", "%H:%M:%S"), datetime.strptime("21:20:00",
                                                                                                "%H:%M:%S")
            current_datetime = datetime.strptime(current_time, "%H:%M:%S")

            if current_datetime < start_time:
                print("Record not started. Current time is before 9:30 AM.")
            elif current_datetime > end_time:
                print("Time over. Record capturing after 10:30 AM.")
                absent_users = check_absent_users()
                print(f"Absent users: {absent_users}")
                print("Time over. Sending mail")

                time.sleep(10)
                send_email_with_csv(Record_file_path, '')
            else:
                add_Record(identified_person)
                print("Record is being captured.")
                if current_datetime > end_time:
                    absent_users = check_absent_users()
                    print(f"Absent users: {absent_users}")
                    print("Time over. Sending mail")

            messages_printed = True

        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            break

    # if current_datetime > end_time:
    #     # Call the function to send the email with the CSV file attached
    #     send_email_with_csv(Record_file_path, 'samidalashankar@gmail.com')

    cap.release()
    cv2.destroyAllWindows()
    names, rolls, times, date, l = extract_Record()
    return render_template('navbar_logout.html', names=names, rolls=rolls, times=times, l=l,
                           totalreg=totalreg(), datetoday2=datetoday2)

#### This function will run when we add a new user
@app.route('/add', methods=['GET', 'POST'])
def add():
    newusername, newuserid = request.form['newusername'], request.form['newuserid']
    userimagefolder = f'static/faces/{newusername}_{newuserid}'
    
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)

    cap = cv2.VideoCapture(0)
    i, j = 0, 0

    while 1:
        _, frame = cap.read()
        faces = extract_faces(frame)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/50', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2,
                        cv2.LINE_AA)

            if j % 10 == 0:
                name = f'{newusername}_{i}.jpg'
                cv2.imwrite(f'{userimagefolder}/{name}', frame[y:y + h, x:x + w])
                i += 1

            j += 1

        if j == 500:
            break

        cv2.imshow('Adding new User', frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train_model()
    names, rolls, times, date, l = extract_Record()

    if totalreg() > 0:
        return redirect(url_for('index'))
    else:
        return render_template('index.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(),
                               datetoday2=datetoday2)

#### Our main function which runs the Flask App
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
