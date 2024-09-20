from twilio.rest import Client
import os
import pandas as pd

# Twilio credentials
account_sid = os.getenv('TWILIO_ACCOUNT_SID')
auth_token = os.getenv('TWILIO_AUTH_TOKEN')
twilio_phone_number = os.getenv('TWILIO_PHONE_NUMBER')
recipient_phone_number = ['9874563210','9874563210']  # Replace with the recipient's phone number

# Create a Twilio client
client = Client(account_sid, auth_token)

# Message to be sent
message_body = 'Hello from Twilio! This is a test message.'

try:
    # Send the message
    message = client.messages.create(
        body=message_body,
        from_=twilio_phone_number,
        to=recipient_phone_number
    )

    print(f"Message sent successfully! SID: {message.sid}")

except Exception as e:
    print(f"Error sending message: {str(e)}")
