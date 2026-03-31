## How to setup Google Cloud OAuth

Open Google Cloud Console and select the project you want to use, or create a new one.

Enable the Gmail API for that project.
Configure the OAuth consent screen:
Go to https://console.cloud.google.com/products and select your project

Find Google Auth platform and then > Branding
Set app name, support email, and contact email
For audience:
Choose External if you’re using a personal gmail.com account
Choose Internal only if this is a Google Workspace org app for users in that org
Finish setup
If you chose External, add yourself as a test user:
Go to Google Auth platform > Audience
Under Test users, add the Google account you’ll log in with
Create the OAuth client:
Go to Google Auth platform > Clients
Click Create Client
Choose Application type > Desktop app
Name it and create it
Download the JSON
Replace credentials.json with that downloaded JSON.
Delete token.json if it exists, so the app does a clean re-auth.
Run python main.py again and complete the browser consent flow.

The PaperCo App currently asks for these Gmail scopes from gmail_tools.py (line 25):

gmail.readonly
gmail.modify
gmail.send


## show the visual agent framework

