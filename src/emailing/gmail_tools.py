"""
Gmail email fetcher using the Gmail API.
Authenticates with OAuth2 and retrieves unread emails from the user's inbox.
Stores credentials in `../cred/token.json` for persistent access.
"""

from email.message import EmailMessage
from email.utils import parseaddr
from pathlib import Path
from typing import Any, cast
import os
import re
from loguru import logger

from bs4 import BeautifulSoup  # For HTML parsing
import base64  # For decoding email body content

from agent_framework import tool

from google.auth.exceptions import RefreshError  # Raised when refresh fails
from google.auth.transport.requests import Request  # For refreshing tokens
from google.oauth2.credentials import Credentials as OAuthCredentials  # OAuth2 credentials
from google_auth_oauthlib.flow import InstalledAppFlow  # For OAuth2 flow
from googleapiclient.discovery import build  # building the Gmail API service

# Read, modify, and send access to Gmail
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.send",
]

# Resolve paths relative to project root for consistent execution
BASE_DIR = Path(__file__).resolve().parents[2]
CREDENTIALS_DIR = BASE_DIR / "cred"
TOKEN_PATH = CREDENTIALS_DIR / "token.json"
CLIENT_SECRETS_PATH = CREDENTIALS_DIR / "credentials.json"

# Cached authenticated Gmail address
_ACCOUNT_EMAIL: str | None = None


def _authenticate_gmail() -> Any:
    """Return authenticated Gmail API client, refreshing tokens as needed.
    
    Loads credentials from GMAIL_CREDENTIALS_JSON env var or cred/credentials.json.
    Loads token from GMAIL_TOKEN_JSON env var or cred/token.json.
    
    Container deployment requires token.json to exist BEFORE deployment
    (OAuth needs browser interaction, impossible in headless containers).
    """
    creds: OAuthCredentials | None = None

    # Load token from file or environment variable
    if TOKEN_PATH.exists():
        creds = OAuthCredentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)
    elif gmail_token := os.getenv("GMAIL_TOKEN_JSON"):
        CREDENTIALS_DIR.mkdir(parents=True, exist_ok=True)
        TOKEN_PATH.write_text(gmail_token)
        logger.info("Reconstructed token.json from Container App secret")
        creds = OAuthCredentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)

    if creds and creds.valid:
        return build("gmail", "v1", credentials=creds)

    # Refresh expired token if possible
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            CREDENTIALS_DIR.mkdir(parents=True, exist_ok=True)
            TOKEN_PATH.write_text(creds.to_json())
            logger.info("Refreshed expired token")
            return build("gmail", "v1", credentials=creds)
        except RefreshError as e:
            logger.error(f"Token refresh failed: {e}")
            creds = None

    # No valid token - need interactive OAuth flow
    if not creds or not creds.valid:
        if TOKEN_PATH.exists():
            TOKEN_PATH.unlink()

        # Load credentials from env var or file
        if gmail_creds := os.getenv("GMAIL_CREDENTIALS_JSON"):
            CREDENTIALS_DIR.mkdir(parents=True, exist_ok=True)
            CLIENT_SECRETS_PATH.write_text(gmail_creds)
            logger.info("Loaded credentials from environment variable")
        elif not CLIENT_SECRETS_PATH.exists():
            raise FileNotFoundError(
                f"Gmail credentials missing: {CLIENT_SECRETS_PATH} or GMAIL_CREDENTIALS_JSON env var"
            )

        logger.warning("="*70)
        logger.warning("INTERACTIVE OAUTH REQUIRED (needs browser, fails in container)")
        logger.warning("For containers: pre-authenticate locally and deploy token as secret")
        logger.warning("="*70)
        
        try:
            flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRETS_PATH), SCOPES)
            creds = cast(OAuthCredentials, flow.run_local_server(port=0))
        except Exception as e:
            logger.error(f"OAuth flow failed: {e}")
            logger.error("In container: ensure GMAIL_TOKEN_JSON secret exists")
            raise

    # Persist token for next run
    CREDENTIALS_DIR.mkdir(parents=True, exist_ok=True)
    TOKEN_PATH.write_text(creds.to_json())
    logger.info("Token persisted")

    return build("gmail", "v1", credentials=creds)


def _get_account_email(service: Any) -> str:
    """Return authenticated Gmail address (cached after first call)."""
    global _ACCOUNT_EMAIL
    if _ACCOUNT_EMAIL:
        return _ACCOUNT_EMAIL

    profile = service.users().getProfile(userId="me").execute()
    _ACCOUNT_EMAIL = profile.get("emailAddress", "").lower()
    return _ACCOUNT_EMAIL


def _load_reply_context(message_id: str) -> tuple[Any, dict[str, str], str]:
    """Fetch Gmail message headers and thread metadata for replies."""
    if not message_id:
        raise ValueError("Gmail message_id required for replies")

    service = _authenticate_gmail()
    original = service.users().messages().get(userId="me", id=message_id, format="full").execute()
    headers = {h["name"]: h["value"] for h in original["payload"]["headers"]}
    return service, headers, original["threadId"]


def _send_reply(service: Any, headers: dict[str, str], thread_id: str, 
                reply_body: str, html_body: str | None = None) -> dict[str, str]:
    """Create and send Gmail reply."""
    msg = EmailMessage()
    msg["To"] = headers.get("From", "")
    msg["Subject"] = "Re: " + headers.get("Subject", "")
    msg["In-Reply-To"] = headers.get("Message-ID", "")
    msg["References"] = headers.get("Message-ID", "")
    msg.set_content(reply_body)
    if html_body:
        msg.add_alternative(html_body, subtype="html")

    encoded_message = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    result = service.users().messages().send(
        userId="me", body={"raw": encoded_message, "threadId": thread_id}
    ).execute()
    return {"id": result["id"], "status": "sent"}


def _extract_body(part: dict) -> str:
    """Recursively extract body content from email parts."""
    if "data" in part.get("body", {}):
        return base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8", errors="ignore")
    if "parts" in part:
        extracted_parts = []
        for subpart in part["parts"]:
            content = _extract_body(subpart)
            if content:
                extracted_parts.append(content)
        return "\n".join(extracted_parts)
    return ""


def _sanitize_email_text(text: str) -> str:
    """Remove control characters that confuse parsers and logs."""
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)


def fetch_unread_emails(gmail_service: Any | None = None) -> list[dict]:
    """Fetch unread emails from Gmail inbox with full content."""
    gmail_service = gmail_service or _authenticate_gmail()

    messages = gmail_service.users().messages().list(
        userId="me", q="is:unread", maxResults=1
    ).execute().get("messages", [])

    account_email = _get_account_email(gmail_service)
    emails = []
    
    for msg in messages:
        full_message = gmail_service.users().messages().get(
            userId="me", id=msg["id"], format="full"
        ).execute()
        
        headers = {h["name"]: h["value"] for h in full_message["payload"]["headers"]}
        sender_email = parseaddr(headers.get("From", ""))[1].lower()

        if sender_email == account_email:
            gmail_service.users().messages().modify(
                userId="me", id=full_message["id"], body={"removeLabelIds": ["UNREAD"]}
            ).execute()
            continue

        body = _extract_body(full_message["payload"])
        soup = BeautifulSoup(body, "html.parser")
        body = _sanitize_email_text(
            soup.get_text(separator="\n", strip=True)
        )

        emails.append({
            "id": full_message["id"],
            "subject": headers.get("Subject", ""),
            "sender": headers.get("From", ""),
            "snippet": full_message.get("snippet", ""),
            "body": body,
        })

    return emails


@tool
def get_unread_emails() -> list[dict]:
    """Fetch unread emails from Gmail inbox."""
    logger.info("Fetching unread emails...")
    return fetch_unread_emails()


def mark_email_as_read(message_id: str) -> dict[str, str]:
    """Mark email as read."""
    service = _authenticate_gmail()
    service.users().messages().modify(
        userId="me", id=message_id, body={"removeLabelIds": ["UNREAD"]}
    ).execute()
    logger.info(f"Marked email {message_id} as read")
    return {"id": message_id, "status": "marked_as_read"}


def _format_reply(customer: str, lines: list[str]) -> str:
    """Format friendly reply body."""
    return "\n".join([f"Hello {customer},", "", *lines, "", "Best regards,", "PaperCo Operations"])


@tool()
def respond_confirmation_email(message_id: str, pdf_url: str | None = None) -> dict[str, str]:
    """Send order confirmation email."""
    service, headers, thread_id = _load_reply_context(message_id)

    customer = headers.get("From", "Valued Customer")
    reply_body = _format_reply(customer, [
        "Your purchase order has been confirmed.",
        "We're processing your items and will notify you once they ship.",
        f"Download invoice: {pdf_url}" if pdf_url else "Invoice link coming soon.",
        "",
        "Thank you for choosing PaperCo!",
    ])

    logger.info(f"Sending fulfillment email for {message_id}")
    return _send_reply(service, headers, thread_id, reply_body)


@tool()
def respond_unfulfillable_email(message_id: str, reason: str) -> dict[str, str]:
    """Send rejection email when order cannot be fulfilled."""
    service, headers, thread_id = _load_reply_context(message_id)

    customer = headers.get("From", "Valued Customer")
    reply_body = _format_reply(customer, [
        "Thanks for your purchase order. Unfortunately, we cannot fulfill it at this time.",
        f"Reason: {reason or 'Not specified'}\n",
        "If you have questions or alternatives, reply to this email.",
    ])

    safe_reason = (reason or "Not specified").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    html_body = (
        f"<p>Hello {customer},</p>"
        "<p>Thanks for your purchase order. Unfortunately, we cannot fulfill it at this time.</p>"
        f"<p>Reason: {safe_reason}</p>"
        "<p>If you have questions or alternatives, reply to this email.</p>"
        "<p>Best regards,<br>PaperCo Operations</p>"
    )

    logger.info(f"Sending rejection email for {message_id}")
    return _send_reply(service, headers, thread_id, reply_body, html_body)




def main() -> None:
    """Authenticate and display unread emails."""
    emails = fetch_unread_emails()
    if not emails:
        raise ValueError("No unread emails found")

    print(f"\n{len(emails)} unread email(s)")
    for email in emails:
        print(f"""
========= EMAIL SENDER =========
{email['sender']}

========= EMAIL SUBJECT =========
{email['subject']}

========= EMAIL SNIPPET =========
{email['snippet']}

========= EMAIL BODY =========
{email['body']}
""")


if __name__ == "__main__":
    main()
