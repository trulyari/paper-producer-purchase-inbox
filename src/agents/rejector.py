"""Rejector agent configuration.

For now, Slack notifications are sent only for fulfilled orders (via the fulfiller).
This agent does not notify Slack. It will be extended later if needed.
"""

from typing import Annotated

from agent_framework import Agent
from pydantic import BaseModel, ConfigDict, Field

from agents.base import chat_client

from emailing.gmail_tools import respond_unfulfillable_email



class RejectResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    email_id: Annotated[
        str,
        Field(
            description="The Gmail message ID that received the rejection response"
        )
    ]
    rejection_messaging_complete: Annotated[
        bool,
        Field(
            description="Whether the rejection was handled successfully"
        )
    ]


rejector = Agent(
    client=chat_client,
    name="rejector",
    instructions=(
        "You handle polite rejection emails for orders marked UNFULFILLABLE.\n\n"
        "Work through these steps:\n"
        "1. Read the Decision object and capture the reason field.\n"
        "2. Pull email_id from the RetrievedPO (input_payload).\n"
        "3. Draft a clear rejection note covering:\n"
        "   • Why we cannot fulfill the order (stock, credit, etc.)\n"
        "   • Suggested next steps the customer can take\n"
        "4. Call respond_unfulfillable_email(message_id, reason).\n\n"
        "Return RejectResult with email_id=input_payload.email_id exactly, unchanged, and "
        "rejection_messaging_complete=true after the email is sent."
    ),
    tools=[
        respond_unfulfillable_email,
    ],
    default_options={"response_format": RejectResult},
)
