"""
Simple Slack approval system for human-in-the-loop order confirmations.

This module posts order details to a Slack channel and blocks execution
until a human replies with 'approve' or 'deny' in the thread.
"""
import os
import re
import time
from typing import Any

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from loguru import logger


def _format_order_summary(retrieved_po: dict[str, Any]) -> str:
    """Build the Slack approval message from enriched PO data.
    
    IMPORTANT: Uses the ACTUAL field names output by the agent:
        product_name, ordered_qty, unit_price, subtotal
    """
    po_data = {k: v for k, v in retrieved_po.items()}
    
    customer_name = po_data.get("customer_name", "Unknown Customer")
    order_total = po_data.get("order_total", 0.0)
    items = po_data.get("items", [])

    item_lines: list[str] = []
    
    for item in items:
        item_data = {key: value for key, value in item.items()}
        
        # Use the ACTUAL field names the agent outputs
        qty = item_data.get("ordered_qty")
        name = item_data.get("product_name")
        price = item_data.get("unit_price")
        subtotal = item_data.get("subtotal")
        
        # Validate all fields are present before formatting
        if qty is not None and name is not None and price is not None and subtotal is not None:
            item_lines.append(
                f"- {qty}x {name} @ EUR {price:.2f} → EUR {subtotal:.2f}"
            )
        else:
            # Log which format keys we tried and what we found
            logger.error("[SLACK] ERROR: Item has wrong schema! Keys: {}", list(item_data.keys()))
            logger.error("[SLACK] Expected: ordered_qty, product_name, unit_price, subtotal")
            logger.error("[SLACK] Got: qty={}, name={}, price={}, subtotal={}", qty, name, price, subtotal)
            item_lines.append("- ERROR: Item schema mismatch")

    if not item_lines:
        item_lines.append("- No line items provided")

    items_block = "\n".join(item_lines)

    return (
        f"📦 *Order Awaiting Approval*\n\n"
        f"*Customer:* {customer_name}\n"
        f"*Total:* EUR {order_total:.2f}\n"
        f"*Items:* {len(items)}\n"
        f"{items_block}\n\n"
        f"Reply `approve` or `deny` to this message."
    )


def post_approval_request(retrieved_po: dict[str, Any]) -> str:
    """Post order details to Slack and return the message timestamp.
    
    Args:
        retrieved_po: The enriched PO data to display.
        
    Returns:
        The message timestamp (thread_ts) for polling replies.
        
    Raises:
        ValueError: If Slack credentials are missing.
        SlackApiError: If posting to Slack fails.
    """
    bot_token = os.getenv("SLACK_BOT_TOKEN")
    channel = os.getenv("SLACK_APPROVAL_CHANNEL", "C09NHPL1QAU")
    
    if not bot_token:
        raise ValueError("SLACK_BOT_TOKEN not found in environment")
    
    client = WebClient(token=bot_token)
    message_text = _format_order_summary(retrieved_po)
    
    try:
        response = client.chat_postMessage(
            channel=channel,
            text=message_text,
            mrkdwn=True,
        )
        # Return the timestamp of this message (it becomes the thread_ts)
        ts = response.get("ts")
        if not ts or not isinstance(ts, str):
            raise ValueError("No timestamp returned from Slack message post")
        
        return ts
        
    except SlackApiError as e:
        raise SlackApiError(
            f"Failed to post approval request to Slack: {e.response['error']}",
            e.response
        )


def _has_keyword(keywords: set[str], text: str) -> bool:
    """Return True if any keyword is present as a standalone word in the text.
    Used for detecting 'approve' or 'deny' in Slack replies.
    """
    return any(
        re.search(rf"\b{re.escape(keyword)}\b", text) for keyword in keywords
    )


def get_approval_from_slack(
    channel: str,
    thread_ts: str,
    timeout: int = 180,
    poll_interval: int = 2,
) -> bool:
    """Poll a Slack thread for human approval or denial.
    
    Blocks execution and checks every poll_interval seconds for a reply
    containing 'approve' or 'deny' (case-insensitive).
    
    Args:
        channel: Slack channel ID or name (e.g., 'C01234567' or '#orders').
        thread_ts: The thread timestamp to monitor (from post_approval_request).
        timeout: Maximum seconds to wait before timing out (default: 180s = 3min).
        poll_interval: Seconds between each poll (default: 2s).
        
    Returns:
        True if approved, False if denied or timeout.
        
    Raises:
        ValueError: If Slack credentials are missing.
    """
    bot_token = os.getenv("SLACK_BOT_TOKEN")
    if not bot_token:
        raise ValueError("SLACK_BOT_TOKEN not found in environment")
    
    client = WebClient(token=bot_token)
    start_time = time.time()
    
    # Approval/denial keywords (case-insensitive)
    approve_keywords = {"approve", "approved", "yes", "y", "yep", "ja", "confirm"}
    deny_keywords = {"deny", "denied", "reject", "rejected", "no", "n", "nope"}

    logger.info("[SLACK APPROVAL] Waiting for human response in Slack (timeout: {}s) | channel={} | thread={}", timeout, channel, thread_ts)

    while (time.time() - start_time) < timeout:
        try:
            # Fetch all replies in this thread
            response = client.conversations_replies(
                channel=channel,
                ts=thread_ts,
                limit=100,  # Should be enough for approval threads
            )

            messages = response.get("messages", [])

            # Debug: show how many messages we found
            if len(messages) > 1:
                logger.info("[SLACK APPROVAL] Found {} replies in thread...", len(messages) - 1)

            # Skip the first message (the original approval request)
            for msg in messages[1:]:
                text = msg.get("text", "").strip().lower()  # Normalize text from Slack for matching
                logger.debug("[SLACK APPROVAL] Checking reply: '{}'", text)

                # Check for approval in the message text by keywords
                if _has_keyword(keywords=approve_keywords,
                                text=text):
                    logger.info("[SLACK APPROVAL] ✓ Human approved the order")
                    return True

                # Check for denial in the message text by keywords
                if _has_keyword(keywords=deny_keywords,
                                text=text):

                    logger.info("[SLACK APPROVAL] ✗ Human denied the order")
                    return False

            # No decision yet, wait before next poll
            time.sleep(poll_interval)

        except SlackApiError as e:
            logger.error("[SLACK APPROVAL] Slack API error during polling: {}", e)
            time.sleep(poll_interval)
        except Exception as e:
            logger.error("[SLACK APPROVAL] Unexpected error during polling: {}", e)
            time.sleep(poll_interval)
    
    # Timeout reached with no decision
    logger.warning("[SLACK APPROVAL] ⏱ Timeout reached ({}s) - defaulting to DENY", timeout)
    return False
