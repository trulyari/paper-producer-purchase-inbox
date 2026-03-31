from typing import Annotated, Literal

from agent_framework import Agent
from pydantic import BaseModel, Field

from agents.base import chat_client
from agents.retriever import RetrievedPO


class Decision(BaseModel):
    status: Annotated[
        Literal[
            "FULFILLABLE",
            "UNFULFILLABLE"
        ],
        Field(
            description="Whether the order can be fulfilled or not"
        )
    ]
    reason: Annotated[
        str,
        Field(
            description="Explanation for the final fulfillment decision"
        )
    ]
    input_payload: Annotated[
        RetrievedPO,
        Field(
            description="The original input RetrievedPO that is being evaluated"
        )
    ]


decider = Agent(
    client=chat_client,
    name="decider",
    instructions=(
        "You are the fulfillment decision authority. Evaluate whether a RetrievedPO can be fulfilled.\n\n"
        "Decision criteria:\n"
        "1. Inventory check: Every item must have product_in_stock=True (this computed field has checked: "
        "product_qty_available >= ordered_qty)\n\n"
        "2. Credit check: Customer must have sufficient credit:\n"
        "   - customer_can_order_with_credit must be True (customer_available_credit >= order_total)\n"
        "3. New customer handling: If customer_id='NEW' or similar placeholder, only the CREDIT check may be bypassed. "
        "Inventory must still be available for every line item.\n"
        "4. Retrieval sanity check: If retrieval_evidence is empty, or product/unit-price fields look unresolved for the whole order, "
        "treat the order as UNFULFILLABLE because the order could not be verified reliably.\n\n"
        "Return Decision JSON (according to given schema) with:\n"
        "- status: 'FULFILLABLE' if all checks pass, 'UNFULFILLABLE' otherwise\n"
        "- reason: Clear but brief explanation (e.g., 'Item X out of stock', 'Insufficient credit: needs €500, has €200', 'Retrieval failed so inventory/pricing could not be verified', 'All checks passed')\n"
        "- input_payload: Pass through the original RetrievedPO as is, for downstream agents"
    ),
    tools=[],
    default_options={"response_format": Decision},
)
