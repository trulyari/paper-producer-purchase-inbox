"""
Airtable CRM Tools - Data Sync and Write Operations
Handles Airtable API for: sync jobs, customer creation, direct lookups.
Resolver agent uses Azure AI Search for reads, not this module.
"""

import os
from datetime import datetime
import requests
from typing import Any
from dotenv import load_dotenv
from loguru import logger

from agent_framework import tool

# ============================================================================
# ENVIRONMENT CONFIGURATION
# ============================================================================

load_dotenv()

AIRTABLE_API_KEY = os.getenv("AIRTABLE_API_KEY")  # Personal access token
AIRTABLE_BASE_ID = os.getenv("AIRTABLE_BASE_ID")  # Base identifier
AIRTABLE_PRODUCTS_TABLE = os.getenv("AIRTABLE_PRODUCTS_TABLE", "Products")
AIRTABLE_CUSTOMERS_TABLE = os.getenv("AIRTABLE_CUSTOMERS_TABLE", "Customers")
AIRTABLE_API_URL = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}"
AIRTABLE_API_HEADER = {"Authorization": f"Bearer {AIRTABLE_API_KEY}"}


# ============================================================================
# AIRTABLE API HELPER FUNCTIONS
# ============================================================================


def _fetch_all_records(table_name: str) -> list[dict[str, Any]]:
    """
    Fetches all records from specified Airtable table.
    Handles pagination automatically for large datasets.

    Args:
        table_name: Name of the Airtable table to query
    Returns:
        list of record dictionaries with fields data
    """
    url = f"{AIRTABLE_API_URL}/{table_name}"  # Table endpoint
    all_records = []  # Accumulator for all pages
    offset = None  # Pagination cursor

    while True:  # Loop until all pages are fetched
        # Add offset if present to get next page
        params = {"offset": offset} if offset else {}

        response = requests.get(
            url,
            headers=AIRTABLE_API_HEADER,
            params=params
        )  # API call to FETCH records

        response.raise_for_status()  # Raise on 4xx/5xx errors

        data = response.json()  # Parse JSON response
        all_records.extend(data.get("records", []))  # Append current page

        offset = data.get("offset")  # Get next page cursor
        if not offset:  # No more pages
            break

    return all_records


def _create_record(
        table_name: str,
        fields: dict[str, Any]
) -> dict[str, Any]:
    """
    Creates a new record in specified Airtable table.
    
    Args:
        table_name: Name of the Airtable table to create record in
        fields: Dictionary of field names and values for the new record
    Returns:
        Created record with ID and fields
    """
    url = f"{AIRTABLE_API_URL}/{table_name}"  # Table endpoint
    payload = {"fields": fields}  # Wrap fields in Airtable format

    response = requests.post(
        url,
        headers=AIRTABLE_API_HEADER,
        json=payload
    )  # API call to CREATE record

    response.raise_for_status()  # Raise on 4xx/5xx errors

    return response.json()  # Return created record


def _update_record(
        table_name: str,
        record_id: str,
        fields: dict[str, Any]
) -> dict[str, Any]:
    """
    Updates an existing record in specified Airtable table.
    
    Args:
        table_name: Name of the Airtable table containing the record
        record_id: ID of the record to update 
        fields: Dictionary of field names and new values for the record
    Returns:
        Updated record with ID and fields
    """
    url = f"{AIRTABLE_API_URL}/{table_name}/{record_id}"  # Record endpoint
    payload = {"fields": fields}  # Wrap fields in Airtable format

    response = requests.patch(
        url,
        headers=AIRTABLE_API_HEADER,
        json=payload
    )  # API call to UPDATE record

    try:
        response.raise_for_status()  # Raise on 4xx/5xx errors
    except requests.exceptions.HTTPError:
        logger.error(
            "[FUNCTION _update_record] Airtable API error """ \
            "| table={} | record_id={} | fields={} | status={} | response={}",
            table_name, record_id, fields, response.status_code, response.text
        )
        raise

    return response.json()  # Return updated record

# ===================================================================
# DATA SYNC FUNCTIONS (for easy Azure AI Search ingestion): fetch all
# ===================================================================

def get_all_products() -> list[dict[str, Any]]:
    """Fetches all products for AI Search sync. Returns raw Airtable records."""
    logger.info("[FUNCTION get_all_products] Fetching all PRODUCTS from Airtable (for AI Search sync).")
    return _fetch_all_records(AIRTABLE_PRODUCTS_TABLE)


def get_all_customers() -> list[dict[str, Any]]:
    """Fetches all customers for AI Search sync. Returns raw Airtable records"""
    logger.info("[FUNCTION get_all_customers] Fetching all CUSTOMERS from Airtable (for AI Search sync).")
    return _fetch_all_records(AIRTABLE_CUSTOMERS_TABLE)


# ============================================================================
# WRITE OPERATIONS (create new records)
# ============================================================================

@tool
def add_new_customer(
        customer_name: str,
        customer_email: str,
        customer_address: str,
) -> dict[str, Any]:
    """Creates a new customer in Airtable when the agent can't find one.

    Returns a short status payload so the agent can log the outcome."""

    all_customers = _fetch_all_records(AIRTABLE_CUSTOMERS_TABLE)
    # columns: Customer ID, Name, Email, Billing Address, Shipping Address,
    #          Credit Limit, Open AR, Currency, Status

    max_id = 5000  # Start from C-5001

    for customer in all_customers:  # Find max existing Customer ID
        cust_id = customer["fields"].get("Customer ID", "C-5000")
        id_num = int(cust_id.split("-")[1])
        max_id = max(max_id, id_num)

    new_id = f"C-{max_id + 1}"  # New Customer ID

    fields = {  # Prepare fields for new record
        "Customer ID": new_id,
        "Name": customer_name,
        "Email": customer_email,
        "Billing Address": customer_address,
        "Credit Limit": 3000.0,  # Initial credit limit
        "Open AR": 0.0,
        "Currency": "EUR",
        "Status": "Active"
    }

    created = _create_record(AIRTABLE_CUSTOMERS_TABLE, fields)
    logger.info(
        "[FUNCTION add_new_customer] Creating NEW customer with ID '{}' and record ID '{}' in Airtable.",
        new_id,
        created.get("id")
    )
    return {
        "status": "created",
        "customer_id": new_id,
        "record_id": created.get("id"),
    }


@tool
def update_inventory(
        ordered_qty: int,
        product_sku: str,
) -> dict[str, Any]:
    """Knock units off inventory for a SKU and report the new quantity."""
    all_products = _fetch_all_records(AIRTABLE_PRODUCTS_TABLE)
    # columns: SKU, Title, Description, UOM, Unit Price, Qty Available,
    #          Active, Attributes JSON, Last Updated

    record_id: str | None = None
    new_inventory: int = 0

    for product in all_products:
        if product["fields"].get("SKU") != product_sku:
            continue

        record_id = product["id"]  # Capture the Airtable record ID
        current_qty = product["fields"].get("Qty Available", 0)
        new_inventory = current_qty - ordered_qty
        break

    if not record_id:
        raise ValueError(
            f"Product with SKU '{product_sku}' not found in Airtable"
        )

    fields = {
        "Qty Available": new_inventory,
        "Last Updated": datetime.now().isoformat(),
    }

    _update_record(AIRTABLE_PRODUCTS_TABLE, record_id, fields)

    logger.info(
        "[FUNCTION update_inventory] Updating inventory for SKU '{}' to new quantity: {} in Airtable...",
        product_sku,
        new_inventory
    )

    return {
        "status": "updated",
        "sku": product_sku,
        "qty_available": new_inventory,
    }


@tool
def update_customer_credit(
        customer_id: str,
        order_amount: float,
) -> dict[str, Any]:
    """Increase a customer's open AR and report back the remaining credit."""
    all_customers = _fetch_all_records(AIRTABLE_CUSTOMERS_TABLE)
    # columns: Customer ID, Name, Email, Billing Address, Shipping Address,
    #          Credit Limit, Open AR, Currency, Status

    updated_available_credit = 0.0
    record_id: str | None = None
    new_open_ar = 0.0  # To be calculated

    for customer in all_customers:
        if customer["fields"].get("Customer ID") != customer_id:
            continue

        record_id = customer["id"]
        current_open_ar = customer["fields"].get("Open AR", 0.0)
        credit_limit = customer["fields"].get("Credit Limit", 0.0)
        new_open_ar = current_open_ar + order_amount  # Update Open AR
        updated_available_credit = credit_limit - new_open_ar  # Calc available credit
        break

    if not record_id:
        raise ValueError(
            f"Customer with ID '{customer_id}' not found in Airtable"
        )

    fields = {
        "Open AR": new_open_ar,
    }

    logger.info(
        "[FUNCTION update_customer_credit] Updating Open AR for Customer ID "
        "'{}' from {} to {} (added {}) in Airtable...",
        customer_id, new_open_ar - order_amount, new_open_ar, order_amount
    )

    # First, update Open AR field in the record of the customer
    try:
        _update_record(AIRTABLE_CUSTOMERS_TABLE, record_id, fields)
        logger.info(
            "[FUNCTION update_customer_credit] Successfully updated Open AR "
            "for Customer ID '{}'", customer_id
        )
    except Exception as e:
        logger.error(
            "[FUNCTION update_customer_credit] Failed to update Airtable record "
            "| customer_id={} | error={}", customer_id,
            str(e)
        )
        raise
    
    # Report the new credit exposure back to the caller.
    return {
        "open_ar": float(new_open_ar),
        "available_credit": float(updated_available_credit),
    }
