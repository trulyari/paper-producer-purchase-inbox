from typing import Annotated
from pydantic import BaseModel, ConfigDict, Field, model_validator

from agent_framework import Agent

from agents.base import chat_client
from agents.middleware_tools import search_evidence
from aisearch.azure_search_tools import (
    create_products_index_schema,
    create_customer_index_schema,
    ingest_products_from_airtable,
    ingest_customers_from_airtable,
    search_customers,
    search_products,
)

class RetrievedItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customer_id: Annotated[
        str,
        Field(
            description="Resolved customer identifier from CRM (Customers) "
            "through similarity search")]
    customer_name: Annotated[
        str,
        Field(
            description="Resolved customer name from CRM (Customers) "
            "through similarity search")]
    customer_address: Annotated[
        str,
        Field(
            description="Resolved customer address from CRM (Customers) "
            "through similarity search")]
    product_sku: Annotated[
        str,
        Field(
            description="Matched product SKU identifier from catalog (Products) "
            "through similarity search")]
    product_name: Annotated[
        str,
        Field(
            description="Matched product name or title from catalog (Products) "
            "through similarity search")]
    product_qty_available: Annotated[
        int,
        Field(
            ge=0,
            description="Quantity available in stock for this SKU, from catalog "
            "retrieved from Products table through similarity search")]
    ordered_qty: Annotated[
        int,
        Field(
            gt=0,
            strict=True,
            description="Quantity ordered by customer for this line item "
            "based on PO email")]
    unit_price: Annotated[
        float,
        Field(
            ge=0,
            strict=True,
            description="Unit price in EUR")]
    vat_rate: Annotated[
        float,
        Field(
            default=0.19,
            description="VAT rate (default 19% for Germany)")]
    product_in_stock: Annotated[
        bool,
        Field(
            default=False,
            description="Whether the item is in stock and available: "
            "only `True` if: `product_qty_available >= ordered_qty` else: `False`")]
    subtotal: Annotated[
        float,
        Field(
            default=0.0,
            ge=0,
            strict=True,
            description="Line item subtotal for each SKU (ordered_qty * unit_price)")]

    # Compute derived fields after initialization, otherwise the retriever agent may
    # miss including them in the output, if we use computed_field decorators.
    @model_validator(mode="after")  # "after" means this runs after the model has been created
    def _set_computed_fields(self) -> "RetrievedItem":
        self.product_in_stock = self.product_qty_available >= self.ordered_qty
        self.subtotal = self.ordered_qty * self.unit_price
        return self


class RetrievedPO(BaseModel):
    model_config = ConfigDict(extra="forbid")
    email_id: Annotated[
        str,
        Field(
            description="Gmail message ID of the original purchase order email")]
    po_number: Annotated[
        str,
        Field(
            description="Purchase order number or reference from the original PO email")]
    customer_id: Annotated[
        str,
        Field(
            description="Customer identifier or account number")]
    customer_name: Annotated[
        str,
        Field(
            description="Customer's business or contact name")]
    customer_overall_credit_limit: Annotated[
        float,
        Field(
            description="Customer's overall credit limit")]
    customer_open_ar: Annotated[
        float,
        Field(
            description="Customer's current open accounts receivable amount "
            "(how much the customer currently owes)")]
    customer_available_credit: Annotated[
        float,
        Field(
            default=0.0,
            description="Customer's available credit to fulfill this order "
            "(customer_overall_credit_limit - customer_open_ar)")]
    items: Annotated[
        list[
            RetrievedItem
        ],
        Field(
            description="list of retrieved order line items (products) matched "
            "through similarity search")]
    tax: Annotated[
        float,
        Field(
            default=0.0,
            ge=0,
            strict=True,
            description="Calculated sales tax (sum of line subtotal * VAT rate)")]
    shipping: Annotated[
        float,
        Field(
            default=0.0,
            ge=0,
            strict=True,
            description="Flat shipping fee (€25 if subtotal > 0)",)]
    subtotal: Annotated[
        float,
        Field(
            default=0.0,
            ge=0,
            strict=True,
            description="Subtotal of all line items before tax and shipping")]
    order_total: Annotated[
        float,
        Field(
            default=0.0,
            ge=0,
            strict=True,
            description="Final total: subtotal + tax + shipping")]
    customer_can_order_with_credit: Annotated[
        bool,
        Field(
            description="Whether the customer can place orders using credit: "
            "Only 'True' if customer_available_credit >= order_total else 'False'",)]
    # Carry forward the evidence so the fact-checker can ground each field.
    retrieval_evidence: Annotated[
        list[str],
        Field(
            default_factory=list,
            description="Raw search documents (as JSON strings) that justify this RetrievedPO")]

    # Compute derived fields after initialization, otherwise the retriever agent may
    # miss including them in the output, if we use computed_field decorators.
    @model_validator(mode="after")  # "after" means this runs after the model has been created
    def _set_totals(self) -> "RetrievedPO":
        self.customer_available_credit = self.customer_overall_credit_limit - self.customer_open_ar
        self.tax = sum(item.subtotal * item.vat_rate for item in self.items)
        self.shipping = 25.0 if sum(item.subtotal for item in self.items) > 0 else 0.0
        self.subtotal = sum(item.subtotal for item in self.items)
        self.order_total = self.subtotal + self.tax + self.shipping
        self.customer_can_order_with_credit = self.customer_available_credit >= self.order_total
        # Auto-populate evidence from middleware capture without relying on LLM
        self.retrieval_evidence = list(search_evidence)
        return self


retriever = Agent(
    client=chat_client,
    name="retriever",
    instructions=(
        "You are an order enrichment specialist. Given a ParsedPO containing customer details and line items, "
        "your job is to refresh Azure AI Search data sources, then resolve the PO details. Follow this exact order:\n\n"
        "1. Call create_customer_index_schema() to create or update the customer index before doing anything else.\n"
        "2. Call create_products_index_schema() so the products index schema is also ready.\n"
        "3. Call ingest_customers_from_airtable() to load the latest customer table from Airtable CRM into Azure Search.\n"
        "4. Call ingest_products_from_airtable() to load the latest products table from Airtable.\n"
        "5. Only after the indexes are refreshed may you search: use search_customers() with the ParsedPO customer info, "
        "then search_products() for every ParsedPO line item. Compare at least the top few results with each other, and "
        "do NOT blindly take the first hit! Pick the SKU/title/finish that best matches the ParsedPO wording. "
        "Pay attention to the possible translations.\n\n"
        "While searching, pick the best matches and capture customerId, companyName, creditLimit, openAR, addresses, "
        "and for products capture sku, title, unitPrice, qtyAvailable. Build the RetrievedPO response object exactly as defined, "
        "carrying forward email_id and po_number untouched.\n\n"
        "Store the raw Azure Search documents you relied on as JSON strings inside retrieval_evidence for grounding.\n"
        "Do not calculate totals yourself—@computed_field logic handles it once the required fields are set.\n"
        "If no customer match exists, fall back to placeholder values (e.g., customerId='NEW', creditLimit=0, openAR=0) "
        "so downstream agents can react appropriately."
    ),
    tools=[
        create_customer_index_schema,    # ensure customers idx exists before search
        create_products_index_schema,    # ensure products idx exists before search
        ingest_customers_from_airtable,  # load customers data from Airtable
        ingest_products_from_airtable,   # load products data from Airtable
        search_customers,                # search tool for customer lookup
        search_products,                 # search tool for product lookup
    ],
    default_options={"response_format": RetrievedPO},
)
