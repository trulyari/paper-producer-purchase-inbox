"""Generate invoice PDFs and upload them to Azure Blob Storage."""

import os
import time
from pathlib import Path
from typing import Any
from datetime import datetime
from loguru import logger

from dotenv import load_dotenv
from jinja2 import Environment, FileSystemLoader, select_autoescape
from weasyprint import HTML

from agent_framework import tool
from azure.core.exceptions import ResourceExistsError
from azure.identity import AzureCliCredential, DefaultAzureCredential
from azure.storage.blob import BlobServiceClient, ContentSettings

load_dotenv()


def transform_retrieved_po_to_invoice_context(retrieved_po: dict) -> dict:
    """Transform RetrievedPO schema to invoice template format.
    
    Maps the agent workflow schema (RetrievedPO) to the structure expected
    by invoice_template.html Jinja2 template.
    
    IMPORTANT: Uses the ACTUAL field names the agent outputs:
    - product_name, ordered_qty, unit_price, subtotal (NOT the schema-defined names)
    """
    items_list = retrieved_po.get("items", [])
      
    # Extract customer info from first item or use retrieved_po level fields
    first_item = items_list[0] if items_list else {}
    
    shipping_address = (
        retrieved_po.get("customer_shipping_address")
        or first_item.get("customer_address")
        or first_item.get("matched_customer_address")
        or "N/A"
    )
    billing_address = retrieved_po.get("customer_billing_address") or shipping_address

    return {
        "customer": {
            "company": retrieved_po.get("customer_name", "N/A"),
            "contact": retrieved_po.get("customer_name", "N/A"),
            "email": retrieved_po.get("customer_email", "customer@example.com"),
            "address1": shipping_address,
            "address2": billing_address if billing_address != shipping_address else None,
        },
        "payment": {
            "terms": "Net 30",
            "method": "Bank transfer",
            "po_number": retrieved_po.get("po_number", retrieved_po.get("email_id", "N/A")),
        },
        "items": [
            {
                "description": item.get("product_name", "Unknown Product"),
                "qty": item.get("ordered_qty", 0),
                "unit": "pcs",
                "unit_price": item.get("unit_price", 0.0),
                "line_total": item.get("subtotal", 0.0),
            }
            for item in items_list
        ],
        "totals": {
            "subtotal": retrieved_po.get("subtotal", 0.0),
            "tax": retrieved_po.get("tax", 0.0),
            "shipping": retrieved_po.get("shipping", 0.0),
            "total": retrieved_po.get("order_total", 0.0),
        },
    }


def _render_invoice_html(template_path: Path, order_context: dict) -> str:
    """Render the invoice HTML from a Jinja2 template and order context."""

    env = Environment(
        loader=FileSystemLoader(str(template_path.parent)),
        autoescape=select_autoescape(enabled_extensions=("html", "xml")),
    )
    return env.get_template(template_path.name).render(**order_context)


def _html_to_pdf_bytes(html_content: str, base_path: Path) -> bytes:
    pdf_content = HTML(string=html_content, base_url=base_path.as_uri()).write_pdf()
    if not pdf_content:
        raise ValueError("Failed to generate PDF content from HTML.")
    return pdf_content


def _ensure_invoice_metadata(order_context: dict[str, Any]) -> dict[str, Any]:
    """Return a shallow copy of the order context with the required invoice fields present."""

    context = dict(order_context)
    invoice_block: dict[str, Any] = dict(context.get("invoice") or {})

    invoice_number = (
        invoice_block.get("number")
        or context.get("invoice_no")
        or context.get("invoice_number")
        or context.get("order_id")
        or context.get("po_number")
        or f"INV-{int(time.time())}"
    )

    # Default to today's date when issue/due dates are missing.
    today = datetime.utcnow().date().isoformat()
    invoice_block.setdefault("number", str(invoice_number))
    invoice_block.setdefault("issue_date", today)
    invoice_block.setdefault("due_date", today)

    context["invoice"] = invoice_block
    return context


def _looks_like_connection_string(value: str | None) -> bool:
    """Return True only for actual Azure Storage connection strings."""
    if not value:
        return False
    return (
        "DefaultEndpointsProtocol=" in value
        or "AccountKey=" in value
        or "SharedAccessSignature=" in value
    )


def _write_local_invoice_copy(pdf_content: bytes, invoice_number: str) -> Path:
    """Persist a local invoice file when Blob upload is unavailable."""
    output_dir = Path(__file__).resolve().parents[2] / "artifacts" / "invoices"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{invoice_number}-{int(time.time())}.pdf"
    output_path.write_bytes(pdf_content)
    return output_path


def _build_blob_service_client() -> BlobServiceClient:
    """Build a Blob client from either a real connection string or Azure AD auth."""
    account_url = os.getenv("AZURE_STORAGE_ACCOUNT_URL")
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

    if _looks_like_connection_string(connection_string):
        return BlobServiceClient.from_connection_string(connection_string)

    if not account_url:
        raise ValueError(
            "Set AZURE_STORAGE_ACCOUNT_URL for Azure AD auth, or provide a real "
            "AZURE_STORAGE_CONNECTION_STRING for local development."
        )

    try:
        return BlobServiceClient(
            account_url=account_url,
            credential=AzureCliCredential(),
        )
    except Exception:
        return BlobServiceClient(
            account_url=account_url,
            credential=DefaultAzureCredential(),
        )

@tool
def generate_invoice_pdf_url(
    order_context: dict,
    html_template: str | Path | None = None,
    
) -> str:
    """Generate an invoice PDF from an HTML template, upload it to Azure 
    Blob Storage, and return the URL.
    
    Args:
        order_context: Dictionary containing RetrievedPO invoice data.
        html_template: Optional path to (another) HTML template file.
    Returns:
        URL string to the uploaded invoice PDF.
    """
    logger.info("[FUNCTION generate_invoice_pdf_url] Generating invoice PDF...")

    html_template = html_template or (
        Path(__file__).resolve().parent / "invoice_template.html"
    ) #  .parent means the directory that directly contains this path.

    template_path = Path(html_template)

    if not template_path.exists():
        raise FileNotFoundError(f"Invoice HTML file not found: {template_path}")

    # Transform RetrievedPO schema to invoice template format
    transformed_context = transform_retrieved_po_to_invoice_context(order_context)
    order_context_with_invoice = _ensure_invoice_metadata(transformed_context)

    html_content = _render_invoice_html(template_path, order_context_with_invoice)
    pdf_content = _html_to_pdf_bytes(html_content, template_path.parent.resolve())

    logger.info("[FUNCTION generate_invoice_pdf_url] Uploading invoice PDF file to Azure Blob Storage...")

    container_name = os.getenv("AZURE_INVOICE_CONTAINER", "invoices")
    invoice_number = str(order_context_with_invoice["invoice"]["number"])

    blob_name = f"{template_path.stem}-{int(time.time())}.pdf"
    try:
        blob_service = _build_blob_service_client()
        container_client = blob_service.get_container_client(container_name)

        try:
            container_client.create_container()
        except ResourceExistsError:
            pass

        blob_client = container_client.get_blob_client(blob_name)
        blob_client.upload_blob(
            pdf_content,
            overwrite=True,
            content_settings=ContentSettings(content_type="application/pdf"),
        )

        logger.info("[FUNCTION generate_invoice_pdf_url] Invoice PDF uploaded successfully!")
        return blob_client.url
    except Exception as exc:
        local_path = _write_local_invoice_copy(pdf_content, invoice_number)
        logger.warning(
            "[FUNCTION generate_invoice_pdf_url] Blob upload unavailable; "
            "saved invoice locally instead | path={} | error={}",
            local_path,
            exc,
        )
        return str(local_path)


    

#################
# LOCAL TESTING # 
#################

if __name__ == "__main__":
    sample_template_path = (
        Path(__file__).parent.parent / "invoice" / "invoice_template.html"
    )
    sample_order_context = {
        "invoice": {
            "number": "INV-12345",
            "issue_date": "2025-10-26",
            "due_date": "2025-11-25",
        },
        "customer": {
            "company": "Acme Corp",
            "contact": "Jane Smith",
            "address1": "123 Main St",
            "address2": "New York City, NY 10001",
        },
        "payment": {
            "terms": "Net 30",
            "method": "Bank wire",
            "po_number": "PO-9876",
        },
        "items": [
            {
                "description": "Widget A",
                "qty": 10,
                "unit": "pcs",
                "unit_price": 2.5,
                "line_total": 25.0,
            },
            {
                "description": "Widget B",
                "qty": 5,
                "unit": "pcs",
                "unit_price": 5.0,
                "line_total": 25.0,
            },
        ],
        "totals": {
            "subtotal": 50.0,
            "tax": 9.5,
            "shipping": 15.0,
            "total": 74.5,
        },
    }

    url = generate_invoice_pdf_url(sample_order_context, sample_template_path)
    print(f"Uploaded PDF URL: {url}")
