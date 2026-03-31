"""
Azure AI Search Index Management, Document Ingestion & Semantic Search
Handles schema creation, document upload, and hybrid search operations.
Data sourced from Airtable via airtable_tools module.
"""

# Add parent directory to path for module imports
import sys
import os
import re
import unicodedata
from loguru import logger

# Add parent to path so that crm.airtable_tools can be imported correctly?
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Airtable data fetchers
import json  # For JSON parsing
from difflib import SequenceMatcher
from functools import lru_cache  # cache function results to optimize performance
from typing import Any, Sequence  # Any: generic type, Sequence: list/tuple
from dotenv import load_dotenv

# Import Airtable data access functions
from crm.airtable_tools import get_all_products, get_all_customers

# Agent framework decorator, for AI function registration
from agent_framework import AgentExecutorResponse, WorkflowContext, executor, tool

# Azure SDK imports
from azure.identity import DefaultAzureCredential  # Managed identity auth

# This performs search operations (queries) against an index & manages documents.
# It is used for searching, uploading, merging, and deleting documents.
from azure.search.documents import SearchClient

# Client is used for creating & managing search indexes & their schemas,
# as well as managing other service-level resources like synonym maps.
from azure.search.documents.indexes import SearchIndexClient

# This model is used to define vector search queries within search requests.
from azure.search.documents.models import VectorizableTextQuery
from azure.search.documents.indexes.models import (
    SearchFieldDataType,  # Enum for search field data types
    SearchField,  # Defines individual fields in an index schema
    SearchIndex,  # SearchIndex is the schema definition for an index
    HnswAlgorithmConfiguration,  # HNSW vector search algorithm config 
    VectorSearch,  # Vector search config object holding algos/profiles/vectorizers
    VectorSearchProfile,  # Links vectorizer & algorithm for fields
    # SynonymMap,  # Not used currently, but cool!
    SemanticConfiguration,  # Semantic search config with field prioritization
    SemanticField,  # Individual field in semantic config
    SemanticPrioritizedFields,  # Groups fields by importance for semantic ranking
    SemanticSearch,  # Semantic search configuration object holding configs
    AzureOpenAIVectorizer,  # Azure OpenAI vectorizer for embeddings in vector search
    AzureOpenAIVectorizerParameters,  # Parameters for Azure OpenAI vectorizer
)


# ============================================================================
# ENVIRONMENT CONFIGURATION
# ============================================================================

load_dotenv()  # Load environment variables from .env file


def _get_env_var(name: str) -> str:
    """Return required environment variable or raise a value error early."""
    value = os.getenv(name)

    if value is None or value.strip() == "":
        raise ValueError(f"Environment var '{name}' must be set & non-empty")

    return value


# Azure Search settings
OPENAI_ENDPOINT = _get_env_var("AZURE_OPENAI_ENDPOINT")
OPENAI_EMBED_DEP_NAME = _get_env_var("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")
SERVICE_ENDPOINT = _get_env_var("AZURE_SEARCH_ENDPOINT")

# Initialize Azure clients with managed identity
CREDENTIAL = DefaultAzureCredential()
INDEX_CLIENT = SearchIndexClient(endpoint=SERVICE_ENDPOINT, credential=CREDENTIAL)

# Index configuration constants
INDEX_NAME_PRODUCTS = "paper-products-index"  # Products index name
INDEX_NAME_CUSTOMERS = "customer-insights-index"  # Customers index name
VECTOR_FIELD_NAME = "searchableContentVector"  # Field for vector embeddings
VECTOR_DIMENSIONS = 3072  # text-embedding-3-large dimensions
VECTOR_ALGO_NAME = "hnsw-algo"  # Algorithm identifier


# ============================================================================
# VECTOR SEARCH CONFIGURATION: HNSW + Azure OpenAI EMBEDDINGS
# ============================================================================

# Python decorator that tells the function below it to "remember" (to cache)
# its result the first time it's called. maxsize=1 means it only keeps the most
# recent result. This is useful for expensive setup functions, so they only
# run once and are fast on later calls.
@lru_cache(maxsize=1)  # Basically: Cache to avoid rebuilding config
def _build_vector_search() -> VectorSearch:
    """
    Builds vector search configuration with HNSW algorithm,
    Azure OpenAI vectorizer, and embedding profile for Azure AI Search.
    (using HNSW (Hierarchical Navigable Small World) for fast ANN search.)

    Returns: VectorSearch configuration object.
    1. HNSW Algorithm: Efficient similarity search.
    2. Azure OpenAI Vectorizer: Embedding model integration.
    3. Embedding Profile: Links vectorizer & algorithm for fields.
    """
    return VectorSearch(
        algorithms=[
            # Cosine similarity is used by default
            HnswAlgorithmConfiguration(name=VECTOR_ALGO_NAME)
        ],
        profiles=[  # Default PROFILE for embedding-based search
            # In Azure AI Search, a "profile" in the vector search configuration
            # (such as VectorSearchProfile) defines a named set of settings that
            # link together the vector search algorithm (like HNSW) and the
            # vectorizer (such as Azure OpenAI embeddings).
            VectorSearchProfile(
                name="embedding-profile",  # Profile name referenced in field defini.
                algorithm_configuration_name=VECTOR_ALGO_NAME,  # Links to algo above
                vectorizer_name="openai-vectorizer",  # Links to vectorizer below
            )
        ],
        vectorizers=[
            # Default VECTORIZER for embedding-based search
            AzureOpenAIVectorizer(
                vectorizer_name="openai-vectorizer",  # Vectorizer identifier
                parameters=AzureOpenAIVectorizerParameters(
                    resource_url=OPENAI_ENDPOINT,  # Azure OpenAI service endpoint
                    deployment_name=OPENAI_EMBED_DEP_NAME,  # Model deployment name
                    model_name="text-embedding-3-large",  # Embedding model
                ),
            ),
        ],
    )


# ============================================================================
# SEMANTIC SEARCH CONFIGURATION: Define semantic search settings with
# field prioritization to improve relevance.
# ============================================================================


def _build_semantic_search(
    config_name: str,  # Unique config name for this index
    title_field: str,  # Primary field for semantic title
    content_fields: list[str],  # Fields with main content
    keywords_fields: list[str],  # Fields with metadata/tags
) -> SemanticSearch:
    """
    Builds semantic search configuration with field prioritization.
    Semantic ranking improves relevance using AI understanding.

    Returns: SemanticSearch configuration object.
    1. Title field: Most important for relevance.
    2. Content fields: Main body text.
    3. Keywords fields: Tags/metadata for filtering.
    and also the config name: Unique identifier for this semantic config.
    """
    return SemanticSearch(
        configurations=[
            SemanticConfiguration(
                name=config_name,  # Referenced in search queries
                prioritized_fields=SemanticPrioritizedFields(
                    title_field=SemanticField(
                        field_name=title_field
                    ),  # Most important field
                    content_fields=[
                        SemanticField(field_name=f) for f in content_fields
                    ],  # Body content
                    keywords_fields=[
                        SemanticField(field_name=f) for f in keywords_fields
                    ],  # Tags/filters
                ),
            )
        ]
    )


# ============================================================================
# FIELD DEFINITIONS
# ============================================================================


def _product_fields() -> list[SearchField]:
    """
    Defines schema for paper products index.
    Maps Airtable Products table fields to Azure AI Search schema.
    """
    return [
        SearchField(
            name="sku",
            type=SearchFieldDataType.String,
            key=True,
            filterable=True,
        ),  # Unique identifier
        SearchField(
            name="title",
            type=SearchFieldDataType.String,
            searchable=True,
        ),  # Product name
        SearchField(
            name="description",
            type=SearchFieldDataType.String,
            searchable=True,
        ),  # Product details
        # Product attributes from Airtable
        SearchField(
            name="size",
            type=SearchFieldDataType.String,
            searchable=True,
            filterable=True,
            facetable=True,
        ),  # A4, A5
        SearchField(
            name="gsm",
            type=SearchFieldDataType.Int32,
            filterable=True,
            facetable=True,
        ),  # Paper weight
        SearchField(
            name="finish",
            type=SearchFieldDataType.String,
            # Glossy, matte, etc.
            searchable=True,
            filterable=True,
            facetable=True,
        ),
        SearchField(
            name="color",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
        ),  # Paper color
        SearchField(
            name="uom",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
        ),  # Unit of Measure
        SearchField(
            name="unitPrice",
            type=SearchFieldDataType.Double,
            filterable=True,
            sortable=True,
        ),  # Price per unit
        SearchField(
            name="qtyAvailable",
            type=SearchFieldDataType.Int32,
            filterable=True,
            sortable=True,
        ),  # Stock quantity
        SearchField(
            name="active",
            type=SearchFieldDataType.Boolean,
            filterable=True,
        ),  # Active status of product
        SearchField(
            name="searchableContentText",
            type=SearchFieldDataType.String,
            searchable=True,
        ),  # Combined text for full-text search
        SearchField(
            name=VECTOR_FIELD_NAME,  # Vector embeddings field
            type=SearchFieldDataType.Collection(
                SearchFieldDataType.Single
            ),  # Array of floats
            searchable=True,  # Enable vector search
            vector_search_dimensions=VECTOR_DIMENSIONS,  # 3072 for embedd large
            vector_search_profile_name="embedding-profile",  # Link to vector config
        ),
    ]


def _customer_fields() -> list[SearchField]:
    """
    Defines schema for customer insights index.
    Maps Airtable Customers table fields to Azure AI Search schema.
    """
    return [
        SearchField(
            name="customerId",
            type=SearchFieldDataType.String,
            key=True,
            filterable=True,
        ),  # Unique identifier
        SearchField(
            name="companyName",
            type=SearchFieldDataType.String,
            searchable=True,
            filterable=True,
            facetable=True,
        ),
        SearchField(
            name="email",
            type=SearchFieldDataType.String,
            searchable=True,
            filterable=True,
            sortable=True,
        ),
        SearchField(
            name="segment",
            type=SearchFieldDataType.String,
            searchable=True,
            filterable=True,
            facetable=True,
        ),
        SearchField(
            name="creditLimit",
            type=SearchFieldDataType.Double,
            filterable=True,
            sortable=True,
        ),
        SearchField(
            name="openAR",
            type=SearchFieldDataType.Double,
            filterable=True,
            sortable=True,
        ),  # Accounts receivable
        SearchField(
            name="billingAddress",
            type=SearchFieldDataType.String,
            searchable=True,
            filterable=True,
        ),  # Billing address
        SearchField(
            name="shippingAddress",
            type=SearchFieldDataType.String,
            searchable=True,
            filterable=True,
        ),  # Shipping address
        SearchField(
            name="status",
            type=SearchFieldDataType.String,
            filterable=True,
            facetable=True,
        ),  # Active/Inactive
        SearchField(
            name="searchableContentText",
            type=SearchFieldDataType.String,
            searchable=True,
        ),  # Combined text for full-text search, including all relevant fields
        SearchField(
            name=VECTOR_FIELD_NAME,  # Vector embeddings field
            type=SearchFieldDataType.Collection(
                SearchFieldDataType.Single
            ),  # Array of floats
            searchable=True,  # Enable vector search
            vector_search_dimensions=VECTOR_DIMENSIONS,  # 3072 for embedd large
            vector_search_profile_name="embedding-profile",  # Link to vector config
        ),
    ]


# ============================================================================
# INDEX SCHEMA CREATION
# ============================================================================

@tool
def create_products_index_schema() -> dict[str, Any]:
    """
    Creates or updates the products search index schema.
    Azure AI Search will update existing index if already present.
    """
    index = SearchIndex(
        name=INDEX_NAME_PRODUCTS,  # Index identifier
        fields=_product_fields(),  # Field definitions
        vector_search=_build_vector_search(),  # Vector search config
        semantic_search=_build_semantic_search(  # Semantic ranking config
            config_name="products-semantic-config",  # Config identifier
            title_field="title",  # Primary field
            content_fields=["searchableContentText"],  # Body (markdown, all content)
            keywords_fields=["sku", "size", "finish", "color"],  # Metadata
        ),
    )
    INDEX_CLIENT.create_or_update_index(index)  # Upsert index

    logger.info(
        "[FUNCTION create_products_index_schema] ✓ Products index '{}' created/updated successfully in Azure AI Search",
        INDEX_NAME_PRODUCTS)

    return {"status": "created_or_updated",
            "index": INDEX_NAME_PRODUCTS,
            "index_fields_count": len(_product_fields())}



@tool
def create_customer_index_schema() -> dict[str, Any]:
    """
    Creates or updates the customers search index schema.
    Azure AI Search will update existing index if already present.
    """
    index = SearchIndex(
        name=INDEX_NAME_CUSTOMERS,  # Index identifier
        fields=_customer_fields(),  # Field definitions
        vector_search=_build_vector_search(),  # Vector search config
        semantic_search=_build_semantic_search(  # Semantic ranking config
            config_name="customers-semantic-config",  # Config identifier
            title_field="companyName",  # Primary field
            content_fields=["searchableContentText"],  # Body (markdown, all content)
            keywords_fields=["customerId", "segment", "status"],
        ),
    )
    INDEX_CLIENT.create_or_update_index(index)  # Upsert index
    
    logger.info(
        "[FUNCTION create_customer_index_schema] ✓ Customer index '{}' created/updated successfully in Azure AI Search",
        INDEX_NAME_CUSTOMERS)

    return {"status": "created_or_updated",
            "index": INDEX_NAME_CUSTOMERS,
            "index_fields_count": len(_customer_fields())}

# ============================================================================
# DOCUMENT INGESTION
# ============================================================================


def _upload_documents_to_index(
    index_name: str,
    documents: list[dict[str, Any]]
) -> None:
    """
    Uploads documents to specified Azure AI Search index.
    Uses batch upload for efficiency.

    Args:
        index_name: Name of the target index
        documents: list of document dictionaries to upload
    """
    search_client = SearchClient(
        endpoint=SERVICE_ENDPOINT,  # AI Search endpoint
        index_name=index_name,  # Target index
        credential=CREDENTIAL,  # Managed identity
    )

    search_client.upload_documents(documents=documents)  # Batch upload


def _normalize_text(value: Any) -> str:
    """Normalize free text for lightweight fallback matching."""
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def _score_match(query: str, *values: Any) -> float:
    """Return a simple overlap/fuzzy score for local fallback search."""
    normalized_query = _normalize_text(query)
    if not normalized_query:
        return 0.0

    combined = " ".join(_normalize_text(value) for value in values if value)
    if not combined:
        return 0.0

    query_tokens = set(normalized_query.split())
    combined_tokens = set(combined.split())
    overlap_score = len(query_tokens & combined_tokens) * 3.0
    substring_bonus = 8.0 if normalized_query in combined else 0.0
    fuzzy_score = SequenceMatcher(None, normalized_query, combined).ratio() * 5.0
    return overlap_score + substring_bonus + fuzzy_score


def _search_customers_from_airtable(query: str, top: int = 3) -> list[dict[str, Any]]:
    """Fallback customer lookup when Azure Search data-plane access is unavailable."""
    ranked: list[tuple[float, dict[str, Any]]] = []
    minimum_score = 3.0

    for record in get_all_customers():
        fields = record.get("fields", {})
        result = {
            "customerId": fields.get("Customer ID", ""),
            "companyName": fields.get("Name", ""),
            "email": fields.get("Email", ""),
            "creditLimit": float(fields.get("Credit Limit", 0) or 0),
            "openAR": float(fields.get("Open AR", 0) or 0),
            "status": fields.get("Status", ""),
            "billingAddress": fields.get("Billing Address", ""),
            "shippingAddress": fields.get("Shipping Address", ""),
        }
        score = _score_match(
            query,
            result["customerId"],
            result["companyName"],
            result["email"],
            result["billingAddress"],
            result["shippingAddress"],
        )
        ranked.append((score, result))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [result for score, result in ranked if score >= minimum_score][:top]


def _search_products_from_airtable(query: str, top: int = 5) -> list[dict[str, Any]]:
    """Fallback product lookup when Azure Search data-plane access is unavailable."""
    ranked: list[tuple[float, dict[str, Any]]] = []
    minimum_score = 3.0

    for record in get_all_products():
        fields = record.get("fields", {})
        attrs = json.loads(fields.get("Attributes JSON", "{}") or "{}")
        result = {
            "sku": fields.get("SKU", ""),
            "title": fields.get("Title", ""),
            "description": fields.get("Description", ""),
            "size": attrs.get("size"),
            "gsm": int(attrs.get("gsm", 0) or 0),
            "finish": attrs.get("finish"),
            "color": attrs.get("color"),
            "uom": fields.get("UOM"),
            "unitPrice": float(fields.get("Unit Price", 0) or 0),
            "qtyAvailable": int(fields.get("Qty Available", 0) or 0),
            "active": bool(fields.get("Active", False)),
        }
        if not result["active"]:
            continue

        score = _score_match(
            query,
            result["sku"],
            result["title"],
            result["description"],
            result["size"],
            result["gsm"],
            result["finish"],
            result["color"],
        )
        ranked.append((score, result))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [result for score, result in ranked if score >= minimum_score][:top]


@tool
def ingest_products_from_airtable() -> dict[str, Any]:
    """
    Fetches product data from Airtable and uploads to products index.
    Transforms Airtable records into Azure AI Search document format.
    """
    # Ensure the index exists before attempting uploads (idempotent upsert).
    # create_products_index_schema()

    records = get_all_products()  # Get all products from Airtable via airtable_tools
    documents = []  # Document accumulator

    for record in records:
        fields = record["fields"]  # Airtable fields object
        
        # Parse attributes from the JSON string in "Attributes JSON" field of
        # the Airtable Products table. If the field is missing or empty,
        # default to an empty dictionary.
        attrs = json.loads(fields.get("Attributes JSON", "{}"))

        # Handle finish synonyms for better search matching e.g. coated matte = matt matte        
        finish = attrs.get('finish', '') or ''
        finish_synonyms = {
            "coated matte": "matt matte",
            "copy": "kopierpapier",
            "uncoated": "ungestrichen uncoated",
        }
        extra_keywords = finish_synonyms.get(finish.lower(), "")
        
        # Construct searchable content field for vector search in markdown format:
        # This field is used for vector embeddings and full-text search
        searchable_content = (
            f"# Product Profile\n"
            f"**Title:** {fields.get('Title', '')}\n"
            f"**Description:** {fields.get('Description', '')}\n"
            f"**Size:** {attrs.get('size', '')}\n"
            f"**Weight:** {attrs.get('gsm', '')} gsm\n"
            f"**Finish:** {finish}\n"
            f"**Color:** {attrs.get('color', '')}\n"
            f"**Keywords:** {extra_keywords}\n"
        )

        # Construct document dictionary for AI Search
        doc = {
            "sku": fields.get("SKU"),  # Unique identifier
            "title": fields.get("Title"),  # Product name
            "description": fields.get("Description"),  # Product details
            "searchableContentText": searchable_content,  # Combined text
            "size": attrs.get("size"),  # Paper size (A4, A5, etc.)
            "gsm": int(attrs.get("gsm", 0)),  # Paper weight
            "finish": attrs.get("finish"),  # Surface finish
            "color": attrs.get("color"),  # Paper color
            "uom": fields.get("UOM"),  # Unit of measure
            "unitPrice": float(fields.get("Unit Price", 0)),  # Price
            "qtyAvailable": int(fields.get("Qty Available", 0)),  # Stock
            "active": fields.get("Active", False),  # Active status
        }

        documents.append(doc)

    try:
        _upload_documents_to_index(INDEX_NAME_PRODUCTS, documents)  # Batch upload
    except Exception as exc:
        logger.warning(
            "[FUNCTION ingest_products_from_airtable] Azure Search upload failed; "
            "continuing with Airtable-only fallback | error={}",
            exc,
        )
        return {
            "status": "fallback_only",
            "index": INDEX_NAME_PRODUCTS,
            "ingested_docs_count": len(documents),
            "error": str(exc),
        }

    logger.info(
        "[FUNCTION ingest_products_from_airtable] ✓ Ingested {} documents from "
        "Airtable into products index '{}' to Azure AI Search",
        len(documents),
        INDEX_NAME_PRODUCTS
    )

    return {"status": "ingested",
            "index": INDEX_NAME_PRODUCTS,
            "ingested_docs_count": len(documents)}


@tool
def ingest_customers_from_airtable() -> dict[str, Any]:
    """
    Fetches customer data from Airtable and uploads to customers index.
    Derives segment from credit limit, and addresses.
    """
    # Ensure the index exists before attempting uploads (idempotent upsert).
    # create_customer_index_schema()

    records = get_all_customers()  # Get all customers via airtable_tools
    documents = []  # Document accumulator

    # This section does not use `attrs` because Customers records do not 
    # have a similar "Attributes JSON" field like Products do.
    for record in records:
        fields = record["fields"]  # Airtable fields object

        # Construct searchable content field for vector search:
        searchable_content = (
            f"# Customer Profile\n"
            f"**Company Name:** {fields.get('Name', '')}\n"
            f"**Email:** {fields.get('Email', '')}\n"
            f"**Full Shipping Address with German city and postal code:** {fields.get('Shipping Address', '')}\n"
            f"**Full Billing Address with German city and postal code:** {fields.get('Billing Address', '')}\n"
            f"**Customer ID:** {fields.get('Customer ID', '')}\n"
        )

        doc = {
            "customerId": fields.get("Customer ID"),  # Unique identifier
            "companyName": fields.get("Name"),  # Company name
            "email": fields.get("Email"),  # Email address
            "billingAddress": fields.get("Billing Address", ""),  # Billing address
            "shippingAddress": fields.get("Shipping Address", ""),  # Shipping address
            "searchableContentText": searchable_content,  # Combined text
            "creditLimit": fields.get("Credit Limit", 0),  # Credit limit
            "openAR": float(fields.get("Open AR", 0)),  # Accounts receivable
            "status": fields.get("Status"),  # Active/Inactive
        }

        documents.append(doc)

    try:
        _upload_documents_to_index(INDEX_NAME_CUSTOMERS, documents)  # Batch upload
    except Exception as exc:
        logger.warning(
            "[FUNCTION ingest_customers_from_airtable] Azure Search upload failed; "
            "continuing with Airtable-only fallback | error={}",
            exc,
        )
        return {
            "status": "fallback_only",
            "index": INDEX_NAME_CUSTOMERS,
            "ingested_docs_count": len(documents),
            "error": str(exc),
        }

    logger.info(
        "[FUNCTION ingest_customers_from_airtable] ✓ Ingested {} documents from "
        "Airtable into customers index '{}' to Azure AI Search",
        len(documents),
        INDEX_NAME_CUSTOMERS
    )

    return {"status": "ingested",
            "index": INDEX_NAME_CUSTOMERS,
            "ingested_docs_count": len(documents)}


# ============================================================================
# SEARCH OPERATIONS (for Retriever Agent)
# ============================================================================


def _semantic_and_hybrid_search(
    index_name: str,
    query_text: str,
    top: int = 3,
    *,  # Force keyword-only arguments after this point (for clarity)
    select: Sequence[str] | None = None,  # Fields to return. Sequence is list/tuple
    filter: str | None = None,  # OData filter expression
    semantic_config: str | None = None,  # Semantic configuration name
    vector_field: str = VECTOR_FIELD_NAME,  # Vector field name
) -> list[dict[str, Any]]:
    """Perform hybrid (keyword + vector) search with optional semantic ranking.

    1. Keyword/BM25 search via the `search_text=query_text` payload: Azure Search
       runs its classic lexical ranking on the text you supply.
    2. Vector similarity search via `vector_queries=[VectorizableTextQuery(...)]`,
       which turns the same text into an embedding and finds the k-nearest
       neighbors in the configured vector field.
    3. Semantic reranking layered on top by setting `query_type="semantic"` and
       providing a semantic configuration, so Azure's semantic ranker reorders
       the blended lexical/vector results.
    
    Returns:
           list[dict[str, Any]]: A list of search result documents.
    """

    search_client = SearchClient(
        endpoint=SERVICE_ENDPOINT,
        index_name=index_name,
        credential=CREDENTIAL
    )

    vector_query = VectorizableTextQuery(
        text=query_text,
        k=top,
        fields=vector_field,
    )

    # Build search parameters, only including optional arguments when provided
    search_kwargs: dict[str, Any] = {
        "search_text": query_text,  # Keyword search text
        "vector_queries": [vector_query],  # Vector search query
        "query_type": "semantic",  # Enable semantic ranking
        "semantic_configuration_name": semantic_config,  # Use provided config
        "top": top,  # Number of results to return
    }

    if select is not None:  # Only include specified fields if provided
        search_kwargs["select"] = list(select)
    else:  # If no fields specified, select all
        search_kwargs["select"] = ["*"]

    if filter is not None:  # Only include filter if given which narrows results
        search_kwargs["filter"] = filter

    results = search_client.search(**search_kwargs)

    # Convert results to list of dictionaries, containing only the document fields
    return [dict(result) for result in results]


def _search_customers(
    query: str,
    top: int = 3,
) -> list[dict[str, Any]]:
    """Searches customers by name, address, and other details given as a query.
    It uses semantic and hybrid search to find relevant customer records from
    the index, which may include various customer attributes.

    Args:
        query (str): The search query string, including customer details.
        top (int, optional): Number of top results to return. Defaults to 5.
    Returns:
        list[dict[str, Any]]: A list of CUSTOMER search result documents.
    """
    logger.info(
            "[FUNCTION search_customers] Performing hybrid and semantic "
            "search on customers index '{}' in Azure AI Search...",
            INDEX_NAME_CUSTOMERS
    )

    try:
        return _semantic_and_hybrid_search(
            INDEX_NAME_CUSTOMERS,
            query,
            top=top,
            select=[
                "customerId",
                "companyName",
                "email",
                "creditLimit",
                "openAR",
                "status",
                "billingAddress",
                "shippingAddress",
            ],
            semantic_config="customers-semantic-config",
        )
    except Exception as exc:
        logger.warning(
            "[FUNCTION search_customers] Azure Search lookup failed; "
            "falling back to Airtable matching | error={}",
            exc,
        )
        return _search_customers_from_airtable(query, top=top)
    

def _search_products(
    query: str,
    top: int = 5,
) -> list[dict[str, Any]]:
    """Searches the indexed Products by description/specs given as a query.
    It uses hybrid search (keyword + vector) with semantic ranking.
    The results include product details like SKU, title, description,
    size, gsm, finish, color, unit price, availability, etc. 

    Args:
        query (str): The search query string, including product details.
        top (int, optional): Number of top results to return. Defaults to 5.
    Returns:
        list[dict[str, Any]]: A list of PRODUCT search result documents.
    """

    logger.info(
        "[FUNCTION search_products] Performing hybrid and semantic "
        "search on products index '{}' in Azure AI Search...",
        INDEX_NAME_PRODUCTS,
    )

    try:
        return _semantic_and_hybrid_search(
            INDEX_NAME_PRODUCTS,
            query,
            top=top,
            filter="active eq true",
            select=[
                "sku",
                "title",
                "description",
                "size",
                "gsm",
                "finish",
                "color",
                "uom",
                "unitPrice",
                "qtyAvailable",
                "active",
            ],
            semantic_config="products-semantic-config",
        )
    except Exception as exc:
        logger.warning(
            "[FUNCTION search_products] Azure Search lookup failed; "
            "falling back to Airtable matching | error={}",
            exc,
        )
        return _search_products_from_airtable(query, top=top)


# IMPORTANT: Register search functions as tools for agent use!
# We couldn't decorate the internal functions directly because
# they have parameters that was giving issues with type checking.
# So we wrap them in these tool-decorated functions.
search_customers = tool(_search_customers)
search_products = tool(_search_products)


# Executor func to delete both indexes from Azure AI Search at the end of workflow
@executor
@logger.catch
def destroy_indexes(
    upstream_agent_response: AgentExecutorResponse,
    ctx: WorkflowContext[AgentExecutorResponse],
) -> AgentExecutorResponse:
    """Deletes both products and customers indexes from Azure AI Search."""
    _ = upstream_agent_response.agent_response.value  # Unused parameter

    deleted_indexes: list[str] = []
    for index_name in (INDEX_NAME_PRODUCTS, INDEX_NAME_CUSTOMERS):
        try:
            INDEX_CLIENT.delete_index(index_name)
            deleted_indexes.append(index_name)
        except Exception as exc:
            logger.warning(
                "[FUNCTION destroy_indexes] Skipping delete for index '{}' | error={}",
                index_name,
                exc,
            )

    if deleted_indexes:
        logger.info(
            "[FUNCTION destroy_indexes] ✓ Deleted indexes from Azure AI Search: {}",
            ", ".join(deleted_indexes),
        )

    __ = ctx  # Unused parameter
    return upstream_agent_response


# ============================================================================
# MAIN EXECUTION: Local testing
# ============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 40 + "\n CREATING INDEX SCHEMAS" + "\n" + "=" * 40 + "\n")

    create_products_index_schema()
    create_customer_index_schema()

    print("\n" + "=" * 40 + "\n INGESTING DOCS FROM AIRTABLE" + "\n" + "=" * 40 + "\n")
      
    ingest_products_from_airtable()
    ingest_customers_from_airtable()

    print("\n" + "=" * 40 + "\n EXAMPLE SEARCHES:" + "\n" + "=" * 40 + "\n")

    # Example searches to demonstrate functionality
    print("\nProducts:\n")

    for result in _search_products("A4 coated gloss 200gsm", top=2):
        print(json.dumps(result, indent=4))

    print("\nCustomers:\n")

    for result in _search_customers("companies whose billing will be sent to Munich", top=2):
        print(json.dumps(result, indent=4))

    print("\n" + "=" * 40 + "\n COMPLETED" + "\n" + "=" * 40 + "\n")


    ########## Delete Indexes from Azure AI Search ############
    # result = INDEX_CLIENT.delete_index(INDEX_NAME_PRODUCTS)
    # print(result)
    # result = INDEX_CLIENT.delete_index(INDEX_NAME_CUSTOMERS)
    # print(result)
