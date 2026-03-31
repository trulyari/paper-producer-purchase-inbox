"""
Groundedness safety check using Azure AI Evaluation.

Validates that retriever agent responses are grounded in source documents
before allowing order fulfillment. Uses a pass-through pattern - accepts
AgentExecutorResponse, validates it, attaches metadata, and passes it through.
"""
import os
import json
from dotenv import load_dotenv
from loguru import logger

from azure.ai.evaluation import GroundednessEvaluator
from azure.identity import DefaultAzureCredential
from agent_framework import executor, AgentExecutorResponse, WorkflowContext

load_dotenv()


@executor(id="check_agent_groundedness")
async def check_agent_groundedness(
    retriever_response: AgentExecutorResponse,
    ctx: WorkflowContext[AgentExecutorResponse],
) -> None:
    """
    Validate that retriever output is grounded in source documents.
    
    Pass-through executor: accepts AgentExecutorResponse[RetrievedPO],
    validates groundedness, attaches pass/fail metadata, returns same response.
    
    Routing condition reads metadata to decide whether to continue to decider.
    """
    from agents.middleware_tools import search_queries
    
    retrieved_po = retriever_response.agent_response.value
    po_number = getattr(retrieved_po, 'po_number', 'UNKNOWN')
    
    logger.info(f"[Groundedness Check] Starting validation for PO: {po_number}")

    # If no response, pass through with failure metadata
    if not retrieved_po:
        logger.error("[Groundedness Check] No response from retriever!")
        _attach_failure_metadata(retriever_response, "No response value")
        await ctx.send_message(retriever_response)
        return
    
    # Get evidence documents
    retrieval_evidence = getattr(retrieved_po, 'retrieval_evidence', [])
    if not retrieval_evidence:
        reason = "No retrieval evidence captured for groundedness evaluation"
        logger.warning("[Groundedness Check] {} | po_number={}", reason, po_number)
        _attach_failure_metadata(retriever_response, reason)
        await ctx.send_message(retriever_response)
        return

    # Current evaluator expects plain strings, not message lists.
    response_text = json.dumps(retrieved_po.model_dump(), ensure_ascii=False)
    
    # Build query from captured search queries
    query_text = " | ".join(search_queries) if search_queries else f"PO {po_number} retrieval"
    # query = [{"role": "user", "content": query_text}]
    
    # Run evaluator
    evaluator = GroundednessEvaluator(
        model_config={
            "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
            "azure_deployment": "gpt-4.1",
        },
        threshold=3,
        credential=DefaultAzureCredential(),
    )
    
    context = "\n\n".join(retrieval_evidence)
    try:
        result = evaluator(
            query=query_text,
            response=response_text,
            context=context,
        )
    except Exception as exc:
        logger.exception(
            "[Groundedness Check] Evaluator call failed for PO {}",
            po_number,
        )
        _attach_failure_metadata(
            retriever_response,
            f"Groundedness evaluator failed: {exc}",
        )
        await ctx.send_message(retriever_response)
        return
    
    # Simple pass/fail check
    result_label = str(result.get("groundedness_result", "")).strip().lower()
    is_grounded = result_label == "pass"
    
    logger.info(f"[Groundedness Check] Result: {result_label} for PO {po_number}")
    
    # Attach metadata to response
    if is_grounded:
        _attach_success_metadata(retriever_response, result)
    else:
        reason = str(result.get("groundedness_reason", "Failed check"))
        _attach_failure_metadata(retriever_response, reason)
    
    await ctx.send_message(retriever_response)


def _attach_success_metadata(response: AgentExecutorResponse, result: dict) -> None:
    """Attach pass metadata to response."""
    if response.agent_response.additional_properties is None:
        response.agent_response.additional_properties = {}
    
    response.agent_response.additional_properties.update({
        "is_grounded_result": True,
        "groundedness_score": int(result.get("groundedness", 5)),
        "groundedness_reason": str(result.get("groundedness_reason", "Passed")),
    })


def _attach_failure_metadata(response: AgentExecutorResponse, reason: str) -> None:
    """Attach fail metadata to response."""
    if response.agent_response.additional_properties is None:
        response.agent_response.additional_properties = {}
    
    response.agent_response.additional_properties.update({
        "is_grounded_result": False,
        "groundedness_score": 0,
        "groundedness_reason": reason,
    })
