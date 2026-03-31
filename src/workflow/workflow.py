import asyncio
import os
import sys
from pathlib import Path
from typing import Any

# Ensure the repository's src/ directory is importable when running as a script
# resolve() gets absolute path; parents[1] goes up two levels
PROJECT_SRC = Path(__file__).resolve().parents[1]  # this gets us to the src/ directory
sys.path.insert(0, str(PROJECT_SRC))  # add src to sys.path, index '0' means start of list

from agent_framework import (  # noqa: E402
    WorkflowBuilder,  # Builder class for constructing workflows (includes .add(), etc. methods)
    WorkflowContext,  # Controlled interface for executors to interact w workflow ecosystem
    AgentExecutorResponse,  # Response object returned by agent executors
    executor,  # Decorator converts standalone functions to FunctionExecutor instances
)

from agents import (  # noqa: E402
    classifier,
    parser,
    retriever,
    # fact_checker,
    decider,
    fulfiller,
    rejector,
)

from agents.middleware_tools import clear_evidence  # noqa: E402
from safety.groundedness_check import check_agent_groundedness  # noqa: E402
from aisearch.azure_search_tools import destroy_indexes # executor to delete indexes after use  # noqa: E402
from emailing.gmail_tools import (  # noqa: E402
    # below is NOT the AI function (get_unread_emails is, which is used by the agent)!
    fetch_unread_emails,
    mark_email_as_read
)

from loguru import logger  # noqa: E402


@logger.catch  # Decorator to catch & log exceptions in the function if any occur
def should_parse(agent_response: AgentExecutorResponse) -> bool:
    """Route to parser only if email is a PO: is_po=True"""
    logger.info(
        "[FUNCTION should_parse] Checking whether the email is a Purchase Order..."
    )
    
    return getattr(agent_response.agent_response.value,
                   'is_po', False)


@logger.catch
def should_consume_non_po(agent_response: AgentExecutorResponse) -> bool:
    """Route to terminal non-PO handler when classification is negative."""
    logger.info(
        "[FUNCTION should_consume_non_po] Checking whether the email is NOT a Purchase Order..."
    )

    return getattr(agent_response.agent_response.value, "is_po", None) is False


@logger.catch
def should_fulfill(agent_response: AgentExecutorResponse) -> bool:
    """Route to fulfiller if order is fulfillable: status='FULFILLABLE'"""
    logger.info(
        "[FUNCTION should_fulfill] Checking whether the order is FULFILLABLE..."
    )

    return getattr(agent_response.agent_response.value,
                   'status', None) == "FULFILLABLE"


@logger.catch
def should_reject(agent_response: AgentExecutorResponse) -> bool:
    """Route to rejector if order is unfulfillable: status='UNFULFILLABLE'"""
    logger.info(
        "[FUNCTION should_reject] Checking whether the order is UNFULFILLABLE..."
    )
        
    return getattr(agent_response.agent_response.value,
                   'status', None) == "UNFULFILLABLE"


@logger.catch
def should_be_grounded(groundedness_response: AgentExecutorResponse) -> bool:
    """Route to decider if groundedness check passes.
    If False, workflow terminates and logs the failure."""
    
    # Read groundedness metadata from additional_properties (not from .value!)
    # The check_agent_groundedness executor attaches metadata to AgentResponse.additional_properties
    additional_props = groundedness_response.agent_response.additional_properties or {}
    is_grounded = additional_props.get('is_grounded_result', False)
    
    # Log failure case before returning
    if not is_grounded:
        logger.warning(
            "Groundedness check FAILED | score={} | reason={}",
            additional_props.get('groundedness_score', 0),
            additional_props.get('groundedness_reason', 'No reason provided')
        )
    else:
        logger.info("Groundedness check PASSED.")
    
    return is_grounded


@logger.catch
def _mark_terminal_email_read(message_id: str | None, terminal_action: str) -> bool:
    """Best-effort Gmail read marking for terminal workflow branches."""
    if not message_id:
        logger.warning(
            "Terminal branch missing email_id; cannot mark read | terminal_action={}",
            terminal_action,
        )
        return False

    try:
        mark_result = mark_email_as_read(message_id)
    except Exception:
        logger.exception(
            "Failed to mark email as read | email_id={} | terminal_action={}",
            message_id,
            terminal_action,
        )
        return False

    logger.info(
        "Email marked read | email_id={} | terminal_action={}",
        mark_result["id"],
        terminal_action,
    )
    return True


def _get_terminal_output(workflow_result: Any, email_id: str) -> dict[str, Any] | None:
    """Extract the latest terminal output for the current email, if one exists."""
    get_outputs = getattr(workflow_result, "get_outputs", None)
    if not callable(get_outputs):
        return None

    for output in reversed(get_outputs()):
        if isinstance(output, dict) and output.get("email_id") == email_id:
            return output

    return None


@executor
@logger.catch
async def consume_non_po_email(
    classifier_response: AgentExecutorResponse,
    ctx: WorkflowContext[AgentExecutorResponse],
) -> None:
    """Terminal executor for non-PO emails."""
    classified = classifier_response.agent_response.value
    email = getattr(classified, "email", None)
    email_id = getattr(email, "id", None)
    reason = getattr(classified, "reason", "")
    marked_read = _mark_terminal_email_read(email_id, "ignored_non_po")

    logger.info(
        "Non-PO email completed | email_id={} | marked_read={} | reason={}",
        email_id,
        marked_read,
        reason or "[no reason]",
    )

    await ctx.yield_output(
        {
            "email_id": email_id or "",
            "terminal_action": "ignored_non_po",
            "should_mark_read": True,
            "marked_read": marked_read,
        }
    )


@executor  # Decorator to make this function an executor in the workflow graph
@logger.catch
async def log_fulfillment(
    fulfillment_response: AgentExecutorResponse,
    ctx: WorkflowContext[AgentExecutorResponse],
) -> None:
    """Terminal logger for successful fulfillment runs."""
    fulfillment_result = fulfillment_response.agent_response.value
    ok = bool(getattr(fulfillment_result, "ok", False))
    email_id = getattr(fulfillment_result, "email_id", None)
    order_id = getattr(fulfillment_result, "order_id", "")
    marked_read = _mark_terminal_email_read(email_id, "fulfilled") if ok else False

    logger.info(
        "Fulfillment branch completed | ok={} | order_id={} | email_id={} | marked_read={}",
        ok,
        order_id,
        email_id,
        marked_read,
    )

    await ctx.yield_output(
        {
            "email_id": email_id or "",
            "terminal_action": "fulfilled" if ok else "fulfillment_incomplete",
            "should_mark_read": ok,
            "marked_read": marked_read,
            "order_id": order_id,
        }
    )

    # Forward the same response so downstream cleanup executes regardless of branch
    await ctx.send_message(fulfillment_response)


@executor
@logger.catch
async def log_rejection(
    rejector_response: AgentExecutorResponse,
    ctx: WorkflowContext[AgentExecutorResponse],
) -> None:
    """Terminal logger for rejection runs."""
    rejector_result = rejector_response.agent_response.value
    notified = bool(getattr(rejector_result, "rejection_messaging_complete", False))
    email_id = getattr(rejector_result, "email_id", None)
    marked_read = _mark_terminal_email_read(email_id, "rejected") if notified else False

    logger.info(
        "Rejection branch completed | notified={} | email_id={} | marked_read={}",
        notified,
        email_id,
        marked_read,
    )

    await ctx.yield_output(
        {
            "email_id": email_id or "",
            "terminal_action": "rejected" if notified else "rejection_incomplete",
            "should_mark_read": notified,
            "marked_read": marked_read,
        }
    )

    await ctx.send_message(rejector_response)


@logger.catch
def create_workflow():
    """Construct a fresh workflow instance for each run."""
    return (
        WorkflowBuilder(name="po_pipeline_agents", start_executor=classifier)
        .add_edge(classifier, parser, condition=should_parse)
        .add_edge(classifier, consume_non_po_email, condition=should_consume_non_po)
        .add_edge(parser, retriever)
        .add_edge(retriever, check_agent_groundedness)
        .add_edge(check_agent_groundedness, decider, condition=should_be_grounded)
        .add_edge(decider, fulfiller, condition=should_fulfill)
        .add_chain([fulfiller, log_fulfillment, destroy_indexes])
        .add_edge(decider, rejector, condition=should_reject)
        .add_chain([rejector, log_rejection, destroy_indexes])
        .build()
    )


# It's a global workflow instance template; fresh instances are created per run.
workflow = create_workflow()

POLL_INTERVAL_SECONDS = int(os.getenv("GMAIL_POLL_INTERVAL_SECONDS", "60"))


@logger.catch
async def run_till_mail_read():  # async cuz we'll need to await workflow.run()
    """Run the workflow repeatedly until no unread Gmail messages remain."""
    processed = 0
    
    while True:
        unread_messages = fetch_unread_emails()
        if not unread_messages:
            logger.info(
                "No unread emails detected | total_processed={} | sleeping {}s",
                processed,
                POLL_INTERVAL_SECONDS,
            )
            await asyncio.sleep(POLL_INTERVAL_SECONDS)
            continue

        current = unread_messages[0]
        subject_preview = current.get("subject", "").strip()
        
        logger.info(
            "Email processing started | email_id={} | subject={}",
            current.get("id"),
            subject_preview or "[no subject]",
        )
    
        kickoff_prompt = (
            "Process the latest unread Gmail message. Classify it, "
            "then continue through parsing, resolution, and routing."
        )

        # Create a fresh workflow instance for this run (to avoid state leakage)
        workflow_instance = create_workflow()

        # Run the workflow
        logger.info("Starting workflow execution for email_id={}", current.get('id'))
        
        # result = await workflow_instance.run(kickoff_prompt)
        workflow_result = await workflow_instance.run(kickoff_prompt)  # await cuz run() is async

        logger.info("Workflow completed for email_id={}", current.get('id'))

        # Clear evidence to prevent leaking between workflow runs
        clear_evidence()

        terminal_output = _get_terminal_output(workflow_result, current["id"])

        if terminal_output and terminal_output.get("should_mark_read") and not terminal_output.get("marked_read"):
            terminal_output["marked_read"] = _mark_terminal_email_read(
                current["id"],
                f"{terminal_output.get('terminal_action', 'unknown')}_post_run_fallback",
            )

        if terminal_output and terminal_output.get("should_mark_read") and terminal_output.get("marked_read"):
            processed += 1
            logger.info(
                "Workflow reached stable terminal action | email_id={} | terminal_action={} | total_processed={}",
                current["id"],
                terminal_output.get("terminal_action", "unknown"),
                processed,
            )
        else:
            logger.warning(
                "Workflow ended without a successful terminal action; leaving email unread | email_id={} | terminal_output={}",
                current.get("id"),
                terminal_output,
            )
            await asyncio.sleep(2)
