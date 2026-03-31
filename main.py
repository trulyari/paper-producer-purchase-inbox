"""Command-line entry point for the PaperCo PO intake workflow."""

import asyncio

from src.shared.logging_config import configure_logging
from src.workflow import workflow
from src.workflow.workflow import (
    run_till_mail_read,  # Import the async function to run the workflow
)

def main() -> None:
    """Start the asynchronous Gmail polling loop."""
    
    # Initialize logging ONCE at application startup
    configure_logging(level="DEBUG")
    
    # # UNCOMMENT BELOW to run the workflow:
    asyncio.run(run_till_mail_read())

    ################################# DevUI ###################################

    # # UNCOMMENT BELOW TWO LINES to start the DevUI to visualize the workflow:
    # from agent_framework_devui import serve  # serve means to start the dev UI
    # serve(entities_dir="src/workflow", auto_open=True)  # Automatically open the UI in a browser


if __name__ == "__main__":
    main()