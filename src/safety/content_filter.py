import os
from dotenv import load_dotenv
from loguru import logger

from azure.ai.contentsafety import ContentSafetyClient
from azure.ai.contentsafety.models import AnalyzeTextOptions
from azure.identity import DefaultAzureCredential
from agent_framework import tool

load_dotenv()


@tool
def check_email_content_safety(email_body: str, threshold: int = 4) -> dict:
    """
    Check if email text contains harmful content (Hate, Self-Harm, Sexual, Violence).
    
    Args:
        email_body: The email body to analyze
        threshold: Severity score threshold (0-7, default 4)
    Returns:
        dict: {
            "is_safe": bool,  # True if no categories exceed threshold
            "categories_flagged": list[dict],  # List of flagged categories with severity
        }
    """
    logger.info(
        "[FUNCTION check_email_content_safety] Content safety check started for email."
    )
    
    # Initialize client with managed identity
    endpoint = os.getenv("CONTENT_SAFETY_ENDPOINT")
    if not endpoint:
        raise ValueError("CONTENT_SAFETY_ENDPOINT env variable must be set!")
    
    # Create Content Safety client instance
    client = ContentSafetyClient(
        endpoint=endpoint,
        credential=DefaultAzureCredential()
    )
    
    # Analyze text, which checks all 4 categories automatically:
    # Hate, Self-Harm, Sexual, Violence
    result = client.analyze_text(AnalyzeTextOptions(text=email_body))
    
    # Flag categories that exceed threshold
    categories_flagged = [
        {"category": cat.category, "severity": cat.severity}
        for cat in result.categories_analysis
        if cat.severity is not None and cat.severity >= threshold
    ]
    
    logger.info(
        "[FUNCTION check_email_content_safety] Content safety check completed, "
        "with safety result: {}, and flagged categories: {}",
        len(categories_flagged) == 0,
        categories_flagged
    )

    return {
        # this line returns True if no categories are flagged (length 0) else False
        "is_safe": len(categories_flagged) == 0,
        "categories_flagged": categories_flagged
    }




#############################################
###### Example local usage for testing ######
#############################################

if __name__ == "__main__":
    # Test safe content
    print(
        "Safe test:",
        check_email_content_safety("Please send 100 reams of paper")
    )

    # Test harmful content
    print(
        "Harmful test:",
        check_email_content_safety( 
            "You're an asshole!",  # intentionally mild profanity for testing
            threshold=2  # Threshold means "flag if severity is 2 or higher" 
        )
    )
