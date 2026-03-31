import shutil
from dotenv import load_dotenv

from agent_framework.azure import AzureOpenAIChatClient
from agent_framework.observability import configure_otel_providers
from azure.identity import AzureCliCredential, DefaultAzureCredential

# Ensure environment and telemetry are configured before agents initialize.
load_dotenv()
configure_otel_providers()

COGNITIVE_SERVICES_SCOPE = "https://cognitiveservices.azure.com/.default"


def _validate_credential(credential: DefaultAzureCredential | AzureCliCredential) -> None:
    """Fail fast unless the credential can mint a Cognitive Services token."""

    credential.get_token(COGNITIVE_SERVICES_SCOPE)


def _build_chat_client() -> AzureOpenAIChatClient:
    """Create a chat client using the best available authentication."""

    if shutil.which("az"):
        try:
            credential = AzureCliCredential()
            _validate_credential(credential)
            return AzureOpenAIChatClient(credential=credential)
        except Exception:
            pass

    # Fall back to managed identity / service principal style credentials.
    try:
        credential = DefaultAzureCredential(
            exclude_cli_credential=True,
            exclude_developer_cli_credential=True,
        )
        _validate_credential(credential)
        return AzureOpenAIChatClient(credential=credential)
    except Exception:
        pass

    raise RuntimeError(
        "Azure authentication not configured. Set service principal environment "
        "variables or run `az login` for the Azure subscription that owns the "
        "Azure OpenAI resource."
    )


# Shared chat client instance used by all agents.
chat_client = _build_chat_client()
