import os
import math
import asyncio
import getpass
import requests

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import (
    MaxMessageTermination,
    TextMentionTermination,
)
from autogen_ext.models.openai import OpenAIChatCompletionClient


if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = getpass.getpass(
        "Enter your Groq API Key: "
    )

API_KEY = os.environ["GROQ_API_KEY"]

def evaluate_expression(expression: str) -> str:
    """
    Evaluate a mathematical expression safely.
    """

    try:
        math_functions = {
            name: getattr(math, name)
            for name in dir(math)
            if not name.startswith("_")
        }

        result = eval(
            expression,
            {"__builtins__": {}},
            math_functions,
        )

        return f"Expression: {expression}\nResult: {result}"

    except Exception as error:
        return f"Calculation Error: {error}"


def fetch_stock_value(symbol: str) -> str:
    """
    Retrieve latest stock value from Yahoo Finance.
    """

    try:
        endpoint = (
            f"https://query1.finance.yahoo.com/v8/finance/chart/"
            f"{symbol.upper()}"
        )

        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        response = requests.get(
            endpoint,
            headers=headers,
            timeout=10,
        )

        payload = response.json()

        metadata = payload["chart"]["result"][0]["meta"]

        company_name = metadata.get(
            "shortName",
            symbol.upper(),
        )

        market_price = metadata["regularMarketPrice"]

        currency = metadata.get(
            "currency",
            "USD",
        )

        return (
            f"Company: {company_name}\n"
            f"Ticker: {symbol.upper()}\n"
            f"Price: {currency} {market_price:.2f}"
        )

    except Exception as error:
        return (
            f"Unable to retrieve stock information "
            f"for {symbol}: {error}"
        )

analyst_model = OpenAIChatCompletionClient(
    model="llama-3.1-8b-instant",
    base_url="https://api.groq.com/openai/v1",
    api_key=API_KEY,
    model_info={
        "vision": False,
        "function_calling": True,
        "json_output": True,
        "structured_output": True,
        "family": "unknown",
    },
)

writer_model = OpenAIChatCompletionClient(
    model="llama-3.3-70b-versatile",
    base_url="https://api.groq.com/openai/v1",
    api_key=API_KEY,
    model_info={
        "vision": False,
        "function_calling": True,
        "json_output": True,
        "structured_output": True,
        "family": "unknown",
    },
)

data_analyst = AssistantAgent(
    name="DataAnalyst",
    model_client=analyst_model,
    tools=[
        evaluate_expression,
        fetch_stock_value,
    ],
    system_message=(
        "You are a financial analysis assistant. "
        "Use the available tools whenever numerical "
        "calculations or stock market information "
        "is required. Present your findings clearly "
        "and conclude with TERMINATE."
    ),
)

approval_user = UserProxyAgent(
    name="ApprovalUser",
    input_func=None,
)

report_writer = AssistantAgent(
    name="ReportWriter",
    model_client=writer_model,
    system_message=(
        "You are responsible for producing the final "
        "investment report. Review the analysis output "
        "and any reviewer feedback. Refine the content "
        "into a professional response and conclude "
        "with TERMINATE."
    ),
)

stop_condition = (
    TextMentionTermination("TERMINATE")
    | MaxMessageTermination(max_messages=8)
)

workflow_team = RoundRobinGroupChat(
    participants=[
        data_analyst,
        approval_user,
        report_writer,
    ],
    termination_condition=stop_condition,
)
async def main():

    print("=" * 60)
    print(" AutoGen + Groq Financial Analysis Workflow ")
    print("=" * 60)

    print("\nInstructions:")
    print(
        "After the analyst finishes, "
        "the workflow pauses for your review."
    )
    print(
        "Press Enter to approve "
        "or type feedback before continuing."
    )
    print(
        "Type TERMINATE if you wish "
        "to stop the workflow.\n"
    )

    task = (
        "Retrieve the latest stock values for "
        "Apple (AAPL) and Nvidia (NVDA). "
        "Compute the investment required for "
        "purchasing 10 shares of each company. "
        "Prepare a short comparative "
        "investment overview."
    )

    print(f"Task:\n{task}\n")
    print("-" * 60)

    async for message in workflow_team.run_stream(
        task=task
    ):

        source = getattr(
            message,
            "source",
            "System",
        )

        content = getattr(
            message,
            "content",
            str(message),
        )

        labels = {
            "DataAnalyst":
                "[Data Analyst]",
            "ApprovalUser":
                "[Reviewer]",
            "ReportWriter":
                "[Report Writer]",
        }

        current_label = labels.get(
            source,
            f"[{source}]",
        )

        print(f"\n{current_label}")
        print(content)
        print("-" * 60)

if __name__ == "__main__":
    asyncio.run(main())