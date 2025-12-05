from typing import List
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool, BaseTool
from langchain_google_genai import ChatGoogleGenerativeAI

from callbacks import AgentCallbackHandler

load_dotenv()


# -------------------------------
# 1) Define Tool
# -------------------------------
@tool
def get_text_length(text: str) -> int:
    """Returns the length of a text by characters."""
    print(f"get_text_length called with: {text=}")
    text = text.strip("'\n").strip('"')
    return len(text)


def find_tool_by_name(tools: List[BaseTool], name: str) -> BaseTool:
    for t in tools:
        if t.name == name:
            return t
    raise ValueError(f"Tool {name} not found")


# -------------------------------
# 2) Main Agent Logic
# -------------------------------
if __name__ == "__main__":
    print("Hello LangChain Tools with Gemini Function Calling!")

    tools = [get_text_length]

    # Gemini LLM with function calling support
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        callbacks=[AgentCallbackHandler()],
    )

    # Bind tools → enables native function calling
    llm_with_tools = llm.bind_tools(tools)

    # Start conversation
    messages = [HumanMessage(content="What is the length of the word: DOG")]

    while True:
        # Call Gemini with current messages
        ai_message = llm_with_tools.invoke(messages)

        # Check if the LLM wants to call a tool
        tool_calls = getattr(ai_message, "tool_calls", None) or []

        # ---------------------------
        # CASE 1: Gemini wants a tool
        # ---------------------------
        if len(tool_calls) > 0:
            messages.append(ai_message)

            for call in tool_calls:
                tool_name = call.get("name")
                tool_args = call.get("args", {})
                call_id = call.get("id")

                tool_to_use = find_tool_by_name(tools, tool_name)
                observation = tool_to_use.invoke(tool_args)

                print(f"Tool result = {observation}")

                # Send tool response back to model
                messages.append(
                    ToolMessage(
                        content=str(observation),
                        tool_call_id=call_id,
                    )
                )

            # Continue loop so model can finish
            continue

        # ---------------------------
        # CASE 2: No tool → Final Answer
        # ---------------------------
        print("\nFINAL ANSWER:")
        print(ai_message.content)
        break
