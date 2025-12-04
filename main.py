from typing import List, Union

from dotenv import load_dotenv
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool, render_text_description, tool
from langchain_google_genai import ChatGoogleGenerativeAI

from callbacks import AgentCallbackHandler

load_dotenv()


@tool
def get_text_length(text: str) -> int:
    """Returns the length of a text by characters"""
    print(f"get_text_length enter with {text=}")
    text = text.strip("'\n").strip('"')
    return len(text)


def format_log_to_str(intermediate_steps: List[tuple]) -> str:
    """Format intermediate steps to string for agent scratchpad."""
    thoughts = ""
    for action, observation in intermediate_steps:
        thoughts += f"{action.log}\nObservation: {observation}\n"
    return thoughts


def parse_react_output(text: str) -> Union[AgentAction, AgentFinish]:
    """Parse ReAct style output from LLM."""
    if "Final Answer:" in text:
        return AgentFinish(
            return_values={"output": text.split("Final Answer:")[-1].strip()},
            log=text,
        )

    # Extract Action and Action Input
    if "Action:" in text and "Action Input:" in text:
        action_match = text.split("Action:")[1].split("Action Input:")[0].strip()
        action_input_match = text.split("Action Input:")[1].strip()

        # Remove any trailing "Observation" if present
        if "Observation:" in action_input_match:
            action_input_match = action_input_match.split("Observation:")[0].strip()

        return AgentAction(tool=action_match, tool_input=action_input_match, log=text)

    raise ValueError(f"Could not parse LLM output: `{text}`")


def find_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool with name {tool_name} not found")


if __name__ == "__main__":
    print("Hello ReAct LangChain!")
    tools = [get_text_length]

    template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools]),
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        callbacks=[AgentCallbackHandler()],
    )

    intermediate_steps = []
    agent_step = ""

    while not isinstance(agent_step, AgentFinish):
        # Format the agent scratchpad
        agent_scratchpad = format_log_to_str(intermediate_steps)

        # Create the full prompt
        prompt_text = prompt.format(
            input="What is the length of the word: DOG",
            agent_scratchpad=agent_scratchpad,
        )

        # Call the LLM
        response = llm.invoke(prompt_text)
        response_text = response.content

        # Parse the response
        try:
            agent_step = parse_react_output(response_text)
        except ValueError as e:
            print(f"Error parsing output: {e}")
            print(f"Response was: {response_text}")
            break

        print(agent_step)

        if isinstance(agent_step, AgentAction):
            tool_name = agent_step.tool
            try:
                tool_to_use = find_tool_by_name(tools, tool_name)
                tool_input = agent_step.tool_input
                observation = tool_to_use.func(str(tool_input))
                print(f"{observation=}")
                intermediate_steps.append((agent_step, str(observation)))
            except ValueError as e:
                print(f"Error: {e}")
                break

    if isinstance(agent_step, AgentFinish):
        print(agent_step.return_values)
