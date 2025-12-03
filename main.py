from langchain_ollama import ChatOllama
# from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()




def main():
    information = """
    Elon Musk is a business magnate, investor, and engineer.
    He is the founder of SpaceX, CEO of Tesla, and owner of X.com.
    """

    summary_template = """
    Given the information {information} about a person, I want you to create:
    1. A short summary
    2. Two interesting facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template
    )

    # OFFLINE LLM — NO API KEYS
    llm = ChatOllama(
        temperature=0,
        model="gemma3:270m"
    )

    chain = summary_prompt_template | llm

    response = chain.invoke({"information": information})
    print(response.content)

if __name__ == "__main__":
    main()
