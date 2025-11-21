import os
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import LanguageModelLike
from langchain_openai import AzureChatOpenAI
from prompt import SUMMARIZE_PROMPT, LITERATURE_REVIEW_PROMPT



# === Load env and init model ===
load_dotenv()

llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    streaming=True
)

# === Helper: Convert frontend chat history ===
def serialize_history(request: Dict[str, Any]):
    chat_history_raw = request.get("chat_history", [])
    converted = []
    for msg in chat_history_raw:
        role = msg.get("role", "user")
        text = msg.get("human") or msg.get("ai") or msg.get("text", "")
        converted.append({"role": role, "content": text})
    return converted


# === 1️⃣ Normal Chat Chain (no retrieval) ===
def create_plain_chain(llm: LanguageModelLike):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant for general conversation."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

    return (
        RunnablePassthrough.assign(
            chat_history=RunnableLambda(
                lambda x: serialize_history({"chat_history": x.get("chat_history_raw", [])})
            )
        )
        | prompt
        | llm
        | StrOutputParser()
    )

plain_chain = create_plain_chain(llm)


# === 2️⃣ Main Chat Entry (decides by mode) ===
def chat_streaming_output(
    question: str,
    chat_history: List[Dict[str, str]],
    mode: str = "normal",
    **kwargs
):
    """
    Handles both normal and RAG chat modes.
    RAG mode will be handled by Agent in agent_runner.py.
    """
    if mode == "normal":
        logging.info("🟢 Normal Chat Mode → using plain_chain")
        return plain_chain.stream({
            "question": question,
            "chat_history_raw": chat_history
        })
    else:
        logging.info("🔵 RAG Mode → delegated to agent_runner (handled externally)")
        yield "RAG mode is now managed by the Agent framework. Please check agent_runner.py."


# === 3️⃣ Other utilities (still useful for API calls) ===
def summarize_output(prompt_data: Dict[str, str]):
    return (i.content for i in llm.stream(SUMMARIZE_PROMPT.format(**prompt_data)))

def literature_review_output(prompt_data: Dict[str, str]):
    return (i.content for i in llm.stream(LITERATURE_REVIEW_PROMPT.format(**prompt_data)))