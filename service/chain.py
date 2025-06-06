# chain.py - with Marqo Retriever + Ollama Mistral + HuggingFace Embedding
from operator import itemgetter
from os import path
from typing import Sequence, Dict, List, Any
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.language_models import LanguageModelLike
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableBranch, RunnableLambda, RunnablePassthrough, RunnableSequence, ConfigurableField
from langchain_core.runnables import chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama

from pydantic import Field
from marqo import Client
import config
from prompt import RESPONSE_TEMPLATE, LITERATURE_REVIEW_PROMPT, SUMMARIZE_PROMPT, REPHRASE_TEMPLATE, COHERE_RESPONSE_TEMPLATE

load_dotenv(path.join(config.PROJ_ROOT_DIR, '.env'))


# ✅ Marqo Retriever (Structured Index)
class MarqoRetriever(BaseRetriever):
    client: Any = Field(...)
    index_name: str = Field(...)
    embedding_model: Any = Field(...)
    embedding_field: str = Field(default="specter_embedding")
    k: int = Field(default=5)

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        query_vector = self.embedding_model.embed_query(query)
        results = self.client.index(self.index_name).search(
            context={"tensor": [{"vector": query_vector, "weight": 1.0}]},
            limit=self.k
        )
        return [
            Document(
                page_content=hit.get("Title", ""),
                metadata={
                    "Title": hit.get("Title", ""),
                    "Abstract": hit.get("Abstract", ""),
                    "Authors": hit.get("Authors", ""),
                    "Keywords": hit.get("Keywords", ""),
                    "Source": hit.get("Source", ""),
                    "Year": hit.get("Year", ""),
                    "id": hit.get("_id", "")
                }
            )
            for hit in results.get("hits", [])
        ]

    async def _aget_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        return self._get_relevant_documents(query)


def get_retriever() -> BaseRetriever:
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    client = Client("http://localhost:8882")
    return MarqoRetriever(client=client, index_name="specter-papers", embedding_model=embedding_model, k=5)


def format_docs(docs: Sequence[Document]) -> str:
    formatted_docs = []
    for i, doc in enumerate(docs):
        doc_string = (f"<doc id='{i}'>"
                      f"<Title>{doc.metadata.get('Title', '')}/<Title>"
                      f"<Abstract>{doc.metadata.get('Abstract', '')}/<Abstract>"
                      f"<Authors>{doc.metadata.get('Authors', '')}/<Authors>"
                      f"<Keywords>{doc.metadata.get('Keywords', '')}/<Keywords>"
                      f"<Source>{doc.metadata.get('Source', '')}/<Source>"
                      f"<Year>{doc.metadata.get('Year', '')}/<Year>"
                      f"</doc>")
        formatted_docs.append(doc_string)
    return "\n".join(formatted_docs)


def serialize_history(request: List[Dict[str, str]]):
    chat_history = request.get("chat_history", [])
    converted_chat_history = []
    for message in chat_history:
        if message.get("human"):
            converted_chat_history.append(HumanMessage(content=message["human"]))
        if message.get("ai"):
            converted_chat_history.append(AIMessage(content=message["ai"]))
    return converted_chat_history


def create_retriever_chain(llm: LanguageModelLike, retriever: BaseRetriever) -> Runnable:
    condense_question_prompt = PromptTemplate.from_template(REPHRASE_TEMPLATE)
    condense_question_chain = (condense_question_prompt | llm | StrOutputParser()).with_config(run_name="CondenseQuestion")
    return RunnableBranch(
        (
            RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(run_name="HasChatHistoryCheck"),
            (condense_question_chain | retriever).with_config(run_name="RetrievalChainWithHistory"),
        ),
        (
            RunnableLambda(itemgetter("question")).with_config(run_name="Itemgetter:question")
            | retriever
        ).with_config(run_name="RetrievalChainWithNoHistory"),
    ).with_config(run_name="RouteDependingOnChatHistory")


def create_chain(llm: LanguageModelLike, retriever: BaseRetriever) -> Runnable:
    retriever_chain = create_retriever_chain(llm, retriever).with_config(run_name="FindDocs")
    context = (
        RunnablePassthrough.assign(docs=retriever_chain)
        .assign(context=lambda x: format_docs(x["docs"]))
        .with_config(run_name="RetrieveDocs")
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", RESPONSE_TEMPLATE),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])
    default_response_synthesizer = prompt | llm

    cohere_prompt = ChatPromptTemplate.from_messages([
        ("system", COHERE_RESPONSE_TEMPLATE),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    @chain
    def cohere_response_synthesizer(input: dict) -> RunnableSequence:
        return cohere_prompt | llm.bind(source_documents=input["docs"])

    response_synthesizer = (
        default_response_synthesizer.configurable_alternatives(
            ConfigurableField("llm"),
            default_key="openai_gpt_3_5_turbo",
            cohere_command=cohere_response_synthesizer,
            anthropic_claude_3_sonnet=default_response_synthesizer,
            fireworks_mixtral=default_response_synthesizer,
            google_gemini_pro=default_response_synthesizer
        ) | StrOutputParser()
    ).with_config(run_name="GenerateResponse")

    return (
        RunnablePassthrough.assign(chat_history=serialize_history)
        | context
        | response_synthesizer
    )


# ✅ Stream summarization and literature review
def summarize_output(prompt_data): return (i.content for i in llm.stream(SUMMARIZE_PROMPT.format(**prompt_data)))
def literature_review_output(prompt_data): return (i.content for i in llm.stream(LITERATURE_REVIEW_PROMPT.format(**prompt_data)))

# ✅ Local model: Ollama Mistral
llm = ChatOllama(model="mistral", temperature=0, streaming=True)

# ✅ Build final chain
answer_chain = create_chain(llm, get_retriever())

# ✅ Streaming interface
def chat_streaming_output(question: str, chat_history: []):
    return answer_chain.stream({
        'question': question,
        'chat_history': chat_history
    })
def format_papers_in_prompt(papers):
    return '\n'.join([(f" --- "
            f"Title: {paper.get('Title', '')}\n"
            f"Authors: {paper.get('Authors', '')}\n"
            f"Abstract: {paper.get('Abstract', '')}\n"
            f"Source: {paper.get('Source', '')}\n"
            f"Year: {paper.get('Year', '')}\n"
            f"Keywords: {paper.get('Keywords', '')}\n"
            f" --- ") for paper in papers])