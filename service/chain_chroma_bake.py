import json
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
from langchain_core.runnables import Runnable, RunnableBranch, RunnableLambda, RunnablePassthrough
# from langchain_ollama import ChatOllama
from langchain_core.embeddings import Embeddings
from service.embed_chroma import LocalSpecterEmbedding, LocalSentenceTransformerEmbedding
import chromadb
from pydantic import Field, model_validator
import config
from prompt import RESPONSE_TEMPLATE, LITERATURE_REVIEW_PROMPT, SUMMARIZE_PROMPT, REPHRASE_TEMPLATE
import numpy as np
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from logger_config import get_logger

# Use centralized logger
logging = get_logger()

load_dotenv(path.join(config.PROJ_ROOT_DIR, '.env'))


# === Chroma Retriever (enhanced localization support) ===
# This class remains unchanged since it was already designed to support dynamic embedding selection.
class LocalizedChromaRetriever(BaseRetriever):
    client: Any = Field(...)
    collection_mapping: Dict[str, str] = Field(...) 
    embedding_models: Dict[str, Embeddings] = Field(...) 
    k: int = Field(default=5)

    @model_validator(mode='after')
    def check_embedding_models(self):
        required_models = set(self.collection_mapping.keys())
        for model_type in required_models:
            if model_type not in self.embedding_models:
                raise ValueError(f"Missing embedding model for type: {model_type}")
        return self

    def _get_relevant_documents(self, query: str, *, run_manager=None, 
                                embedding_type: str = "specter", lang: str = None) -> List[Document]:
        if embedding_type not in self.embedding_models:
            logging.error(f"Unsupported embedding type '{embedding_type}'. Available: {list(self.embedding_models.keys())}")
            return []
        
        embedding_model = self.embedding_models[embedding_type]
        collection_name = self.collection_mapping.get(embedding_type)

        if not collection_name:
            logging.error(f"No collection name mapped for embedding type: {embedding_type}")
            return []

        if self.client is None:
            logging.error("ChromaDB client is not initialized.")
            return []

        try:
            collection = self.client.get_collection(collection_name)
        except Exception as e:
            logging.error(f"Failed to get ChromaDB collection '{collection_name}': {e}")
            return []

        query_vector = embedding_model.embed_query(query)
        
        where_clause = {}
        if lang and lang.lower() != "all": 
            where_clause["lang"] = lang.lower()
            logging.info(f"Applying language filter: {where_clause} for collection '{collection_name}'")
        else:
            logging.info(f"No specific language filter applied for collection '{collection_name}'. Performing cross-lingual search.")


        results = collection.query(
            query_embeddings=[query_vector],
            n_results=self.k,
            include=["metadatas", "distances"],
            where=where_clause if where_clause else None
        )

        metadatas = results['metadatas'][0]
        distances = results['distances'][0]

        docs = []
        for i, (m, d) in enumerate(zip(metadatas, distances)):
            lowercase_m = {k.lower(): v for k, v in m.items()}
            
            doc_id = lowercase_m.get("original_id", lowercase_m.get("id", f"doc_{i}")) 
            
            similarity_score = max(0.0, min(1.0, 1 / (1 + d)))

            docs.append(
                Document(
                    page_content=lowercase_m.get("abstract", "") or lowercase_m.get("title", ""),
                    metadata={
                        "title": lowercase_m.get("title", ""),
                        "abstract": lowercase_m.get("abstract", ""),
                        "authors": lowercase_m.get("authors", []),
                        "keywords": lowercase_m.get("keywords", []),
                        "source": lowercase_m.get("source", ""),
                        "year": lowercase_m.get("year", ""),
                        "id": doc_id,
                        "lang": lowercase_m.get("lang", "unknown"),
                        "_score": similarity_score
                    }
                )
            )
        return docs

    async def _aget_relevant_documents(self, query: str, *, run_manager=None, 
                                     embedding_type: str = "specter", lang: str = None) -> List[Document]:
        return self._get_relevant_documents(query, run_manager=run_manager, embedding_type=embedding_type, lang=lang)


def get_retriever(embedding_type: str = "specter", lang: str = None) -> BaseRetriever:
    """
    Get a configured retriever instance based on embedding type and language.
    
    Args:
        embedding_type: A string indicating the embedding type ("specter", "ada", "glove").
        lang: Optional language code (e.g., "en", "zh").
    Returns:
        A BaseRetriever instance.
    """
    client = chromadb.PersistentClient(path="chroma_db")
    logging.info(f"🔍 Using ChromaDB client at path: chroma_db. Collections: {client.list_collections()}")

    # Instantiate all required embedding models
    embedding_models_instances = {
        # Specter embedding model using local class
        "specter": LocalSpecterEmbedding(model_name="allenai/specter"), 
        # Ada embedding model using local class
        "ada": LocalSentenceTransformerEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        # GloVe embedding model using local class
        "glove": LocalSentenceTransformerEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") 
    }

    # ChromaDB collection mapping (unchanged)
    chroma_collection_mapping = {
        "specter": "paper_specter",
        "ada": "paper_ada_localized",
        "glove": "paper_glove_localized"
    }

    if embedding_type not in chroma_collection_mapping:
        logging.error(f"Requested embedding type '{embedding_type}' is not supported in retriever configuration. Falling back to 'specter'.")
        embedding_type = "specter"

    retriever_instance = LocalizedChromaRetriever(
        client=client,
        collection_mapping=chroma_collection_mapping,
        embedding_models=embedding_models_instances,
        k=5
    )
    logging.info(f"✅ Created LocalizedChromaRetriever with embedding type '{embedding_type}' and language '{lang}'.")
    return retriever_instance


# def format_docs(docs: Sequence[Document]) -> str:
#     formatted_docs = []
#     for i, doc in enumerate(docs):
#         metadata = doc.metadata or {}
#         doc_id = metadata.get('id', f"doc_{i}")
#         doc_string = (
#             f"<doc id='{doc_id}' score='{metadata.get('_score', 0.0):.4f}' lang='{metadata.get('lang', 'unknown')}'>"
#             # f"<title>{metadata.get('title', '')}</title>"
#             f"<title>{metadata.get('title', '')} [[ID:{doc_id}]]</title>"  # Include ID in title for clarity
#             f"<abstract>{metadata.get('abstract', '')}</abstract>"
#             f"<authors>{', '.join(metadata.get('authors', []))}</authors>"
#             f"<keywords>{', '.join(metadata.get('keywords', []))}</keywords>"
#             f"<source>{metadata.get('source', '')}</source>"
#             f"<year>{metadata.get('year', '')}</year>"
#             f"</doc>"
#         )
#         formatted_docs.append(doc_string)
#     return "\n".join(formatted_docs)

# def format_docs(docs: Sequence[Document]) -> str:
#     formatted_docs = []
#     for i, doc in enumerate(docs):
#         metadata = doc.metadata or {}
#         doc_id = metadata.get('id', f"doc_{i}")

#         # Handle authors and keywords which might be lists
#         authors = metadata.get('authors', "")
#         if isinstance(authors, list):
#             authors = ", ".join([str(a).strip() for a in authors])

#         keywords = metadata.get('keywords', "")
#         if isinstance(keywords, list):
#             keywords = ", ".join([str(k).strip() for k in keywords])

#         doc_string = (
#             f"<doc id='{doc_id}' score='{metadata.get('_score', 0.0):.4f}' lang='{metadata.get('lang', 'unknown')}'>"
#             f"<title>{metadata.get('title', '')} [[ID:{doc_id}]]</title>"
#             f"<abstract>{metadata.get('abstract', '')}</abstract>"
#             f"<authors>{authors}</authors>"
#             f"<keywords>{keywords}</keywords>"
#             f"<source>{metadata.get('source', '')}</source>"
#             f"<year>{metadata.get('year', '')}</year>"
#             f"</doc>"
#         )
#         formatted_docs.append(doc_string)
#     return "\n".join(formatted_docs)


def format_docs(docs: Sequence[Document]) -> str:
    """
    Format docs into Markdown list items so that front-end Markdown parser
    can render titles with clickable blue text + action buttons.
    """
    formatted_docs = []
    for i, doc in enumerate(docs):
        metadata = doc.metadata or {}
        doc_id = metadata.get('id', f"doc_{i}")

        # Authors
        authors = metadata.get('authors', "")
        if isinstance(authors, list):
            authors = ", ".join([str(a).strip() for a in authors])

        # Keywords
        keywords = metadata.get('keywords', "")
        if isinstance(keywords, list):
            keywords = ", ".join([str(k).strip() for k in keywords])

        # Construct Markdown string
        doc_string = (
            f"- **Title: {metadata.get('title', '')} [[ID:{doc_id}]]**\n"
            f"  - Authors: {authors}\n"
            f"  - Year: {metadata.get('year', '')}\n"
            f"  - Source: {metadata.get('source', '')}\n"
            f"  - Keywords: {keywords}\n"
            f"  - Score: {metadata.get('_score', 0.0):.4f}\n"
        )
        formatted_docs.append(doc_string)

    return "\n".join(formatted_docs)



# def serialize_history(request: Dict[str, Any]):
#     chat_history_raw = request.get("chat_history", [])
#     converted_chat_history = []
#     for message in chat_history_raw:
#         if message.get("human"):
#             converted_chat_history.append(HumanMessage(content=message["human"]))
#         if message.get("ai"):
#             converted_chat_history.append(AIMessage(content=message["ai"]))
#     return converted_chat_history

# def serialize_history(request: Dict[str, Any]):
#     chat_history_raw = request.get("chat_history", [])
#     converted_chat_history = []
#     for message in chat_history_raw:
#         if message.get("human") or message.get("role") == "user":
#             converted_chat_history.append(
#                 HumanMessage(content=message.get("human") or message.get("text"))
#             )
#         if message.get("ai") or message.get("role") == "ai":
#             converted_chat_history.append(
#                 AIMessage(content=message.get("ai") or message.get("text"))
#             )
#     return converted_chat_history


def serialize_history(request: Dict[str, Any]):
    chat_history_raw = request.get("chat_history", [])
    print("🔥 serialize_history output:", chat_history_raw)   
    converted_chat_history = []
    for message in chat_history_raw:
        if message.get("human") or message.get("role") == "user":
            converted_chat_history.append(
                HumanMessage(content=message.get("human") or message.get("text"))
            )
        if message.get("ai") or message.get("role") == "ai":
            converted_chat_history.append(
                AIMessage(content=message.get("ai") or message.get("text"))
            )
    print("✅ serialize_history output:", converted_chat_history)
    return converted_chat_history


def create_retriever_chain(llm: LanguageModelLike) -> Runnable:
    condense_question_prompt = PromptTemplate.from_template(REPHRASE_TEMPLATE)
    condense_question_chain = (condense_question_prompt | llm | StrOutputParser()).with_config(run_name="CondenseQuestion")

    def get_dynamic_retriever(inputs: Dict[str, Any]) -> BaseRetriever:
        embedding_type = inputs.get("embedding_type", "specter")
        lang = inputs.get("lang", None)
        return get_retriever(embedding_type=embedding_type, lang=lang)

    return RunnableBranch(
        (
            RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(run_name="HasChatHistoryCheck"),
            (
                RunnablePassthrough.assign(
                    _retriever_params=RunnableLambda(lambda x: {"embedding_type": x.get("embedding_type"), "lang": x.get("lang")})
                )
                | RunnablePassthrough.assign(
                    rephrased_question=condense_question_chain
                )
                | RunnableLambda(
                    lambda x: get_dynamic_retriever(x["_retriever_params"])._get_relevant_documents(
                        x["rephrased_question"], 
                        embedding_type=x["_retriever_params"]["embedding_type"],
                        lang=x["_retriever_params"]["lang"]
                    )
                )
            ).with_config(run_name="RetrievalChainWithHistory"),
        ),
        (
            RunnablePassthrough.assign(
                _retriever_params=RunnableLambda(lambda x: {"embedding_type": x.get("embedding_type"), "lang": x.get("lang")})
            )
            | RunnableLambda(
                lambda x: get_dynamic_retriever(x["_retriever_params"])._get_relevant_documents(
                    x["question"], 
                    embedding_type=x["_retriever_params"]["embedding_type"],
                    lang=x["_retriever_params"]["lang"]
                )
            )
        ).with_config(run_name="RetrievalChainWithNoHistory"),
    ).with_config(run_name="RouteDependingOnChatHistory")


def create_chain(llm: LanguageModelLike) -> Runnable:
    retriever_chain = create_retriever_chain(llm).with_config(run_name="FindDocs")
    
    context = (
        RunnablePassthrough.assign(docs=retriever_chain)
        .assign(context=RunnableLambda(lambda x: logging.info(f"\U0001F4DA Retrieved Docs:\n {format_docs(x['docs'])}") or format_docs(x["docs"])))
        .with_config(run_name="RetrieveDocs")
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", RESPONSE_TEMPLATE),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}\n\nPlease answer the question based on the following documents:\n{context}"),
    ])
    default_response_synthesizer = prompt | llm | StrOutputParser()
    
    return (
        RunnablePassthrough.assign(
            chat_history=RunnableLambda(
                lambda x: serialize_history({"chat_history": x.get("chat_history_raw", [])})
            )
        )
        | context
        | default_response_synthesizer
    )

# def create_chain(llm: LanguageModelLike) -> Runnable:
#     retriever_chain = create_retriever_chain(llm).with_config(run_name="FindDocs")
    
#     context = (
#         RunnablePassthrough.assign(docs=retriever_chain)
#         .assign(context=RunnableLambda(lambda x: logging.info(f"📚 Retrieved Docs:\n {format_docs(x['docs'])}") or format_docs(x["docs"])))
#         .with_config(run_name="RetrieveDocs")
#     )
    
#     prompt = ChatPromptTemplate.from_messages([
#         ("system", RESPONSE_TEMPLATE),
#         MessagesPlaceholder(variable_name="chat_history"),
#         ("human", "{question}\n\nPlease answer based only on {context}")
#     ])
#     default_response_synthesizer = prompt | llm | StrOutputParser()
    
#     def attach_titles(x):
#         llm_answer = x["output"]
#         titles = format_docs(x["docs"])
#         return llm_answer + "\n\nRetrieved papers:\n" + titles

#     return (
#         RunnablePassthrough.assign(chat_history=serialize_history)
#         | context
#         | RunnablePassthrough.assign(output=default_response_synthesizer)
#         | RunnableLambda(attach_titles)
#     )







from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import LanguageModelLike

def create_plain_chain(llm: LanguageModelLike):
    """
    Create a simple chat chain without retrieval.
    """
    plain_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Answer clearly and concisely."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

    return (
        RunnablePassthrough.assign(
            chat_history=RunnableLambda(
                lambda x: serialize_history({"chat_history": x.get("chat_history_raw", [])})
            )
        )
        | plain_prompt
        | llm
        | StrOutputParser()
    )


# llm = ChatOllama(model="mistral", temperature=0, streaming=True)
load_dotenv()

llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    streaming=True   # enable streaming
)

answer_chain = create_chain(llm)         ## RAG
plain_chain = create_plain_chain(llm)    ## noraml chat


# def chat_streaming_output(question: str, chat_history: List[Dict[str, str]], embedding_type: str = "specter", lang: str = None):
#     """
#     Main chat function supporting custom embedding and language.
#     Args:
#         question: The user's question.
#         chat_history: Previous chat history.
#         embedding_type: Embedding type to use ("specter", "ada", "glove").
#         lang: Language code (e.g., "en", "zh").
#     """
#     return answer_chain.stream({
#         'question': question,
#         'chat_history': chat_history,
#         'embedding_type': embedding_type,
#         'lang': lang
#     })


def chat_streaming_output(
    question: str,
    chat_history: List[Dict[str, str]],
    embedding_type: str = "specter",
    lang: str = None,
    mode: str = "rag"
):
    """
    Main chat function supporting both normal and RAG modes.
    Args:
        question: The user's question.
        chat_history: Previous chat history.
        embedding_type: Embedding type to use ("specter", "ada", "glove").
        lang: Language code (e.g., "en", "zh").
        mode: "normal" for plain LLM chat, "rag" for retrieval-augmented chat
    """
    if mode == "normal":
     
        # normal chat, do not use retrieval
        return plain_chain.stream({
            "question": question,
            "chat_history_raw": chat_history
        })
    else:
        # default to RAG mode
        return answer_chain.stream({
            "question": question,
            "chat_history_raw": chat_history,
            "embedding_type": embedding_type,
            "lang": lang
        })


def summarize_output(prompt_data: Dict[str, str]):
    return (i.content for i in llm.stream(SUMMARIZE_PROMPT.format(**prompt_data)))


def literature_review_output(prompt_data: Dict[str, str]):
    return (i.content for i in llm.stream(LITERATURE_REVIEW_PROMPT.format(**prompt_data)))


def format_papers_in_prompt(papers: List[Dict[str, Any]]) -> str:
    """
    Format a list of papers into a string suitable for LLM prompt input.
    """
    return '\n'.join([
        f" --- \nTitle: {p.get('Title', '')}\nAuthors: {', '.join(p.get('Authors', []))}\nAbstract: {p.get('Abstract', '')}\nSource: {p.get('Source', '')}\nYear: {p.get('Year', '')}\nKeywords: {', '.join(p.get('Keywords', []))}\n ---"
        for p in papers
    ])

####################################################
import json
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
from langchain_core.runnables import Runnable, RunnableBranch, RunnableLambda, RunnablePassthrough
# from langchain_ollama import ChatOllama
from langchain_core.embeddings import Embeddings 
from service.embed_chroma import LocalSpecterEmbedding, LocalSentenceTransformerEmbedding 
import chromadb  
from pydantic import Field, model_validator 
import config
from prompt import RESPONSE_TEMPLATE, LITERATURE_REVIEW_PROMPT, SUMMARIZE_PROMPT, REPHRASE_TEMPLATE
import numpy as np 
import logging 
import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
import re
from dataclasses import dataclass

# Note: Centralized logger already initialized at top of file

load_dotenv(path.join(config.PROJ_ROOT_DIR, '.env'))


# === Chroma Retriever (enhanced localization support) ===
# This class remains unchanged since it was already designed to support dynamic embedding selection.
class LocalizedChromaRetriever(BaseRetriever):
    client: Any = Field(...)
    collection_mapping: Dict[str, str] = Field(...) 
    embedding_models: Dict[str, Embeddings] = Field(...) 
    k: int = Field(default=5)

    @model_validator(mode='after')
    def check_embedding_models(self):
        required_models = set(self.collection_mapping.keys())
        for model_type in required_models:
            if model_type not in self.embedding_models:
                raise ValueError(f"Missing embedding model for type: {model_type}")
        return self

    def _get_relevant_documents(self, query: str, *, run_manager=None, 
                                embedding_type: str = "specter", lang: str = None) -> List[Document]:
        if embedding_type not in self.embedding_models:
            logging.error(f"Unsupported embedding type '{embedding_type}'. Available: {list(self.embedding_models.keys())}")
            return []
        
        embedding_model = self.embedding_models[embedding_type]
        collection_name = self.collection_mapping.get(embedding_type)

        if not collection_name:
            logging.error(f"No collection name mapped for embedding type: {embedding_type}")
            return []

        if self.client is None:
            logging.error("ChromaDB client is not initialized.")
            return []

        try:
            collection = self.client.get_collection(collection_name)
        except Exception as e:
            logging.error(f"Failed to get ChromaDB collection '{collection_name}': {e}")
            return []

        query_vector = embedding_model.embed_query(query)
        
        where_clause = {}
        if lang and lang.lower() != "all": 
            where_clause["lang"] = lang.lower()
            logging.info(f"Applying language filter: {where_clause} for collection '{collection_name}'")
        else:
            logging.info(f"No specific language filter applied for collection '{collection_name}'. Performing cross-lingual search.")


        results = collection.query(
            query_embeddings=[query_vector],
            n_results=self.k,
            include=["metadatas", "distances"],
            where=where_clause if where_clause else None
        )

        metadatas = results['metadatas'][0]
        distances = results['distances'][0]

        docs = []
        for i, (m, d) in enumerate(zip(metadatas, distances)):
            lowercase_m = {k.lower(): v for k, v in m.items()}
            
            doc_id = lowercase_m.get("original_id", lowercase_m.get("id", f"doc_{i}")) 
            
            similarity_score = max(0.0, min(1.0, 1 / (1 + d)))

            docs.append(
                Document(
                    page_content=lowercase_m.get("abstract", "") or lowercase_m.get("title", ""),
                    metadata={
                        "title": lowercase_m.get("title", ""),
                        "abstract": lowercase_m.get("abstract", ""),
                        "authors": lowercase_m.get("authors", []),
                        "keywords": lowercase_m.get("keywords", []),
                        "source": lowercase_m.get("source", ""),
                        "year": lowercase_m.get("year", ""),
                        "id": doc_id,
                        "lang": lowercase_m.get("lang", "unknown"),
                        "_score": similarity_score
                    }
                )
            )
        return docs

    async def _aget_relevant_documents(self, query: str, *, run_manager=None, 
                                     embedding_type: str = "specter", lang: str = None) -> List[Document]:
        return self._get_relevant_documents(query, run_manager=run_manager, embedding_type=embedding_type, lang=lang)


def get_retriever(embedding_type: str = "specter", lang: str = None) -> BaseRetriever:
    """
    Get a configured retriever instance based on embedding type and language.
    
    Args:
        embedding_type: A string indicating the embedding type ("specter", "ada", "glove").
        lang: Optional language code (e.g., "en", "zh").
    Returns:
        A BaseRetriever instance.
    """
    client = chromadb.PersistentClient(path="chroma_db")
    logging.info(f"🔍 Using ChromaDB client at path: chroma_db. Collections: {client.list_collections()}")

    # Instantiate all required embedding models
    embedding_models_instances = {
        # Specter embedding model using local class
        "specter": LocalSpecterEmbedding(model_name="allenai/specter"), 
        # Ada embedding model using local class
        "ada": LocalSentenceTransformerEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        # GloVe embedding model using local class
        "glove": LocalSentenceTransformerEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") 
    }

    # ChromaDB collection mapping (unchanged)
    chroma_collection_mapping = {
        "specter": "paper_specter",
        "ada": "paper_ada_localized",
        "glove": "paper_glove_localized"
    }

    if embedding_type not in chroma_collection_mapping:
        logging.error(f"Requested embedding type '{embedding_type}' is not supported in retriever configuration. Falling back to 'specter'.")
        embedding_type = "specter"

    retriever_instance = LocalizedChromaRetriever(
        client=client,
        collection_mapping=chroma_collection_mapping,
        embedding_models=embedding_models_instances,
        k=5
    )
    logging.info(f"✅ Created LocalizedChromaRetriever with embedding type '{embedding_type}' and language '{lang}'.")
    return retriever_instance


# def format_docs(docs: Sequence[Document]) -> str:
#     formatted_docs = []
#     for i, doc in enumerate(docs):
#         metadata = doc.metadata or {}
#         doc_id = metadata.get('id', f"doc_{i}")
#         doc_string = (
#             f"<doc id='{doc_id}' score='{metadata.get('_score', 0.0):.4f}' lang='{metadata.get('lang', 'unknown')}'>"
#             # f"<title>{metadata.get('title', '')}</title>"
#             f"<title>{metadata.get('title', '')} [[ID:{doc_id}]]</title>"  # Include ID in title for clarity
#             f"<abstract>{metadata.get('abstract', '')}</abstract>"
#             f"<authors>{', '.join(metadata.get('authors', []))}</authors>"
#             f"<keywords>{', '.join(metadata.get('keywords', []))}</keywords>"
#             f"<source>{metadata.get('source', '')}</source>"
#             f"<year>{metadata.get('year', '')}</year>"
#             f"</doc>"
#         )
#         formatted_docs.append(doc_string)
#     return "\n".join(formatted_docs)

# def format_docs(docs: Sequence[Document]) -> str:
#     formatted_docs = []
#     for i, doc in enumerate(docs):
#         metadata = doc.metadata or {}
#         doc_id = metadata.get('id', f"doc_{i}")

#         # Handle authors and keywords which might be lists
#         authors = metadata.get('authors', "")
#         if isinstance(authors, list):
#             authors = ", ".join([str(a).strip() for a in authors])

#         keywords = metadata.get('keywords', "")
#         if isinstance(keywords, list):
#             keywords = ", ".join([str(k).strip() for k in keywords])

#         doc_string = (
#             f"<doc id='{doc_id}' score='{metadata.get('_score', 0.0):.4f}' lang='{metadata.get('lang', 'unknown')}'>"
#             f"<title>{metadata.get('title', '')} [[ID:{doc_id}]]</title>"
#             f"<abstract>{metadata.get('abstract', '')}</abstract>"
#             f"<authors>{authors}</authors>"
#             f"<keywords>{keywords}</keywords>"
#             f"<source>{metadata.get('source', '')}</source>"
#             f"<year>{metadata.get('year', '')}</year>"
#             f"</doc>"
#         )
#         formatted_docs.append(doc_string)
#     return "\n".join(formatted_docs)


def format_docs(docs: Sequence[Document]) -> str:
    """
    Format docs into Markdown list items so that front-end Markdown parser
    can render titles with clickable blue text + action buttons.
    """
    formatted_docs = []
    for i, doc in enumerate(docs):
        metadata = doc.metadata or {}
        doc_id = metadata.get('id', f"doc_{i}")

        # Authors
        authors = metadata.get('authors', "")
        if isinstance(authors, list):
            authors = ", ".join([str(a).strip() for a in authors])

        # Keywords
        keywords = metadata.get('keywords', "")
        if isinstance(keywords, list):
            keywords = ", ".join([str(k).strip() for k in keywords])

        # Construct Markdown string
        doc_string = (
            f"- **Title: {metadata.get('title', '')} [[ID:{doc_id}]]**\n"
            f"  - Authors: {authors}\n"
            f"  - Year: {metadata.get('year', '')}\n"
            f"  - Source: {metadata.get('source', '')}\n"
            f"  - Keywords: {keywords}\n"
            f"  - Score: {metadata.get('_score', 0.0):.4f}\n"
        )
        formatted_docs.append(doc_string)

    return "\n".join(formatted_docs)


@dataclass
class RouterPlan:
    action: str = "semantic_search"  # e.g., author_search, year_search, keyword_search, etc.
    filters: dict = None
    top_k: int = 5

def _safe_json_loads(s: str) -> dict:
    try:
        return json.loads(s)
    except Exception:
        s = re.sub(r"^```(json)?|```$", "", s.strip(), flags=re.MULTILINE)
        try:
            return json.loads(s)
        except Exception:
            return {}



# def serialize_history(request: Dict[str, Any]):
#     chat_history_raw = request.get("chat_history", [])
#     converted_chat_history = []
#     for message in chat_history_raw:
#         if message.get("human"):
#             converted_chat_history.append(HumanMessage(content=message["human"]))
#         if message.get("ai"):
#             converted_chat_history.append(AIMessage(content=message["ai"]))
#     return converted_chat_history

# def serialize_history(request: Dict[str, Any]):
#     chat_history_raw = request.get("chat_history", [])
#     converted_chat_history = []
#     for message in chat_history_raw:
#         if message.get("human") or message.get("role") == "user":
#             converted_chat_history.append(
#                 HumanMessage(content=message.get("human") or message.get("text"))
#             )
#         if message.get("ai") or message.get("role") == "ai":
#             converted_chat_history.append(
#                 AIMessage(content=message.get("ai") or message.get("text"))
#             )
#     return converted_chat_history


def serialize_history(request: Dict[str, Any]):
    chat_history_raw = request.get("chat_history", [])
    print("🔥 serialize_history output:", chat_history_raw)   
    converted_chat_history = []
    for message in chat_history_raw:
        if message.get("human") or message.get("role") == "user":
            converted_chat_history.append(
                HumanMessage(content=message.get("human") or message.get("text"))
            )
        if message.get("ai") or message.get("role") == "ai":
            converted_chat_history.append(
                AIMessage(content=message.get("ai") or message.get("text"))
            )
    print("✅ serialize_history output:", converted_chat_history)
    return converted_chat_history

######################################################################


ROUTER_PROMPT = PromptTemplate.from_template(
    """
You are a planner for a research-paper assistant. Given the user message, produce a JSON plan that chooses the best retrieval strategy.

Return ONLY valid JSON with this schema:
{
  "action": "<one of: author_search | year_search | keyword_search | title_search | source_search | id_lookup | semantic_search | other>",
  "filters": {
      "authors": [string],
      "keywords": [string],
      "sources": [string],
      "title": "",
      "paper_ids": [string],
      "year_min": null,
      "year_max": null,
      "free_text": ""
  },
  "top_k": 5
}

Guidelines:
- If the user mentions an author name(s), prefer "author_search".
- If the user specifies a year or year range, prefer "year_search".
- If they list keywords, prefer "keyword_search".
- If they mention a title, prefer "title_search".
- If they mention a venue/source, prefer "source_search".
- If IDs are given, use "id_lookup".
- Otherwise, use "semantic_search" with free_text.
User message: "{query}"
"""
)

def _metadata_to_documents(items: List[dict]) -> List[Document]:
    docs = []
    for i, m in enumerate(items or []):
        docs.append(
            Document(
                page_content=(m.get("Abstract") or m.get("Title") or ""),
                metadata={
                    "title": m.get("Title", ""),
                    "abstract": m.get("Abstract", ""),
                    "authors": m.get("Authors", []),
                    "keywords": m.get("Keywords", []),
                    "source": m.get("Source", ""),
                    "year": m.get("Year", ""),
                    "id": str(m.get("ID", f"doc_{i}")),
                    "lang": m.get("lang", "unknown"),
                    "_score": float(m.get("score", 0.0)),
                }
            )
        )
    return docs


def _run_metadata_search(plan: RouterPlan) -> List[Document]:
    f = plan.filters or {}
    q = ChromaQuerySchema(
        title=f.get("title") or None,
        abstract=None,
        author=f.get("authors") or None,
        source=f.get("sources") or None,
        keyword=f.get("keywords") or None,
        min_year=f.get("year_min"),
        max_year=f.get("year_max"),
        id_list=f.get("paper_ids") or None,
        offset=0,
        limit=int(plan.top_k or 5),
    )
    items = chroma.query_docs(q)
    return _metadata_to_documents(items)


def _run_semantic_search(query_text: str, embedding_type: str, lang: str, top_k: int) -> List[Document]:
    retriever = get_retriever(embedding_type=embedding_type, lang=lang)
    return retriever._get_relevant_documents(query=query_text, embedding_type=embedding_type, lang=lang)[:top_k]

###########################################################################


# def create_retriever_chain(llm: LanguageModelLike) -> Runnable:
#     condense_question_prompt = PromptTemplate.from_template(REPHRASE_TEMPLATE)
#     condense_question_chain = (condense_question_prompt | llm | StrOutputParser()).with_config(run_name="CondenseQuestion")

#     def get_dynamic_retriever(inputs: Dict[str, Any]) -> BaseRetriever:
#         embedding_type = inputs.get("embedding_type", "specter")
#         lang = inputs.get("lang", None)
#         return get_retriever(embedding_type=embedding_type, lang=lang)

#     return RunnableBranch(
#         (
#             RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(run_name="HasChatHistoryCheck"),
#             (
#                 RunnablePassthrough.assign(
#                     _retriever_params=RunnableLambda(lambda x: {"embedding_type": x.get("embedding_type"), "lang": x.get("lang")})
#                 )
#                 | RunnablePassthrough.assign(
#                     rephrased_question=condense_question_chain
#                 )
#                 | RunnableLambda(
#                     lambda x: get_dynamic_retriever(x["_retriever_params"])._get_relevant_documents(
#                         x["rephrased_question"], 
#                         embedding_type=x["_retriever_params"]["embedding_type"],
#                         lang=x["_retriever_params"]["lang"]
#                     )
#                 )
#             ).with_config(run_name="RetrievalChainWithHistory"),
#         ),
#         (
#             RunnablePassthrough.assign(
#                 _retriever_params=RunnableLambda(lambda x: {"embedding_type": x.get("embedding_type"), "lang": x.get("lang")})
#             )
#             | RunnableLambda(
#                 lambda x: get_dynamic_retriever(x["_retriever_params"])._get_relevant_documents(
#                     x["question"], 
#                     embedding_type=x["_retriever_params"]["embedding_type"],
#                     lang=x["_retriever_params"]["lang"]
#                 )
#             )
#         ).with_config(run_name="RetrievalChainWithNoHistory"),
#     ).with_config(run_name="RouteDependingOnChatHistory")



def create_retriever_chain(llm: LanguageModelLike) -> Runnable:
    """
    Reason→Act retriever chain with multi-turn support:
    - If chat history exists, first condense the question.
    - Then use Router (LLM) to decide whether to run metadata filtering or semantic search.
    """
    condense_question_prompt = PromptTemplate.from_template(REPHRASE_TEMPLATE)
    condense_question_chain = (condense_question_prompt | llm | StrOutputParser()).with_config(run_name="CondenseQuestion")

    router_chain = (ROUTER_PROMPT | llm | StrOutputParser()).with_config(run_name="RouterPlan")

    def run_reason_act(question: str, embedding_type: str, lang: str) -> List[Document]:
        raw = router_chain.invoke({"query": question})
        plan_dict = _safe_json_loads(raw)
        plan = RouterPlan(
            action=(plan_dict.get("action") or "semantic_search"),
            filters=(plan_dict.get("filters") or {}),
            top_k=int(plan_dict.get("top_k") or 5)
        )

        if plan.action in {"author_search", "year_search", "keyword_search", "title_search", "source_search", "id_lookup"}:
            docs = _run_metadata_search(plan)
        elif plan.action == "semantic_search":
            free_text = (plan.filters or {}).get("free_text") or question
            docs = _run_semantic_search(free_text, embedding_type, lang, plan.top_k)
        else:
            docs = _run_semantic_search(question, embedding_type, lang, plan.top_k)

        return docs

    return RunnableBranch(
        # Case 1: has chat history → condense question first
        (
            RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(run_name="HasChatHistoryCheck"),
            (
                RunnablePassthrough.assign(
                    rephrased_question=condense_question_chain
                )
                | RunnableLambda(
                    lambda x: run_reason_act(
                        x["rephrased_question"],
                        embedding_type=x.get("embedding_type", "specter"),
                        lang=x.get("lang")
                    )
                )
            ).with_config(run_name="RetrievalChainWithHistory"),
        ),
        # Case 2: no chat history → direct Reason→Act
        (
            RunnableLambda(
                lambda x: run_reason_act(
                    x["question"],
                    embedding_type=x.get("embedding_type", "specter"),
                    lang=x.get("lang")
                )
            )
        ).with_config(run_name="RetrievalChainWithNoHistory"),
    ).with_config(run_name="ReasonActRetrieverChain")



    


def create_chain(llm: LanguageModelLike) -> Runnable:
    retriever_chain = create_retriever_chain(llm).with_config(run_name="FindDocs")
    
    context = (
        RunnablePassthrough.assign(docs=retriever_chain)
        .assign(context=RunnableLambda(lambda x: logging.info(f"\U0001F4DA Retrieved Docs:\n {format_docs(x['docs'])}") or format_docs(x["docs"])))
        .with_config(run_name="RetrieveDocs")
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", RESPONSE_TEMPLATE),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}\n\nPlease answer the question based on the following documents:\n{context}"),
    ])
    default_response_synthesizer = prompt | llm | StrOutputParser()
    
    return (
        RunnablePassthrough.assign(
            chat_history=RunnableLambda(
                lambda x: serialize_history({"chat_history": x.get("chat_history_raw", [])})
            )
        )
        | context
        | default_response_synthesizer
    )

# def create_chain(llm: LanguageModelLike) -> Runnable:
#     retriever_chain = create_retriever_chain(llm).with_config(run_name="FindDocs")
    
#     context = (
#         RunnablePassthrough.assign(docs=retriever_chain)
#         .assign(context=RunnableLambda(lambda x: logging.info(f"📚 Retrieved Docs:\n {format_docs(x['docs'])}") or format_docs(x["docs"])))
#         .with_config(run_name="RetrieveDocs")
#     )
    
#     prompt = ChatPromptTemplate.from_messages([
#         ("system", RESPONSE_TEMPLATE),
#         MessagesPlaceholder(variable_name="chat_history"),
#         ("human", "{question}\n\nPlease answer based only on {context}")
#     ])
#     default_response_synthesizer = prompt | llm | StrOutputParser()
    
#     def attach_titles(x):
#         llm_answer = x["output"]
#         titles = format_docs(x["docs"])
#         return llm_answer + "\n\nRetrieved papers:\n" + titles

#     return (
#         RunnablePassthrough.assign(chat_history=serialize_history)
#         | context
#         | RunnablePassthrough.assign(output=default_response_synthesizer)
#         | RunnableLambda(attach_titles)
#     )







from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models import LanguageModelLike

def create_plain_chain(llm: LanguageModelLike):
    """
    Create a simple chat chain without retrieval.
    """
    plain_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Answer clearly and concisely."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

    return (
        RunnablePassthrough.assign(
            chat_history=RunnableLambda(
                lambda x: serialize_history({"chat_history": x.get("chat_history_raw", [])})
            )
        )
        | plain_prompt
        | llm
        | StrOutputParser()
    )


# llm = ChatOllama(model="mistral", temperature=0, streaming=True)
load_dotenv()

llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    streaming=True   # enable streaming
)

answer_chain = create_chain(llm)         ## RAG
plain_chain = create_plain_chain(llm)    ## noraml chat


# def chat_streaming_output(question: str, chat_history: List[Dict[str, str]], embedding_type: str = "specter", lang: str = None):
#     """
#     Main chat function supporting custom embedding and language.
#     Args:
#         question: The user's question.
#         chat_history: Previous chat history.
#         embedding_type: Embedding type to use ("specter", "ada", "glove").
#         lang: Language code (e.g., "en", "zh").
#     """
#     return answer_chain.stream({
#         'question': question,
#         'chat_history': chat_history,
#         'embedding_type': embedding_type,
#         'lang': lang
#     })


def chat_streaming_output(
    question: str,
    chat_history: List[Dict[str, str]],
    embedding_type: str = "specter",
    lang: str = None,
    mode: str = "rag"
):
    """
    Main chat function supporting both normal and RAG modes.
    Args:
        question: The user's question.
        chat_history: Previous chat history.
        embedding_type: Embedding type to use ("specter", "ada", "glove").
        lang: Language code (e.g., "en", "zh").
        mode: "normal" for plain LLM chat, "rag" for retrieval-augmented chat
    """
    if mode == "normal":
     
        # normal chat, do not use retrieval
        return plain_chain.stream({
            "question": question,
            "chat_history_raw": chat_history
        })
    else:
        # default to RAG mode
        return answer_chain.stream({
            "question": question,
            "chat_history_raw": chat_history,
            "embedding_type": embedding_type,
            "lang": lang
        })


def summarize_output(prompt_data: Dict[str, str]):
    return (i.content for i in llm.stream(SUMMARIZE_PROMPT.format(**prompt_data)))


def literature_review_output(prompt_data: Dict[str, str]):
    return (i.content for i in llm.stream(LITERATURE_REVIEW_PROMPT.format(**prompt_data)))


def format_papers_in_prompt(papers: List[Dict[str, Any]]) -> str:
    """
    Format a list of papers into a string suitable for LLM prompt input.
    """
    return '\n'.join([
        f" --- \nTitle: {p.get('Title', '')}\nAuthors: {', '.join(p.get('Authors', []))}\nAbstract: {p.get('Abstract', '')}\nSource: {p.get('Source', '')}\nYear: {p.get('Year', '')}\nKeywords: {', '.join(p.get('Keywords', []))}\n ---"
        for p in papers
    ])















# import json
# from operator import itemgetter
# from os import path
# from typing import Sequence, Dict, List, Any
# from dotenv import load_dotenv
# from langchain_core.documents import Document
# from langchain_core.language_models import LanguageModelLike
# from langchain_core.messages import HumanMessage, AIMessage
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.retrievers import BaseRetriever
# from langchain_core.runnables import Runnable, RunnableBranch, RunnableLambda, RunnablePassthrough
# # from langchain_ollama import ChatOllama
# from langchain_core.embeddings import Embeddings 
# from service.embed_chroma import LocalSpecterEmbedding, LocalSentenceTransformerEmbedding 
# import chromadb  
# from pydantic import Field, model_validator 
# import config
# from prompt import RESPONSE_TEMPLATE, LITERATURE_REVIEW_PROMPT, SUMMARIZE_PROMPT, REPHRASE_TEMPLATE
# import numpy as np 
# import logging 
# import os
# from dotenv import load_dotenv
# from langchain_openai import AzureChatOpenAI


# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# load_dotenv(path.join(config.PROJ_ROOT_DIR, '.env'))


# # === Chroma Retriever (enhanced localization support) ===
# # This class remains unchanged since it was already designed to support dynamic embedding selection.
# class LocalizedChromaRetriever(BaseRetriever):
#     client: Any = Field(...)
#     collection_mapping: Dict[str, str] = Field(...) 
#     embedding_models: Dict[str, Embeddings] = Field(...) 
#     k: int = Field(default=5)

#     @model_validator(mode='after')
#     def check_embedding_models(self):
#         required_models = set(self.collection_mapping.keys())
#         for model_type in required_models:
#             if model_type not in self.embedding_models:
#                 raise ValueError(f"Missing embedding model for type: {model_type}")
#         return self

#     def _get_relevant_documents(self, query: str, *, run_manager=None, 
#                                 embedding_type: str = "specter", lang: str = None) -> List[Document]:
#         if embedding_type not in self.embedding_models:
#             logging.error(f"Unsupported embedding type '{embedding_type}'. Available: {list(self.embedding_models.keys())}")
#             return []
        
#         embedding_model = self.embedding_models[embedding_type]
#         collection_name = self.collection_mapping.get(embedding_type)

#         if not collection_name:
#             logging.error(f"No collection name mapped for embedding type: {embedding_type}")
#             return []

#         if self.client is None:
#             logging.error("ChromaDB client is not initialized.")
#             return []

#         try:
#             collection = self.client.get_collection(collection_name)
#         except Exception as e:
#             logging.error(f"Failed to get ChromaDB collection '{collection_name}': {e}")
#             return []

#         query_vector = embedding_model.embed_query(query)
        
#         where_clause = {}
#         if lang and lang.lower() != "all": 
#             where_clause["lang"] = lang.lower()
#             logging.info(f"Applying language filter: {where_clause} for collection '{collection_name}'")
#         else:
#             logging.info(f"No specific language filter applied for collection '{collection_name}'. Performing cross-lingual search.")


#         results = collection.query(
#             query_embeddings=[query_vector],
#             n_results=self.k,
#             include=["metadatas", "distances"],
#             where=where_clause if where_clause else None
#         )

#         metadatas = results['metadatas'][0]
#         distances = results['distances'][0]

#         docs = []
#         for i, (m, d) in enumerate(zip(metadatas, distances)):
#             lowercase_m = {k.lower(): v for k, v in m.items()}
            
#             doc_id = lowercase_m.get("original_id", lowercase_m.get("id", f"doc_{i}")) 
            
#             similarity_score = max(0.0, min(1.0, 1 / (1 + d)))

#             docs.append(
#                 Document(
#                     page_content=lowercase_m.get("abstract", "") or lowercase_m.get("title", ""),
#                     metadata={
#                         "title": lowercase_m.get("title", ""),
#                         "abstract": lowercase_m.get("abstract", ""),
#                         "authors": lowercase_m.get("authors", []),
#                         "keywords": lowercase_m.get("keywords", []),
#                         "source": lowercase_m.get("source", ""),
#                         "year": lowercase_m.get("year", ""),
#                         "id": doc_id,
#                         "lang": lowercase_m.get("lang", "unknown"),
#                         "_score": similarity_score
#                     }
#                 )
#             )
#         return docs

#     async def _aget_relevant_documents(self, query: str, *, run_manager=None, 
#                                      embedding_type: str = "specter", lang: str = None) -> List[Document]:
#         return self._get_relevant_documents(query, run_manager=run_manager, embedding_type=embedding_type, lang=lang)


# def get_retriever(embedding_type: str = "specter", lang: str = None) -> BaseRetriever:
#     """
#     Get a configured retriever instance based on embedding type and language.
    
#     Args:
#         embedding_type: A string indicating the embedding type ("specter", "ada", "glove").
#         lang: Optional language code (e.g., "en", "zh").
#     Returns:
#         A BaseRetriever instance.
#     """
#     client = chromadb.PersistentClient(path="chroma_db")
#     logging.info(f"🔍 Using ChromaDB client at path: chroma_db. Collections: {client.list_collections()}")

#     # Instantiate all required embedding models
#     embedding_models_instances = {
#         # Specter embedding model using local class
#         "specter": LocalSpecterEmbedding(model_name="allenai/specter"), 
#         # Ada embedding model using local class
#         "ada": LocalSentenceTransformerEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2"),
#         # GloVe embedding model using local class
#         "glove": LocalSentenceTransformerEmbedding(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") 
#     }

#     # ChromaDB collection mapping (unchanged)
#     chroma_collection_mapping = {
#         "specter": "paper_specter",
#         "ada": "paper_ada_localized",
#         "glove": "paper_glove_localized"
#     }

#     if embedding_type not in chroma_collection_mapping:
#         logging.error(f"Requested embedding type '{embedding_type}' is not supported in retriever configuration. Falling back to 'specter'.")
#         embedding_type = "specter"

#     retriever_instance = LocalizedChromaRetriever(
#         client=client,
#         collection_mapping=chroma_collection_mapping,
#         embedding_models=embedding_models_instances,
#         k=5
#     )
#     logging.info(f"✅ Created LocalizedChromaRetriever with embedding type '{embedding_type}' and language '{lang}'.")
#     return retriever_instance


# # def format_docs(docs: Sequence[Document]) -> str:
# #     formatted_docs = []
# #     for i, doc in enumerate(docs):
# #         metadata = doc.metadata or {}
# #         doc_id = metadata.get('id', f"doc_{i}")
# #         doc_string = (
# #             f"<doc id='{doc_id}' score='{metadata.get('_score', 0.0):.4f}' lang='{metadata.get('lang', 'unknown')}'>"
# #             # f"<title>{metadata.get('title', '')}</title>"
# #             f"<title>{metadata.get('title', '')} [[ID:{doc_id}]]</title>"  # Include ID in title for clarity
# #             f"<abstract>{metadata.get('abstract', '')}</abstract>"
# #             f"<authors>{', '.join(metadata.get('authors', []))}</authors>"
# #             f"<keywords>{', '.join(metadata.get('keywords', []))}</keywords>"
# #             f"<source>{metadata.get('source', '')}</source>"
# #             f"<year>{metadata.get('year', '')}</year>"
# #             f"</doc>"
# #         )
# #         formatted_docs.append(doc_string)
# #     return "\n".join(formatted_docs)

# # def format_docs(docs: Sequence[Document]) -> str:
# #     formatted_docs = []
# #     for i, doc in enumerate(docs):
# #         metadata = doc.metadata or {}
# #         doc_id = metadata.get('id', f"doc_{i}")

# #         # Handle authors and keywords which might be lists
# #         authors = metadata.get('authors', "")
# #         if isinstance(authors, list):
# #             authors = ", ".join([str(a).strip() for a in authors])

# #         keywords = metadata.get('keywords', "")
# #         if isinstance(keywords, list):
# #             keywords = ", ".join([str(k).strip() for k in keywords])

# #         doc_string = (
# #             f"<doc id='{doc_id}' score='{metadata.get('_score', 0.0):.4f}' lang='{metadata.get('lang', 'unknown')}'>"
# #             f"<title>{metadata.get('title', '')} [[ID:{doc_id}]]</title>"
# #             f"<abstract>{metadata.get('abstract', '')}</abstract>"
# #             f"<authors>{authors}</authors>"
# #             f"<keywords>{keywords}</keywords>"
# #             f"<source>{metadata.get('source', '')}</source>"
# #             f"<year>{metadata.get('year', '')}</year>"
# #             f"</doc>"
# #         )
# #         formatted_docs.append(doc_string)
# #     return "\n".join(formatted_docs)


# def format_docs(docs: Sequence[Document]) -> str:
#     """
#     Format docs into Markdown list items so that front-end Markdown parser
#     can render titles with clickable blue text + action buttons.
#     """
#     formatted_docs = []
#     for i, doc in enumerate(docs):
#         metadata = doc.metadata or {}
#         doc_id = metadata.get('id', f"doc_{i}")

#         # Authors
#         authors = metadata.get('authors', "")
#         if isinstance(authors, list):
#             authors = ", ".join([str(a).strip() for a in authors])

#         # Keywords
#         keywords = metadata.get('keywords', "")
#         if isinstance(keywords, list):
#             keywords = ", ".join([str(k).strip() for k in keywords])

#         # Construct Markdown string
#         doc_string = (
#             f"- **Title: {metadata.get('title', '')} [[ID:{doc_id}]]**\n"
#             f"  - Authors: {authors}\n"
#             f"  - Year: {metadata.get('year', '')}\n"
#             f"  - Source: {metadata.get('source', '')}\n"
#             f"  - Keywords: {keywords}\n"
#             f"  - Score: {metadata.get('_score', 0.0):.4f}\n"
#         )
#         formatted_docs.append(doc_string)

#     return "\n".join(formatted_docs)



# # def serialize_history(request: Dict[str, Any]):
# #     chat_history_raw = request.get("chat_history", [])
# #     converted_chat_history = []
# #     for message in chat_history_raw:
# #         if message.get("human"):
# #             converted_chat_history.append(HumanMessage(content=message["human"]))
# #         if message.get("ai"):
# #             converted_chat_history.append(AIMessage(content=message["ai"]))
# #     return converted_chat_history

# # def serialize_history(request: Dict[str, Any]):
# #     chat_history_raw = request.get("chat_history", [])
# #     converted_chat_history = []
# #     for message in chat_history_raw:
# #         if message.get("human") or message.get("role") == "user":
# #             converted_chat_history.append(
# #                 HumanMessage(content=message.get("human") or message.get("text"))
# #             )
# #         if message.get("ai") or message.get("role") == "ai":
# #             converted_chat_history.append(
# #                 AIMessage(content=message.get("ai") or message.get("text"))
# #             )
# #     return converted_chat_history


# def serialize_history(request: Dict[str, Any]):
#     chat_history_raw = request.get("chat_history", [])
#     print("🔥 serialize_history output:", chat_history_raw)   
#     converted_chat_history = []
#     for message in chat_history_raw:
#         if message.get("human") or message.get("role") == "user":
#             converted_chat_history.append(
#                 HumanMessage(content=message.get("human") or message.get("text"))
#             )
#         if message.get("ai") or message.get("role") == "ai":
#             converted_chat_history.append(
#                 AIMessage(content=message.get("ai") or message.get("text"))
#             )
#     print("✅ serialize_history output:", converted_chat_history)
#     return converted_chat_history


# def create_retriever_chain(llm: LanguageModelLike) -> Runnable:
#     condense_question_prompt = PromptTemplate.from_template(REPHRASE_TEMPLATE)
#     condense_question_chain = (condense_question_prompt | llm | StrOutputParser()).with_config(run_name="CondenseQuestion")

#     def get_dynamic_retriever(inputs: Dict[str, Any]) -> BaseRetriever:
#         embedding_type = inputs.get("embedding_type", "specter")
#         lang = inputs.get("lang", None)
#         return get_retriever(embedding_type=embedding_type, lang=lang)

#     return RunnableBranch(
#         (
#             RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(run_name="HasChatHistoryCheck"),
#             (
#                 RunnablePassthrough.assign(
#                     _retriever_params=RunnableLambda(lambda x: {"embedding_type": x.get("embedding_type"), "lang": x.get("lang")})
#                 )
#                 | RunnablePassthrough.assign(
#                     rephrased_question=condense_question_chain
#                 )
#                 | RunnableLambda(
#                     lambda x: get_dynamic_retriever(x["_retriever_params"])._get_relevant_documents(
#                         x["rephrased_question"], 
#                         embedding_type=x["_retriever_params"]["embedding_type"],
#                         lang=x["_retriever_params"]["lang"]
#                     )
#                 )
#             ).with_config(run_name="RetrievalChainWithHistory"),
#         ),
#         (
#             RunnablePassthrough.assign(
#                 _retriever_params=RunnableLambda(lambda x: {"embedding_type": x.get("embedding_type"), "lang": x.get("lang")})
#             )
#             | RunnableLambda(
#                 lambda x: get_dynamic_retriever(x["_retriever_params"])._get_relevant_documents(
#                     x["question"], 
#                     embedding_type=x["_retriever_params"]["embedding_type"],
#                     lang=x["_retriever_params"]["lang"]
#                 )
#             )
#         ).with_config(run_name="RetrievalChainWithNoHistory"),
#     ).with_config(run_name="RouteDependingOnChatHistory")


# def create_chain(llm: LanguageModelLike) -> Runnable:
#     retriever_chain = create_retriever_chain(llm).with_config(run_name="FindDocs")
    
#     context = (
#         RunnablePassthrough.assign(docs=retriever_chain)
#         .assign(context=RunnableLambda(lambda x: logging.info(f"\U0001F4DA Retrieved Docs:\n {format_docs(x['docs'])}") or format_docs(x["docs"])))
#         .with_config(run_name="RetrieveDocs")
#     )
    
#     prompt = ChatPromptTemplate.from_messages([
#         ("system", RESPONSE_TEMPLATE),
#         MessagesPlaceholder(variable_name="chat_history"),
#         ("human", "{question}\n\nPlease answer the question based on the following documents:\n{context}"),
#     ])
#     default_response_synthesizer = prompt | llm | StrOutputParser()
    
#     return (
#         RunnablePassthrough.assign(
#             chat_history=RunnableLambda(
#                 lambda x: serialize_history({"chat_history": x.get("chat_history_raw", [])})
#             )
#         )
#         | context
#         | default_response_synthesizer
#     )

# # def create_chain(llm: LanguageModelLike) -> Runnable:
# #     retriever_chain = create_retriever_chain(llm).with_config(run_name="FindDocs")
    
# #     context = (
# #         RunnablePassthrough.assign(docs=retriever_chain)
# #         .assign(context=RunnableLambda(lambda x: logging.info(f"📚 Retrieved Docs:\n {format_docs(x['docs'])}") or format_docs(x["docs"])))
# #         .with_config(run_name="RetrieveDocs")
# #     )
    
# #     prompt = ChatPromptTemplate.from_messages([
# #         ("system", RESPONSE_TEMPLATE),
# #         MessagesPlaceholder(variable_name="chat_history"),
# #         ("human", "{question}\n\nPlease answer based only on {context}")
# #     ])
# #     default_response_synthesizer = prompt | llm | StrOutputParser()
    
# #     def attach_titles(x):
# #         llm_answer = x["output"]
# #         titles = format_docs(x["docs"])
# #         return llm_answer + "\n\nRetrieved papers:\n" + titles

# #     return (
# #         RunnablePassthrough.assign(chat_history=serialize_history)
# #         | context
# #         | RunnablePassthrough.assign(output=default_response_synthesizer)
# #         | RunnableLambda(attach_titles)
# #     )







# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.language_models import LanguageModelLike

# def create_plain_chain(llm: LanguageModelLike):
#     """
#     Create a simple chat chain without retrieval.
#     """
#     plain_prompt = ChatPromptTemplate.from_messages([
#         ("system", "You are a helpful AI assistant. Answer clearly and concisely."),
#         MessagesPlaceholder(variable_name="chat_history"),
#         ("human", "{question}")
#     ])

#     return (
#         RunnablePassthrough.assign(
#             chat_history=RunnableLambda(
#                 lambda x: serialize_history({"chat_history": x.get("chat_history_raw", [])})
#             )
#         )
#         | plain_prompt
#         | llm
#         | StrOutputParser()
#     )


# # llm = ChatOllama(model="mistral", temperature=0, streaming=True)
# load_dotenv()

# llm = AzureChatOpenAI(
#     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
#     azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
#     api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
#     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
#     streaming=True   # enable streaming
# )

# answer_chain = create_chain(llm)         ## RAG
# plain_chain = create_plain_chain(llm)    ## noraml chat


# # def chat_streaming_output(question: str, chat_history: List[Dict[str, str]], embedding_type: str = "specter", lang: str = None):
# #     """
# #     Main chat function supporting custom embedding and language.
# #     Args:
# #         question: The user's question.
# #         chat_history: Previous chat history.
# #         embedding_type: Embedding type to use ("specter", "ada", "glove").
# #         lang: Language code (e.g., "en", "zh").
# #     """
# #     return answer_chain.stream({
# #         'question': question,
# #         'chat_history': chat_history,
# #         'embedding_type': embedding_type,
# #         'lang': lang
# #     })


# def chat_streaming_output(
#     question: str,
#     chat_history: List[Dict[str, str]],
#     embedding_type: str = "specter",
#     lang: str = None,
#     mode: str = "rag"
# ):
#     """
#     Main chat function supporting both normal and RAG modes.
#     Args:
#         question: The user's question.
#         chat_history: Previous chat history.
#         embedding_type: Embedding type to use ("specter", "ada", "glove").
#         lang: Language code (e.g., "en", "zh").
#         mode: "normal" for plain LLM chat, "rag" for retrieval-augmented chat
#     """
#     if mode == "normal":
     
#         # normal chat, do not use retrieval
#         return plain_chain.stream({
#             "question": question,
#             "chat_history_raw": chat_history
#         })
#     else:
#         # default to RAG mode
#         return answer_chain.stream({
#             "question": question,
#             "chat_history_raw": chat_history,
#             "embedding_type": embedding_type,
#             "lang": lang
#         })


# def summarize_output(prompt_data: Dict[str, str]):
#     return (i.content for i in llm.stream(SUMMARIZE_PROMPT.format(**prompt_data)))


# def literature_review_output(prompt_data: Dict[str, str]):
#     return (i.content for i in llm.stream(LITERATURE_REVIEW_PROMPT.format(**prompt_data)))


# def format_papers_in_prompt(papers: List[Dict[str, Any]]) -> str:
#     """
#     Format a list of papers into a string suitable for LLM prompt input.
#     """
#     return '\n'.join([
#         f" --- \nTitle: {p.get('Title', '')}\nAuthors: {', '.join(p.get('Authors', []))}\nAbstract: {p.get('Abstract', '')}\nSource: {p.get('Source', '')}\nYear: {p.get('Year', '')}\nKeywords: {', '.join(p.get('Keywords', []))}\n ---"
#         for p in papers
#     ])