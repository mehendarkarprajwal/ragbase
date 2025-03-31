import re
from operator import itemgetter
from typing import List

from langchain.schema.runnable import RunnablePassthrough
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tracers.stdout import ConsoleCallbackHandler
from langchain_core.vectorstores import VectorStoreRetriever

from ragbase.config import Config
from ragbase.session_history import get_session_history

SYSTEM_PROMPT = """ 
Objective:  
You are an AI assistant designed to retrieve and summarize information based on the provided contextual data. Your goal is to generate accurate, concise, and well-structured responses by leveraging the retrieved information while clearly indicating any limitations in the available context.  

---

 Instructions for Response Generation:  

1. Use Only Provided Context:  
   - Generate responses exclusively based on the given contextual information.  
   - Do not include external knowledge or assumptions beyond the provided sources.  

2. Handle Missing Information Gracefully:  
   - If the context does not contain the required answer, explicitly state:  
     "The answer cannot be found in the provided context."  
   - Avoid speculation or fabricated information.  

3. Prioritize Conciseness and Clarity:  
   - Limit responses to a maximum of three sentences unless a more detailed response is explicitly required.  
   - Use bullet points or numbered lists for better readability when applicable.  

4. Contextual Relevance and Formatting:  
   - The provided context is organized by relevance, with the most pertinent information appearing first.  
   - Different sources are separated by a horizontal rule (`---`), and each source should be treated as a standalone reference.  
   - Summarize and synthesize information from multiple sources when applicable, avoiding redundancy.  

5. Maintain a Professional and Neutral Tone:  
   - Ensure responses are factual, neutral, and free from personal opinions.  
   - If a user’s query is unclear, request clarification rather than making assumptions.  

The contextual information is organized with the most relevant source appearing first.
Each source is separated by a horizontal rule (---).

Context:
{context}

Use markdown formatting where appropriate.

"""


def remove_links(text: str) -> str:
    url_pattern = r"https?://\S+|www\.\S+"
    return re.sub(url_pattern, "", text)


def format_documents(documents: List[Document]) -> str:
    texts = []
    for doc in documents:
        texts.append(doc.page_content)
        texts.append("---")

    return remove_links("\n".join(texts))


def create_chain(llm: BaseLanguageModel, retriever: VectorStoreRetriever) -> Runnable:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}"),
        ]
    )

    chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question")
            | retriever.with_config({"run_name": "context_retriever"})
            | format_documents
        )
        | prompt
        | llm
    )

    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    ).with_config({"run_name": "chain_answer"})


async def ask_question(chain: Runnable, question: str, session_id: str):
    async for event in chain.astream_events(
        {"question": question},
        config={
            "callbacks": [ConsoleCallbackHandler()] if Config.DEBUG else [],
            "configurable": {"session_id": session_id},
        },
        version="v2",
        include_names=["context_retriever", "chain_answer"],
    ):
        event_type = event["event"]
        if event_type == "on_retriever_end":
            yield event["data"]["output"]
        if event_type == "on_chain_stream":
            yield event["data"]["chunk"].content
