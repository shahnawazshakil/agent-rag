from multiprocessing import context
from operator import itemgetter
import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI()

vectorstore = PineconeVectorStore(index_name=os.environ["INDEX_NAME"],embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k":3})

prompt_template = ChatPromptTemplate.from_template(
    """ Answer the question based only on the following context:
    {context}
    Question:{question}
    Provide a detailed answer:"""
)

def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

# ==============================================================
# IMPLEMENTATION 1: Without LCEL (Simple Function-Based Approach)
# ==================================================================

def retrival_chain_without_lcel(query:str):
    """
    Simple retrieval chain without LCEL
    Manually retrieves documents, formats them and generates a response.

    Limitations:
    - Manual step by step execution
    - No built in streaming support
    - No async support without additional code
    - Harder to compose with other chains
    - More verbose and error prone
    """
    # Step 1: Retrieve relevant documents
    docs = retriever.invoke(query)

    #Step 2: Format documents into context String
    context = format_docs(docs)

    #Step 3: Format the prompt with format and query
    messages = prompt_template.format_messages(context=context,question=query)

    #Step 4: Invoke the llm with the list of messages
    response = llm.invoke(messages)

    #Step 5: return the content
    return response.content

def create_retrival_chain_with_lcel():
    """
    Create a retrieval chain using LCEL (Langchain Expression Language).
    Returns a chain that can be invoked with {"question":"..."}

    Advantages over non-LCEL approach
    - Declarations and Composable: Easy to chain operations with pipe operator
    - Built in streaming: chain.stream() works out of the box
    - Built-in async: chain.ainvoke() and chain.astream() available
    - Batch processing: chain.batch() for multiple inputs
    - Type safety: Better integration with langchain type system.
    - Less code: More concise and readable
    - Reusable: Chain can be saved, shared and composed with other chains
    - Better debugging: Langchain provides better observability tools
    """
    retrieval_chain = (
        RunnablePassthrough.assign(
            context=itemgetter("question") | retriever | format_docs
            )
        | prompt_template | llm | StrOutputParser()
    )
    return retrieval_chain


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("IMPLEMENTATION WITHOUT LCEL.")
    print("\n" + "=" * 70)
    query = "what is pinecone in machine learning?"
    result_without_lcel = retrival_chain_without_lcel(query)
    print("\nAnswer:")
    print(result_without_lcel)

    print("\n" + "=" * 70)
    print("IMPLEMENTATION WITH LCEL.")
    print("\n" + "=" * 70)
    query = "what is pinecone in machine learning?"
    chain_with_lcel = create_retrival_chain_with_lcel()
    result_with_lcel = chain_with_lcel.invoke({"question":query})
    print("\nAnswer:")
    print(result_with_lcel)




