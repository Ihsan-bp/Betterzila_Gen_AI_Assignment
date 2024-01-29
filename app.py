from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.llms import Anyscale
from langchain.vectorstores import Chroma
from langchain.chains import StuffDocumentsChain, LLMChain
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate
from langchain.messaging import HumanMessage, AIMessage
from langchain.trace import trace_as_chain_group
import gradio as gr

# Set up retriever
embeddings = GPT4AllEmbeddings()
vectorstore = Chroma(embedding_function=embeddings, persist_directory="chroma")
retriever = vectorstore.as_retriever()

# Set up chains for answering questions based on documents
document_prompt = PromptTemplate(input_variables=["page_content"], template="{page_content}")
document_variable_name = "context"

ANYSCALE_API_BASE = "https://api.endpoints.anyscale.com/v1"
ANYSCALE_API_KEY = "esecret_pmx42up5fwnqdlwrffrgpvin5u"
ANYSCALE_MODEL_NAME = "meta-llama/Llama-2-70b-chat-hf"
os.environ["ANYSCALE_API_BASE"] = ANYSCALE_API_BASE
os.environ["ANYSCALE_API_KEY"] = ANYSCALE_API_KEY
llm = Anyscale(model_name=ANYSCALE_MODEL_NAME, temperature=0)

prompt_template = """Use the following pieces of context to answer user questions. If you don't know the answer, just say that you don't know, don't try to make up an answer.

--------------

{context}"""
system_prompt = SystemMessagePromptTemplate.from_template(prompt_template)
prompt = ChatPromptTemplate(messages=[system_prompt, MessagesPlaceholder(variable_name="chat_history"), HumanMessagePromptTemplate.from_template("{question}")])

llm_chain = LLMChain(llm=llm, prompt=prompt)
combine_docs_chain = StuffDocumentsChain(llm_chain=llm_chain, document_prompt=document_prompt, document_variable_name=document_variable_name, document_separator="---------")

# Set up a chain for generating search queries
template = """Combine the chat history and follow-up question into a search query.

Chat History:

{chat_history}

Follow-up question: {question}
"""
prompt = PromptTemplate.from_template(template)
question_generator_chain = LLMChain(llm=llm, prompt=prompt)

# Function to handle question-answering
def qa_response(message, history):
    convo_string = "\n\n".join([f"Human: {h}\nAssistant: {a}" for h, a in history])

    messages = []
    for human, ai in history:
        messages.append(HumanMessage(content=human))
        messages.append(AIMessage(content=ai))

    with trace_as_chain_group("qa_response") as group_manager:
        search_query = question_generator_chain.run(question=message, chat_history=convo_string, callbacks=group_manager)
        docs = retriever.get_relevant_documents(search_query, callbacks=group_manager)
        return combine_docs_chain.run(input_documents=docs, chat_history=messages, question=message, callbacks=group_manager)

# Launch the app
gr.ChatInterface(qa_response).launch()
