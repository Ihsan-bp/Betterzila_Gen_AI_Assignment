from langchain_community.embeddings import GPT4AllEmbeddings
# from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("48lawsofpower.pdf")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=600, chunk_overlap=50)
documents = text_splitter.split_documents(documents)

embeddings = GPT4AllEmbeddings()
vectorstore = Chroma.from_documents(documents, embeddings, persist_directory="chroma")
