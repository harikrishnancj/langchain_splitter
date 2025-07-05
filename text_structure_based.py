from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

loader=TextLoader("data.txt",encoding="utf-8")

text=loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=1,
)

res=splitter.split_documents(text)

print(res[5])