from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter,Language
from dotenv import load_dotenv

load_dotenv()

loader=TextLoader("py.txt",encoding="utf-8")

text=loader.load()

split=RecursiveCharacterTextSplitter.from_language(
    chunk_size=200,
    language=Language.PYTHON
)

res=split.split_documents(text)


print(res[1])