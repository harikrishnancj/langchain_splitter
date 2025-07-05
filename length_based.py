from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
load_dotenv()

'''llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

prompt1=PromptTemplate(
    template="summarie the {text}",
    input_variables=['text']
)'''

loader=TextLoader("data.txt",encoding="utf-8")

text=loader.load()

splitter=CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=1
)

res=splitter.split_documents(text)

print(res[5].page_content)