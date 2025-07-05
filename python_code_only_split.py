from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import PythonCodeTextSplitter,Language
from dotenv import load_dotenv
from langchain.schema.runnable import RunnablePassthrough,RunnableParallel 
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model=ChatHuggingFace(llm=llm)

prompt1=PromptTemplate(
    template="summarie the {code}",
    input_variables=['code']
)

prompt2 = PromptTemplate(
    template="Merge the following code and explanation:\n\nCode:\n{code}\n\nExplanation:\n{explanation}",
    input_variables=['code', 'explanation']
)

loader=TextLoader("data.py",encoding="utf-8")

text=loader.load()

split=PythonCodeTextSplitter(
    chunk_size=100,
    chunk_overlap=1
)

chunk=split.split_documents(text)

parser=StrOutputParser()

par=RunnableParallel({
    "code":RunnablePassthrough(),
    "explanation":prompt1|model|parser
}
    
)

chain=par|prompt2|model|parser

res = chain.invoke({"code": chunk[0].page_content})

print("\n✅ Final Result:\n")
print(res)