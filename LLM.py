from langchain_community.llms import Ollama, HuggingFaceEndpoint
from langchain_openai import OpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import dotenv_values
import os
from typing import List
from langchain_core.output_parsers import JsonOutputParser, PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field


### Read API Keys from env
env = dotenv_values("env")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = env['HUGGINGFACEHUB_API_TOKEN']
os.environ["OPENAI_API_KEY"] = env["OPENAI_API_KEY"]
###

### Define data format fro Json output
class DictData(BaseModel):
    kanji: str = Field(description="相關的台語漢字")
    lomajis: List[str] = Field(description="相關的羅馬字")
    wordIDs: List[int] = Field(description="相關的台語詞ID")
    sentenceIDs: List[int] = Field(description="相關的例句ID")


class TaiwaneseDictionaryRetreiver:
    json_parser = JsonOutputParser(pydantic_object=DictData)
        
    def __init__(self, model="wangshenzhi/llama3-8b-chinese-chat-ollama-q4"):
        self.retriever = self.initialize_retriever()
        self.prompt = self.initialize_prompt()
        self.json_prompt = self.initialize_json_prompt()
        self.llm = self.initialize_llm(model=model)
    
    def initialize_llm(self, model):
        if model == "mistralai/Mistral-7B-Instruct-v0.2":
            return HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.2", temperature=0.75,)
        elif model == "ChatGPT":
            return OpenAI(model_name="gpt-3.5-turbo-instruct")
        else:
            return Ollama(model=model)
        
    def initialize_retriever(self):
        my_embeddings = HuggingFaceInferenceAPIEmbeddings(
            model_name= "aspire/acge_text_embedding", #"sentence-transformers/all-MiniLM-l6-v2",
            api_key=os.environ["HUGGINGFACEHUB_API_TOKEN"],
        )
        my_vectorstore = Chroma(embedding_function = my_embeddings,
                                persist_directory = './ChromaDB')
        return my_vectorstore.as_retriever(search_kwargs={"k": 3})
        
    def initialize_prompt(self):
        ## 定義 prompt
        template = """
        請根據 context 回答問題。請使用台語字典的資料的詞彙、例句、羅馬字來回答，請確保你的回答跟台語字典的資料的一致性。

        請參考下面台語字典的資料並使用在你的回答：
        {context}
        
        問題:
        {question}
        
        請用以下格式回答：
        \n
        台語漢字： 
        羅馬字： 
        意思：
        \n
        """
        return ChatPromptTemplate.from_template(template)
        
    def initialize_json_prompt(self):
        ## 定義 prompt
        template = """請根據台語字典的資料回答問題，並嚴格遵循 json 的格式來回答:

        請參考下面台語字典的資料並使用在你的回答：
        {context}
        
        問題:
        {question}
        
        {format_instructions}
        """        
        # Set up a parser + inject instructions into the prompt template.
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"],
            partial_variables={"format_instructions": self.json_parser.get_format_instructions()},
        )
        
    ## 多加了一個整理 splitted doc 的小函數
    def format_docs(self, docs):
        str = "\n\n".join([d.page_content for d in docs])
        print(f"\n---------------\n{str}\n---------------\n")
        return str
    
    def ask(self, text):
        return self.llm.invoke(text)
    
    def query(self, text):
        chain = (
            {"context": self.retriever | self.format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        return chain.stream(text)
    
    def query_json(self, text):
        chain = (
            {"context": self.retriever | self.format_docs, "question": RunnablePassthrough()}
            | self.json_prompt
            | self.llm
            | self.json_parser
        )
        return chain.invoke(text)
        