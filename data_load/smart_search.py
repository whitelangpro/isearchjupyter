import os
import shutil
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import TextLoader
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import UnstructuredPowerPointLoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.embeddings import SagemakerEndpointEmbeddings
from langchain.embeddings.sagemaker_endpoint import EmbeddingsContentHandler
from langchain.vectorstores import OpenSearchVectorSearch
from langchain.docstore.document import Document
from langchain.text_splitter import NLTKTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain import SagemakerEndpoint
from langchain.llms.sagemaker_endpoint import ContentHandlerBase
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from chinese_text_splitter import ChineseTextSplitter
import json
from typing import Dict, List, Tuple
from tqdm import tqdm
from datetime import datetime
import boto3

def load_file(filepath,language):
    
    if filepath.lower().endswith(".pdf"):
        print('begin to load pdf file')
        loader = PyPDFLoader(filepath)
    elif filepath.lower().endswith(".docx"):
        loader = Docx2txtLoader(filepath)
    elif filepath.lower().endswith(".pptx"):
        loader = UnstructuredPowerPointLoader(filepath)
    elif filepath.lower().endswith(".csv"):
        loader = CSVLoader(filepath)
    elif filepath.lower().endswith(".txt"):
        loader = TextLoader(filepath)
    else:
        loader = TextLoader(filepath)

    if language == "chinese":
        if filepath.lower().endswith(".pdf"):
            print("language is chinese and file is pdf")
            textsplitter = ChineseTextSplitter(pdf=True)
        else:
            textsplitter = ChineseTextSplitter()
    elif language == "english":
        textsplitter = NLTKTextSplitter(chunk_size=1000, chunk_overlap=100)

    docs = loader.load_and_split(textsplitter)
    
    return docs


def init_embeddings(endpoint_name,region_name,language: str = "chinese"):
    
    class ContentHandler(EmbeddingsContentHandler):
        content_type = "application/json"
        accepts = "application/json"

        def transform_input(self, inputs: List[str], model_kwargs: Dict) -> bytes:
            input_str = json.dumps({"inputs": inputs, **model_kwargs})
            return input_str.encode('utf-8')

        def transform_output(self, output: bytes) -> List[List[float]]:
            response_json = json.loads(output.read().decode("utf-8"))
            if language == "chinese":
                return response_json[0][0][0]
            elif language == "english":
                return response_json["vectors"][0]

    content_handler = ContentHandler()

    embeddings = SagemakerEndpointEmbeddings(
        endpoint_name=endpoint_name, 
        region_name=region_name, 
        content_handler=content_handler
    )
    return embeddings


def init_vector_store(embeddings,
             index_name,
             opensearch_host,
             opensearch_port,
             opensearch_user_name,
             opensearch_user_password):

    vector_store = OpenSearchVectorSearch(
        index_name=index_name,
        embedding_function=embeddings, 
        opensearch_url="aws-opensearch-url",
        hosts = [{'host': opensearch_host, 'port': opensearch_port}],
        http_auth = (opensearch_user_name, opensearch_user_password),
    )
    return vector_store


def init_model(endpoint_name,
               region_name,
               temperature: float = 0.01):
    try:
        class ContentHandler(LLMContentHandler):
            content_type = "application/json"
            accepts = "application/json"

            def transform_input(self, prompt: str, model_kwargs: Dict) -> bytes:
                input_str = json.dumps({"ask": prompt, **model_kwargs})
                return input_str.encode('utf-8')

            def transform_output(self, output: bytes) -> str:
                response_json = json.loads(output.read().decode("utf-8"))
                return response_json['answer']

        content_handler = ContentHandler()

        llm=SagemakerEndpoint(
                endpoint_name=endpoint_name, 
                region_name=region_name, 
                model_kwargs={"temperature":temperature},
                content_handler=content_handler,
        )
        return llm
    except Exception as e:
        return None


class SmartSearchQA:
    
    def init_cfg(self,
                 opensearch_index_name,
                 opensearch_user_name,
                 opensearch_user_password,
                 opensearch_host,
                 opensearch_port,
                 embedding_endpoint_name,
                 region,
                 llm_endpoint_name: str = 'pytorch-inference-llm-v1',
                 temperature: float = 0.01,
                 language: str = "chinese"
                ):
        self.language = language
        self.llm = init_model(llm_endpoint_name,region,temperature)
        embeddings = init_embeddings(embedding_endpoint_name,region,self.language)
        self.vector_store = init_vector_store(embeddings,
                                             opensearch_index_name,
                                             opensearch_host,
                                             opensearch_port,
                                             opensearch_user_name,
                                             opensearch_user_password)
        
    def init_knowledge_vector(self,filepath: str or List[str], bulk_size: int = 10000):
        loaded_files = []
        failed_files = []
        if isinstance(filepath, str):
            if not os.path.exists(filepath):
                print("Path does not exist")
                return None
            elif os.path.isfile(filepath):
                file = os.path.split(filepath)[-1]
                try:
                    docs = load_file(filepath,self.language)
                    print(f"{file} Loaded successfully")
                    loaded_files.append(filepath)
                except Exception as e:
                    print(e)
                    print(f"{file} Failed to load successfully")
                    return None
            elif os.path.isdir(filepath):
                docs = []
                for file in tqdm(os.listdir(filepath), desc="Load the file"):
                    fullfilepath = os.path.join(filepath, file)
                    try:
                        docs += load_file(fullfilepath,self.language)
                        loaded_files.append(fullfilepath)
                    except Exception as e:
                        failed_files.append(file)

                if len(failed_files) > 0:
                    print("The following files failed to load successfully:")
                    for file in failed_files:
                        print(file,end="\n")
        else:
            docs = []
            for file in filepath:
                try:
                    print("begin to load file, file:",file,language)
                    docs += load_file(file,self.language)
                    print(f"{file} Loaded successfully")
                    loaded_files.append(file)
                except Exception as e:
                    print(e)
                    print(f"{file} Failed to load successfully")
        if len(docs) > 0:
            print("The file is loaded and the vector library is being generated")
            if self.vector_store is not None:
                texts = [d.page_content for d in docs]
                metadatas = [d.metadata for d in docs]
                ids = self.vector_store.add_texts(texts, metadatas, bulk_size=bulk_size, language=self.language)
                return loaded_files
            else:
                print("Vector library is not specified, please specify the vector database")
        else:
            print("None of the files loaded successfully, please check the file to upload again.")
            return loaded_files
        
        
    def get_answer_from_RetrievalQA(self,query,
                                        prompt_template: str = "请根据{context}，回答{question}",
                                        top_k: int = 3):
        
        prompt = PromptTemplate(template=prompt_template,
                                input_variables=["context", "question"])
        
        QA_chain = RetrievalQA.from_llm(
            llm=self.llm,
            retriever=self.vector_store.as_retriever(search_kwargs={"k": top_k}),
            prompt=prompt)
        
        # QA_chain.combine_documents_chain.document_prompt = PromptTemplate(
        #     input_variables=["page_content"], template="{page_content}")

        QA_chain.return_source_documents = True
        result = QA_chain({"query": query})
        return result
        
    def get_answer_from_load_qa_chain(self,query,
                                        prompt_template: str = "请根据{context}，回答{question}",
                                        top_k: int = 3):
                                       
        prompt = PromptTemplate(template=prompt_template,
                                input_variables=["context", "question"])

        chain = load_qa_chain(self.llm, chain_type="stuff", prompt=prompt)
        docs = self.vector_store.similarity_search(query,k=top_k)
        result = chain({"input_documents": docs, "question": query}, return_only_outputs=True)
        return result,docs