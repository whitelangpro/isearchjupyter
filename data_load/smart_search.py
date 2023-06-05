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
from langchain.chains.summarize import load_summarize_chain
from langchain.memory import ConversationBufferMemory
from langchain import ConversationChain, LLMChain
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


def get_session_info(table_name, session_id):

    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(table_name)
    
    session_result = ""
    response = table.get_item(Key={'session-id': session_id})
    if "Item" in response.keys():
        session_result = json.loads(response["Item"]["content"])
    else:
        session_result = ""

    return session_result
    
    
def update_session_info(table_name, session_id, question, answer, intention):

    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table(table_name)
    session_result = ""

    response = table.get_item(Key={'session-id': session_id})

    if "Item" in response.keys():
        chat_history = json.loads(response["Item"]["content"])
    else:
        chat_history = []

    chat_history.append([question, answer, intention])
    content = json.dumps(chat_history)

    response = table.put_item(
        Item={
            'session-id': session_id,
            'content': content
        }
    )

    if "ResponseMetadata" in response.keys():
        if response["ResponseMetadata"]["HTTPStatusCode"] == 200:
            update_result = "success"
        else:
            update_result = "failed"
    else:
        update_result = "failed"

    return update_result

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
                    print(f"{file} Failed to load")
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
                    print("The following files failed to load:")
                    for file in failed_files:
                        print(file,end="\n")
        else:
            docs = []
            for file in filepath:
                try:
                    print("begin to load file, file:",file,self.language)
                    docs += load_file(file,self.language)
                    print(f"{file} Loaded successfully")
                    loaded_files.append(file)
                except Exception as e:
                    print(e)
                    print(f"{file} Failed to load")
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
        docs_with_scores = self.vector_store.similarity_search(query,k=top_k)
        docs = [doc[0] for doc in docs_with_scores]
        result = chain({"input_documents": docs, "question": query}, return_only_outputs=True)
        return result,docs_with_scores

    def get_summarize(self,texts,
                        chain_type: str = "stuff",
                        prompt_template: str = "请根据{text}，总结一段摘要",
                        combine_prompt_template: str = "请根据{text}，总结一段摘要"
                        ):
                            
        texts = texts.split(';')
        texts_len = len(texts)
        print("texts len:",texts_len)
        
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
        COMBINE_PROMPT = PromptTemplate(template=combine_prompt_template, input_variables=["text"])
        
        print('prompt:',PROMPT)
        
        if chain_type == "stuff":
            docs = [Document(page_content=t) for t in texts]
            chain = load_summarize_chain(self.llm, chain_type="stuff", prompt=PROMPT)
            result = chain.run(docs)
            
        else:
            new_texts = []
            num = 20
            for i in range(0,texts_len,num):
                if i + num < texts_len:
                    end = i + num
                else:
                    end = texts_len - i
                if len(texts[i:end]) > 0:
                    new_texts.append(";".join(texts[i:end]))
            docs = [Document(page_content=t) for t in new_texts]
            
            chain = load_summarize_chain(self.llm, 
                                         chain_type=chain_type, 
                                         map_prompt=PROMPT,
                                         combine_prompt=COMBINE_PROMPT)
            result = chain({"input_documents": docs}, return_only_outputs=True)
            result = result['output_text']
        
        return result

    def get_chat(self,query,language,table_name,session_id):
        if language == "chinese":
            template = """
            你是一个乐于助人的助手，旨在能够协助完成各种任务，从回答简单的问题到就各种主题提供深入的解释和讨论。助手能够根据收到的人类输入，生成问题回答文本，使其能够参与听起来自然的对话，并提供与手头主题连贯且相关的响应。
            
    {history}
    人类输入: {human_input}
    助手:"""
        elif language == "english":
            template = """
            Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.
            
            {history}
            Human: {human_input}
            Assistant:"""
            
        prompt = PromptTemplate(
            input_variables=["history", "human_input"], 
            template=template
        )
        
        memory = ConversationBufferMemory(return_messages=True)
        session_info = ""
        if len(session_id) > 0:
            session_info = get_session_info(table_name,session_id)
            if len(session_info) > 0:
                for item in session_info:
                    print("session info:",item[0]," ; ",item[1]," ; ",item[2])
                    if item[2] == "chat":
                        memory.chat_memory.add_user_message(item[0])
                        memory.chat_memory.add_ai_message(item[1])
        
        chat_chain = LLMChain(
            llm=self.llm,
            prompt=prompt, 
            # verbose=True, 
            memory=memory,
        )
        output = chat_chain.predict(human_input=query)
        update_session_info(table_name, session_id, query, output, "chat")
        
        return output