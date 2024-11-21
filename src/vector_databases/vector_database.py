import os
from langchain_community.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.document_loaders.csv_loader import CSVLoader
from abc import abstractmethod
from typing import Union

DATA_DIR = os.path.join(os.path.dirname(os.getcwd()), 'data')


class VectorDatabase:
    @classmethod
    @abstractmethod
    def get(cls, embedding_model: Union[HuggingFaceEmbeddings, OpenAIEmbeddings], country: str):
        pass

    @classmethod
    @abstractmethod
    def _create_db(cls, embedding_model: Union[HuggingFaceEmbeddings, OpenAIEmbeddings], country: str):
        pass

    @classmethod
    @abstractmethod
    def _load_db(cls, embedding_model: Union[HuggingFaceEmbeddings, OpenAIEmbeddings]):
        pass

    @staticmethod
    def _check_path_exist(path) -> bool:
        if os.path.exists(path):
            return len(os.listdir(path)) > 0
        else:
            return False


class ChromaDB(VectorDatabase):
    CHROMA_DB_PATH = os.path.join(DATA_DIR, "chroma_db")

    @classmethod
    def get(cls, embedding_model: Union[HuggingFaceEmbeddings, OpenAIEmbeddings], country: str) -> Chroma:
        if not cls._check_path_exist(cls.CHROMA_DB_PATH):
            print(f"ChromaDB doesn't exist. Creating ChromaDB.")
            cls._create_db(embedding_model, country)
        print(f"Loading ChromaDB from {cls.CHROMA_DB_PATH}")
        return cls._load_db(embedding_model)

    @classmethod
    def _create_db(cls, embedding_model: Union[HuggingFaceEmbeddings, OpenAIEmbeddings], country: str):
        os.mkdir(cls.CHROMA_DB_PATH)
        loader = CSVLoader(file_path=os.path.join(DATA_DIR, f"processed/{country}_processed_df.csv"))
        documents = loader.load()
        Chroma.from_documents(documents, embedding_model, persist_directory=cls.CHROMA_DB_PATH)

    @classmethod
    def _load_db(cls, embedding_model: Union[HuggingFaceEmbeddings, OpenAIEmbeddings]) -> Chroma:
        return Chroma(persist_directory=cls.CHROMA_DB_PATH, embedding_function=embedding_model)


