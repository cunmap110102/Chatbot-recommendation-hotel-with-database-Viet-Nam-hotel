from langchain.tools.retriever import create_retriever_tool
from langchain_community.utilities import SerpAPIWrapper
from langchain.agents import Tool
from langchain_core.vectorstores import VectorStoreRetriever
from abc import abstractmethod

class Tools:
    """
    The base class for tools.
    """
    @classmethod
    @abstractmethod
    def get(cls) -> Tool:
        pass

class RetrieverTool(Tool):
    @classmethod
    def get(cls, retriever: VectorStoreRetriever) -> Tool:
        print(f"[INFO] Using Retriever Tool")
        query = "5-star hotel in Da Nang"  # Đảm bảo rằng truy vấn đúng
        results = retriever.get_relevant_documents(query)  # Kiểm tra kết quả tìm kiếm
        print(f"[DEBUG] Retriever results: {results}")  # Thêm dòng này để debug
        return create_retriever_tool(
            retriever=retriever,
            name="retriever-tool",
            description="Search related documents to answer questions"
        )


class OnlineSearchTool(Tool):
    @classmethod
    def get(cls) -> Tool:
        print(f"[INFO] Using Online Search Tool")
        search = SerpAPIWrapper()  # Make sure SerpAPIWrapper is properly configured
        return Tool(
            name="online-search-tool",
            description="Online search to answer questions when the retriever tool fails",
            func=search.run
        )

