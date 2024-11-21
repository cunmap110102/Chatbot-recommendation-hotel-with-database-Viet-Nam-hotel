from enum import Enum

class ToolType(str, Enum):
    RETRIEVER = "retriever-tool"  # Dùng để tìm kiếm trong cơ sở dữ liệu nội bộ
    ONLINE_SEARCH = "online-search-tool"  # Dùng để tìm kiếm trực tuyến khi không có kết quả
