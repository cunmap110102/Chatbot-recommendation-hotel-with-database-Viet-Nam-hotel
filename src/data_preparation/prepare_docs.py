import os, re
import pandas as pd
import numpy as np
from abc import abstractmethod

np.random.seed(0)

DATA_DIR = os.path.join(os.path.dirname(os.getcwd()), 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')


class Data:
    def __init__(self, raw_file_name: str):
        self.raw_file_name = raw_file_name
        self.raw_data_name = f"{self.raw_file_name}.csv"  # Đảm bảo tên tệp ở đây đúng (Hotel_Reviews.csv)
        self.processed_data_name = ""
        self.data = None
        self._check_raw_data()

    @abstractmethod
    def create_processed_data(self, country: str):
        pass

    @abstractmethod
    def _check_raw_data(self) -> bool:
        pass

    @abstractmethod
    def _check_processed_data(self) -> bool:
        pass


class CSVData(Data):
    def __init__(self, raw_file_name):
        super().__init__(raw_file_name)

    def create_processed_data(self, country):
        # Kiểm tra xem quốc gia có phải "Vietnam" không, nếu không sẽ báo lỗi
        if country != "Vietnam":
            raise NotImplementedError("Need to implement for other countries.")

        self.processed_data_name = f"{country}_processed_df.csv"
        data_path = os.path.join(PROCESSED_DATA_DIR, self.processed_data_name)

        # Cập nhật đường dẫn đến bộ dữ liệu mới
        if os.path.isfile(data_path):
            self._check_processed_data()
        else:
            print("[INFO] Creating processed data.")
            
            # Đọc dữ liệu từ file CSV mới
            df = pd.read_csv(r"C:\Users\USER\Hotel-Recommendation-Chatbot\data\processed\Final_Updated_Hotel_Data_v3.csv")
            
            # Tiến hành xử lý dữ liệu, tách và chuẩn hóa các cột, v.v.
            if 'text' in df.columns:
                df['Clean_Tags'] = df['text'].apply(self._clean_tag)
            else:
                print("[INFO] 'Tags' or 'text' column not found. Skipping tag cleaning.")
                df['Clean_Tags'] = None  # Nếu không có 'Tags' hoặc 'text', gán cột 'Clean_Tags' là None

            # Xử lý đặc thù cho Vietnam (có thể là cách chia cột địa chỉ)
            df['City'] = df['hotelName'].apply(lambda x: x.split(" ")[-1] if len(x.split(" ")) > 1 else None)
            df['Postal_Code'] = df['hotelName'].apply(lambda x: x.split(" ")[-2] if len(x.split(" ")) > 1 else None)

            # Lưu dữ liệu đã xử lý vào file mới
            df.to_csv(data_path, index=False)
            
            # Lọc và xử lý các bài đánh giá
            df = df[
                (df['hotelName'].str.contains(country))
                & (df['username'].str.contains(country))
            ]
            df.reset_index(drop=True, inplace=True)
            df = df[df['rating'] >= 8]  # Chỉ lấy các khách sạn có điểm đánh giá từ 8 trở lên

            take_cols = [
                'hotelName', 'rating', 'text', 'locationId', 'createdDate', 'stayDate', 'tripType',
                'Clean_Tags', 'username', 'userId'
            ]
            df = df[take_cols]

            # Tạo bảng tóm tắt với khách sạn có tên, điểm đánh giá và các cột khác
            agg_df = df.groupby('hotelName').agg(
                {
                    'rating': 'first',
                    'locationId': 'first',
                    'createdDate': 'first',
                    'stayDate': 'first',
                    'tripType': 'first',
                    'Clean_Tags': 'first',
                }
            ).reset_index()

            # Lọc và xử lý các bài đánh giá
            review_dct = dict(hotelName=[], Positive_Review=[], Negative_Review=[])
            excl_reviews = ["no negative", "no positive", "none", "nothing", "n a", "na"]
            n_review = 3  # collect only last 3 reviews
            for hotel_name in agg_df['hotelName']:
                review_dct['hotelName'].append(hotel_name)
                for col in ['text']:  # Chỉ sử dụng cột 'text' nếu không có 'Positive_Review' hoặc 'Negative_Review'
                    reviews = df[df['hotelName'] == hotel_name][col].values
                    review = []
                    for text in reviews:
                        if text.lower() in excl_reviews: continue
                        review.append(text.strip())
                        if len(review) == n_review: break
                    review_dct['Positive_Review'] = review  # You can adjust how to split reviews here
            review_df = pd.DataFrame(review_dct)

            # Ghép dữ liệu đánh giá vào bảng tổng hợp
            final_df = pd.merge(left=agg_df, right=review_df, on='hotelName', how='left')
            # Lưu kết quả cuối vào file CSV
            final_df.to_csv(os.path.join(DATA_DIR, 'processed', self.processed_data_name), index=False)
