import pandas as pd

# Đường dẫn đến dữ liệu khách sạn
HOTEL_DATA_PATH = r"C:\Windows\System32\Hotel-Recommendation-Chatbot\data\processed\Vietnam_processed_df.csv"

# Đọc dữ liệu khách sạn từ tệp CSV
hotel_data = pd.read_csv(HOTEL_DATA_PATH)

def get_hotel_info(hotel_name):
    hotel_info = hotel_data[hotel_data['hotelName'] == hotel_name]
    if not hotel_info.empty:
        info = hotel_info.iloc[0]
        return f"Name: {info['hotelName']}, City: {info['City']}, Rating: {info['rating']}"
    else:
        return "No information available."

def get_hotel_reviews(hotel_name):
    hotel_reviews = hotel_data[hotel_data['hotelName'] == hotel_name]
    reviews = hotel_reviews['text'].head(4).tolist()
    if reviews:
        return reviews
    else:
        return ["No reviews available."]
