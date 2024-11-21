import pandas as pd
import os
from data_preparation import CSVData 
from models import Models, ModelType
from embeddings import Embeddings, EmbeddingType
from vector_databases import ChromaDB
from tools import RetrieverTool, OnlineSearchTool
from prompts import ReactPrompt
from agents import Agents
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ChatMessageHistory
from dotenv import load_dotenv
import streamlit as st
from tools.utils import get_hotel_info, get_hotel_reviews

# Load environment variables
ENV_DIR = os.path.join(os.path.dirname(os.getcwd()), ".env")
load_dotenv(ENV_DIR)

# Cập nhật đường dẫn đến bộ dữ liệu mới
data_file_path = r"C:\Users\USER\Hotel-Recommendation-Chatbot\data\processed\Final_Updated_Hotel_Data_v3.csv"
df = pd.read_csv(data_file_path)

def run(model_name, embedding_name, country="Vietnam", online_search=False):  
    # Processed data creation
    CSVData("Hotel_Reviews").create_processed_data(country)
    llm = Models.get(model_name=model_name)
    embedding_model = Embeddings.get(embedding_name=embedding_name)

    # Create vector database from embeddings
    vector_db = ChromaDB.get(embedding_model=embedding_model, country=country)
    retriever = vector_db.as_retriever(search_kwargs={'k': 3})
    tools = [RetrieverTool.get(retriever)]
    if online_search:
        search_tool = OnlineSearchTool.get()
        tools.append(search_tool)

    # Create prompt and agent
    prompt = ReactPrompt(conversation_history=True).get()
    agent = Agents.get(llm=llm, tools=tools, prompt=prompt, react=True, verbose=True)

    # Use default chat memory
    store = {}

    def get_session_history(session_id: str):
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    agent_executor = RunnableWithMessageHistory(
        agent,
        get_session_history,
        input_messages_key='input',
        history_messages_key='chat_history'
    )

    st.text("------------- Chatting -------------")
    question = st.text_input("Your question: ")
    submit = st.button("Ask!")

    # Handle hotel questions
    if question and submit:
        full_question = f"Use the 'retriever_tool' first to answer: {question}. If cannot get the answer, use other tool"
        print(f"[INFO] Agent is ready!")
        print(f"[INFO] Question: {full_question}")

        # Get response from agent
        response = agent_executor.invoke(
            {"input": full_question},
            config={"configurable": {"session_id": "test-session"}})
        response_text = response['output']
        st.text(response_text)

        # If RetrieverTool doesn't return result, trigger OnlineSearchTool
        if "I recommend the 5-star hotel" not in response_text:
            print("[INFO] Retriever Tool failed, using Online Search Tool...")
            hotel_info_query = f"5-star hotels in {question}"
            hotel_info = search_tool.run(hotel_info_query)  # Run search tool here
            st.write(f"Online Search Results: {hotel_info}")  # Display the online search results

        # Save hotel name in session state for further use
        st.session_state["hotel_name"] = response_text

    # Show hotel information if available
    if "hotel_name" in st.session_state:
        hotel_name = st.session_state["hotel_name"]
        if st.button("Show Hotel Info"):
            # Get hotel information from the local dataset first
            hotel_info = get_hotel_info(hotel_name)
            if hotel_info == "No information available." and online_search:
                # Search online if no information is available locally
                hotel_info_query = f"Basic information about {hotel_name}"
                hotel_info = search_tool.run(hotel_info_query)
            
            # Prepare information in a tabular format
            st.write(f"Hotel Information for {hotel_name}:")
            if isinstance(hotel_info, dict):
                # Create a DataFrame to display information in table format
                df_info = pd.DataFrame([{
                    'Hotel_name': hotel_info.get('hotelName', ''),
                    'Star': hotel_info.get('star', ''),
                    'Address': hotel_info.get('address', ''),
                    'Phone_number': hotel_info.get('phone_number', '')
                }])
                st.table(df_info)
            else:
                st.write(hotel_info)

        if st.button("Show Hotel Reviews"):
            # Get hotel reviews from the local dataset first
            hotel_reviews = get_hotel_reviews(hotel_name)
            if hotel_reviews == ["No reviews available."] and online_search:
                # Search online if no reviews are available locally
                hotel_reviews_query = f"Top reviews for {hotel_name}"
                hotel_reviews_result = search_tool.run(hotel_reviews_query)
                hotel_reviews = hotel_reviews_result.split("\n")[:4]  # Limit to 4 reviews
            
            # Show reviews in a more concise format as a table
            st.write(f"Top Reviews for {hotel_name}:")
            reviews_data = []
            for i, review in enumerate(hotel_reviews, 1):
                reviews_data.append({
                    'Review': f'Review {i}',
                    'Rating': '',  # Replace with actual rating if available
                    'Assessment Information': review
                })

            df_reviews = pd.DataFrame(reviews_data)
            st.table(df_reviews)

# Run Streamlit app
if __name__ == "__main__":
    st.title("StayChat: A Hotel Recommendation Chatbot")

    # Model selection
    model_options = ["ChatGPT 3.5", "Phi3 4K"]
    REGISTRY_MODEL = {
        model_options[0]: ModelType.CHATGPTSTANDARD,
        model_options[1]: ModelType.PHITHREE4k
    }
    selected_model = st.selectbox("Select model: ", model_options)
    selected_model = REGISTRY_MODEL[selected_model]

    # Embedding selection
    embedding_options = ["Sentence Transformer", "OpenAI Embedding Small"]
    REGISTRY_EMBEDDING = {
        embedding_options[0]: EmbeddingType.SENTENCE_TRANSFORMER,
        embedding_options[1]: EmbeddingType.OPENAI_EMBEDDING_SMALL
    }
    selected_embedding = st.selectbox("Select embedding: ", embedding_options)
    selected_embedding = REGISTRY_EMBEDDING[selected_embedding]

    # Country selection
    selected_country = st.selectbox("Select country: ", ["Vietnam"])
    
    # Online search option
    REGISTRY_SEARCH = {"No": False, "Yes": True}
    use_online_search = st.selectbox("Use online search? ", ["No", "Yes"])
    use_online_search = REGISTRY_SEARCH[use_online_search]

    # Run with selected parameters
    run(model_name=selected_model,
        embedding_name=selected_embedding,
        country=selected_country,
        online_search=use_online_search)
