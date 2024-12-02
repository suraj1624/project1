import streamlit as st
from groq import Groq
import os
from dotenv import load_dotenv
load_dotenv()

# Load the API key from environment variables
api = os.getenv("API")

# Initialize the Groq client
client = Groq(api_key=api)

# Function to get a solution from the Groq API
def get_solution_from_groq(question):
    # Construct the prompt for the Groq API
    prompt = f"""
    Please provide a step-by-step solution for the given math or statistics problem: {question}
    Your solution should be to the point. Do not extra explanation. Just explanation the concept and 
    solve the question if porivded with data. 
    """
    # Create a completion request
    completion = client.chat.completions.create(
        model="llama-3.2-11b-vision-preview",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.5,
        max_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )
    
    # Collect the response text
    extracted_text = ""
    for chunk in completion:
        extracted_text += chunk.choices[0].delta.content or ""
    return extracted_text.strip()

# Main function to handle text input and maintain session state
def process_text_question(question):
    # Call the Groq API to get the solution
    solution = get_solution_from_groq(question)
    return solution

# Streamlit UI code
def chat_interface():
    # Display the title of the app
    st.title("AI Tutor for Math & Statistics")
    st.write("Ask a math or statistics question, and I will provide a step-by-step solution.")

    # Check if the session state for chat history exists, if not, initialize it
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Create a text input box for the user to enter their question
    user_input = st.text_input("Enter your question:")

    # If the user submits a question, process it
    if user_input:
        # Get the solution from the Groq API
        solution = process_text_question(user_input)
        
        # Store the question and answer in session state
        st.session_state.chat_history.append({"question": user_input, "answer": solution})
        
        # Limit the history to the last 5 interactions
        if len(st.session_state.chat_history) > 5:
            st.session_state.chat_history.pop(0)

    # Display the conversation (last 5 questions and answers)
    if st.session_state.chat_history:
        st.subheader("Conversation:")
        for i, chat in enumerate(st.session_state.chat_history):
            st.write(f"**Q{i+1}:** {chat['question']}")
            st.write(f"**A{i+1}:** {chat['answer']}")
    
# Run the chat interface
if __name__ == "__main__":
    chat_interface()
