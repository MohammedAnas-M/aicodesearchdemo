import streamlit as st
import requests

st.set_page_config(page_title="AI Code Search", layout="wide")  # Use 'wide' layout

# Set up the title of the app
st.title("AI Code Search")

# Text area for user input
prompt = st.text_area("Prompt:", height=150)

# Button to send the prompt
if st.button("Send"):
    if prompt:
        try:
            # Send the prompt to the Flask backend
            response = requests.post("http://127.0.0.1:5000/members", json={"prompt": prompt})
            
            if response.status_code == 200:
                response_data = response.json()

                # Create a container for the response
                with st.container():
                    st.subheader("Response:")
                    st.code(response_data.get("members", "No response received."), language='javascript')
            else:
                st.error(f"Error: {response.text}")
        except Exception as e:
            st.error(f"Error while fetching data from server: {e}")
    else:
        st.warning("Please enter a prompt.")
