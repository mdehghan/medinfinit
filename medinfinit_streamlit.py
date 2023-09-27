import streamlit as st

import faiss
import openai
import openpyxl
import pandas as pd

from langchain import PromptTemplate
from langchain.docstore.document import Document
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
# from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.vectorstores import FAISS


# load_dotenv()

openai_api_key = st.sidebar.text_input('OpenAI API Key')

st.title("MEDinfinit Assistant Chatbot")

MODEL_NAME = "gpt-4-0613"

system_prompt = """
# YOU ARE
You are 'MedInfinit Assistant', a friendly Psychology clinic assistant.

# YOUR GOAL
Your task is to answer user questions and help them find a psychiatrist they can consult with.

## Answer
You should use DSM-5 and other Psychology context in generating a response,
and you should respond in user language.

## Ask
While you anwer user questions, also think of other questions they might have,
or additional context you might need in better responding to the user.
Ask questions from the user if you need more data to better diagnose their situation
or have enough context to make a better recommendation for a therapist they can work with.
You can also ask questions from the user to understand if:
 - they may have experienced any symptoms recently,
 - they may have noticed changes in their thoughts, feelings, or behaviors
 - they may have noticed any patterns or triggers related to the syptoms,
 - ...

## Recommend
If the user has thoughts of suicide, ask them to call 911 immediately.
Otherwise, recommend a therapist. To find a therapist make a function call to `recommend_therapist`
with your diagnosis of patient needs as the function argument.

# YOUR RESPONSE
The output you generate should include the answer to any questions the user had,
and optionally questions to further clarify user needs or symptoms.
Once you have enough context recommend a therapist the user can work with.
Limit your responses to about 100 words.
"""

if not openai_api_key.startswith('sk-'):
    st.warning('Please enter your OpenAI API key!', icon='⚠')
else:
    openai.api_key = openai_api_key

def reset_session():
    st.session_state.messages = []
    st.session_state.openai_messages = [{"role": "system", "content": system_prompt}]

def initialize_therapist_retreiver():
    file_path = './TherapistInfo.xlsx'
    # Load the XLSX file into a DataFrame
    df = pd.read_excel(file_path)

    therapistInfo = []

    # ID - Name - Title - Treatment Approach - Areas of Expertise
    for index, row in df.iterrows():
      if not row[2] == 'nan':
        therapistInfo.append(Document(page_content=row[4], metadata={'therapist': row[1], 'approach': row[3]}))

    embeddings_model = HuggingFaceEmbeddings()

    db = FAISS.from_documents(therapistInfo, embeddings_model)
    st.session_state.retriever = db.as_retriever()

# Initialize chat history
if 'initialized' not in st.session_state:
    reset_session()
    initialize_therapist_retreiver()
    st.session_state['initialized'] = True

def recommendTherapist(symptoms):
  therapist_info = (st.session_state.retriever.get_relevant_documents(symptoms))[0]
  recommendation = therapist_info.metadata['therapist'] + 'who takes the approach ' + therapist_info.metadata['approach']
  return recommendation

recommender_function = {
    "name": "recommend_therapist",
    "description": "get a list of therapists based on user needs and symptoms",       
    "parameters": {
      "type": "object",
      "properties": {
        "symptoms": {
          "description": "The symptoms mentioned by the user.",
          "type": "string"
        }
      },
      "required": ["symptoms"]
    }
}

def openAIChat(user_prompt, is_user_prompt):
  if is_user_prompt:
    st.session_state.openai_messages.append({"role": "user", "content": user_prompt})

  completion = openai.ChatCompletion.create(
    model=MODEL_NAME,
    messages=st.session_state.openai_messages,
    functions=[
        recommender_function,
    ],
    function_call="auto",
    temperature=0,
  )

  response = completion.choices[0].message
  
  # Extend conversation with assistant's reply.
  st.session_state.openai_messages.append(response)

  if hasattr(response, "function_call"):
    symptoms = response.function_call.arguments

    # Extend conversation with function value.
    st.session_state.openai_messages.append(
            {
                "role": "function",
                "name": "recommend_therapist",
                "content": recommendTherapist(symptoms),
            }
    )
    return openAIChat("", False)
  
  return response.content


def medInfinitChat(user_prompt):
  st.session_state.messages.append({"role": "user", "content": user_prompt})
  r = openAIChat(user_prompt, True)
  st.session_state.messages.append({"role": "assistant", "content": r})
  return r


st.button("Reset", on_click=reset_session)

if prompt := st.chat_input():
    not_used = medInfinitChat(prompt)

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
