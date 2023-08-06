import streamlit as st
import openai
import pandas as pd

from langchain import PromptTemplate
from langchain.docstore.document import Document
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.vectorstores import FAISS


# load_dotenv()

openai_api_key = st.sidebar.text_input('OpenAI API Key')

st.title("MEDinfinit Assistant Chatbot")

MODEL_NAME = "gpt-4-0613"

response_schemas = [
    ResponseSchema(name="english_query", description="Translation of user prompt to English."),
    ResponseSchema(name="response", description="Response to user prompt."),
    ResponseSchema(name="translated_response", description="translated response")
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()


template = """
You are 'MedInfinit Assistant' a friendly Psychology clinic assistant.
Your task is to answer basic questions and understand patient needs.
Your goal is to collect sufficient data about the patient so that you can recommend a Psychologist who can work with them.

User query may come in any language, so you first need to translate user prompt into English.
Then generate a response in less that about 50 words.
If user is asking about a disorder, ask them if they observe any symptoms and how severe they are.
You can ask questions like the following:
- Have you noticed any changes in your thoughts, feelings, or behaviors that are
causing you distress or difficulty in functioning?
- How long have you been experiencing these symptoms?
- Have you noticed any patterns or triggers related to your symptoms?
- Do these feelings or behaviors happen across different situations and settings (like at work, home, school)?
- Have you experienced any thoughts of death or suicide, or made any plans or attempts?

You should use DSM-5 and other Psychology context in generating a response.
Once your response is complete translate it to match the user language.

If you need to recommend a therapist, make the function call `recommend_therapist'.

To generate a response:
- first translate the user query to English,
- use the translated query to respond to the user prompt in English,
- translate your generated response to match the user language.

{format_instructions}
"""

system_prompt = PromptTemplate(
    input_variables=[],
    template=template,
    partial_variables={"format_instructions": format_instructions}
).format()

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

  parsed_response = output_parser.parse(response.content)
  return parsed_response.get('translated_response')


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
