import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import os

st.set_page_config(page_title="Asistente de Convenios Colectivos – SACYR", layout="wide")

st.title("🤖 Asistente de Convenios Colectivos – SACYR")
st.markdown("Haz preguntas sobre **vacaciones, jornadas, salarios, subrogaciones, tipos de contrato** y más. El asistente buscará en los convenios de Bizkaia, A Coruña y Madrid y te mostrará la página y el documento correspondiente.")

# Campo para la clave API
api_key = st.text_input("🔑 Introduce tu clave API de OpenAI para usar el asistente:", type="password")
if not api_key:
    st.warning("Por favor, introduce tu clave API de OpenAI para continuar.")
    st.stop()

# Inicializamos embeddings
os.environ["OPENAI_API_KEY"] = api_key
embeddings = OpenAIEmbeddings()

# Cargar base vectorial (ya preprocesada)
persist_directory = "vectorstore"
db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

# Crear el agente QA
retriever = db.as_retriever(search_kwargs={"k": 3})
qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(temperature=0), retriever=retriever, return_source_documents=True)

# Campo de pregunta
query = st.text_input("✍️ Escribe tu pregunta:")

if query:
    with st.spinner("Buscando en los convenios..."):
        result = qa(query)
        st.markdown("### ✅ Respuesta:")
        st.write(result["result"])

        st.markdown("---")
        st.markdown("### 📄 Documentos usados:")
        for doc in result["source_documents"]:
            name = doc.metadata.get("source", "Desconocido")
            page = doc.metadata.get("page", "¿?")
            st.markdown(f"🔹 **{name}** – Página {page}")
            st.code(doc.page_content[:1000])