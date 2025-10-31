import os
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import platform

# 🌻 Configuración del tema floral girasol
st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #fff8dc 0%, #fff6bf 50%, #fff3b0 100%);
    color: #4a3000;
    font-family: 'Trebuchet MS', sans-serif;
}
h1, h2, h3, h4 {
    color: #5a3e00 !important;
    text-align: center;
    font-weight: bold;
}
.stButton button {
    background-color: #f6c700 !important;
    color: #4a3000 !important;
    border: none;
    border-radius: 10px;
    font-weight: bold;
    transition: 0.3s ease-in-out;
}
.stButton button:hover {
    background-color: #ffd93b !important;
    transform: scale(1.05);
}
.stTextInput input, .stTextArea textarea {
    border-radius: 10px;
    border: 2px solid #f6c700;
    background-color: #fff9e6;
}
.stFileUploader label {
    color: #5a3e00 !important;
    font-weight: bold;
}
.stAlert {
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# 🌻 Configuración de la página
st.set_page_config(page_title="Analizador PDF 🌻", layout="centered", initial_sidebar_state="collapsed")

# 🌻 Título principal
st.title('🌻 Generación Aumentada por Recuperación (RAG) 💬')
st.write("Versión de Python:", platform.python_version())

# 🌼 Imagen decorativa
try:
    image = Image.open('Chat_pdf.png')
    st.image(image, width=350, caption="🌼 Analiza documentos con IA 🌼")
except Exception as e:
    st.warning(f"No se pudo cargar la imagen: {e}")

# 🌻 Sidebar
with st.sidebar:
    st.subheader("🌼 Asistente de Análisis de PDF 🌼")
    st.markdown("Este agente te ayudará a analizar y responder preguntas sobre el contenido de tu archivo PDF.")
    st.markdown("---")

# 🌻 Ingreso de clave de API
ke = st.text_input('🔑 Ingresa tu Clave de OpenAI', type="password")
if ke:
    os.environ['OPENAI_API_KEY'] = ke
else:
    st.warning("🌻 Por favor ingresa tu clave de API de OpenAI para continuar")

# 🌻 Carga del PDF
pdf = st.file_uploader("📄 Carga tu archivo PDF", type="pdf")

# 🌻 Procesamiento del PDF
if pdf is not None and ke:
    try:
        # Extraer texto del PDF
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        st.info(f"📜 Texto extraído: {len(text)} caracteres encontrados 🌼")

        # Dividir el texto en fragmentos
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=500,
            chunk_overlap=20,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        st.success(f"🌻 Documento dividido en {len(chunks)} fragmentos")

        # Crear embeddings y base de conocimiento
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # Pregunta del usuario
        st.subheader("💬 Pregunta sobre el documento")
        user_question = st.text_area("🌼 Escribe tu pregunta aquí...")

        # Procesar la pregunta
        if user_question:
            docs = knowledge_base.similarity_search(user_question)

            llm = OpenAI(temperature=0, model_name="gpt-4o")
            chain = load_qa_chain(llm, chain_type="stuff")

            with st.spinner("🌻 Analizando el documento..."):
                response = chain.run(input_documents=docs, question=user_question)

            # Mostrar respuesta
            st.markdown("### 🌼 Respuesta:")
            st.markdown(response)

    except Exception as e:
        st.error(f"🚫 Error al procesar el PDF: {str(e)}")
        import traceback
        st.error(traceback.format_exc())

elif pdf is not None and not ke:
    st.warning("🌻 Por favor ingresa tu clave de API de OpenAI para continuar")
else:
    st.info("🌼 Por favor carga un archivo PDF para comenzar 🌼")
