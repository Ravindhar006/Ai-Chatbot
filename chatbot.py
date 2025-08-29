!pip install -U langchain-community
from google.colab import drive
drive.mount('/content/drive')
!pip install gradio langchain huggingface_hub langchain_groq
!pip install -U langchain-community
!pip install pypdf
!pip install chromadb
!pip install gradio






import os
import shutil
import gradio as gr
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from transformers import pipeline

# ✅ Set Groq API Key securely
os.environ["GROQ_API_KEY"] = "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"

# ✅ Global chain
qa_chain = None

# ✅ Initialize Groq LLM (LLaMA3)
def initialize_llm():
    return ChatGroq(
        temperature=0,
        groq_api_key=os.environ["GROQ_API_KEY"],
        model_name="llama3-70b-8192"
    )

# ✅ Sentiment analysis model
sentiment_model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# ✅ Create vector DB from uploaded PDF
def create_vector_db(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError("No PDF file found.")

    loader = PyPDFLoader(filepath)
    documents = loader.load()

    if not documents:
        raise ValueError("No content found in the PDF.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_documents(documents)

    embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db")
    db.persist()
    return db

# ✅ Setup RetrievalQA chain with custom prompt
def setup_qa_chain(db, llm):
    retriever = db.as_retriever()
    prompt_template = """You are a compassionate mental health chatbot. Be supportive and helpful in responses.

Context:
{context}

User: {question}
Chatbot:"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )

# ✅ Handle file upload in Colab (Gradio gives path, not file-like)
def upload_and_embed(file):
    if file is None:
        return "⚠ Please upload a PDF file."

    filepath = "data/temp.pdf"
    os.makedirs("data", exist_ok=True)

    try:
        shutil.copy(file, filepath)
    except Exception as e:
        return f"❌ Failed to copy uploaded file: {e}"

    try:
        db = create_vector_db(filepath)
        llm = initialize_llm()
        global qa_chain
        qa_chain = setup_qa_chain(db, llm)
        return "✅ PDF uploaded and chatbot initialized."
    except Exception as e:
        return f"❌ Error processing file: {e}"

# ✅ Chat function
def chatbot_response(message, history):
    global qa_chain
    if not message.strip():
        return history + [{"role": "assistant", "content": "Please enter a valid message."}]

    if qa_chain is None:
        return history + [{"role": "assistant", "content": "❌ Chatbot not ready. Please upload a PDF first."}]

    try:
        sentiment = sentiment_model(message)[0]['label']
        response = qa_chain.run(message)

        # Add encouragement or suggestions
        if "anxiety" in message.lower():
            response += "\n\n🧘 Try this breathing exercise: https://www.youtube.com/watch?v=odADwWzHR24"
        elif sentiment == "NEGATIVE":
            response += "\n\n💙 I'm here for you. You're not alone."

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})
        return history
    except Exception as e:
        return history + [{"role": "assistant", "content": f"❌ Error: {e}"}]

# ✅ Gradio UI
with gr.Blocks(theme='Respair/Shiki@1.2.3') as app:
    gr.Markdown("## 🧠 Mental Health Chatbot\nUpload a PDF and start chatting.")
    file_upload = gr.File(label="📄 Upload Mental Health PDF", file_types=[".pdf"])
    upload_btn = gr.Button("📥 Upload & Initialize")
    upload_output = gr.Textbox(label="Upload Status")

    chat_interface = gr.ChatInterface(
        fn=chatbot_response,
        title="🧠 Mental Health Chatbot",
        chatbot=gr.Chatbot(type="messages")
    )

    upload_btn.click(fn=upload_and_embed, inputs=file_upload, outputs=upload_output)

# ✅ Run in Colab with debug=True
app.launch(debug=True, share=True)




