import os
from langchain import hub
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
import gradio as gr
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Constants
ABS_PATH: str = os.path.dirname(os.path.abspath(__file__))
DB_DIR: str = os.path.join(ABS_PATH, "db")
LANGSMITH_API_KEY = os.getenv('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')

# Ensure the API key is set
if not LANGSMITH_API_KEY:
    raise ValueError("LangSmith API key is not set. Please set the LANGSMITH_API_KEY environment variable.")

# Set up RetrievalQA model
rag_prompt_mistral = hub.pull("rlm/rag-prompt-mistral")

def load_model():
    llm = Ollama(
        model="mistral",
        verbose=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    )
    return llm

def retrieval_qa_chain(llm, vectorstore):
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": rag_prompt_mistral},
        return_source_documents=True,
    )
    return qa_chain

def qa_bot():
    llm = load_model()
    vectorstore = Chroma(
        persist_directory=DB_DIR, embedding_function=OllamaEmbeddings(model="mistral")
    )
    qa = retrieval_qa_chain(llm, vectorstore)
    return qa

# Initialize the QA chain
chain = qa_bot()

# Gradio interface
def chatbot_response(message):
    if chain is None:
        return "The chatbot is currently unavailable."

    cb = CallbackManager([StreamingStdOutCallbackHandler()])
    res = chain({"query": message}, callbacks=[cb])
    answer = res["result"]
    source_documents = res["source_documents"]

    response = answer
    if source_documents:
        sources = "\n".join([f"Source {i+1}: {doc.page_content}" for i, doc in enumerate(source_documents)])
        response += f"\n\nSources:\n{sources}"
    return response

iface = gr.Interface(
    fn=chatbot_response,
    inputs=gr.Textbox(lines=2, placeholder="Enter your message here...", label="Your Message"),
    outputs=gr.Textbox(label="Response"),
    title="Vedic Chatbot",
    description="Hi, Welcome to Vedic Chatbot. Let me know how can I assist you?",
    theme="default",
    live=True
)

if __name__ == "__main__":
    iface.launch()
