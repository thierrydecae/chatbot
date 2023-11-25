import os
import sys
from dotenv import load_dotenv
import openai
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain.vectorstores import Chroma
import gradio as gr

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAPIKEY")

docs = []

for f in os.listdir("input/multiple_docs"):
    if f.endswith(".pdf"):
        pdf_path = "./input/multiple_docs/" + f
        loader = PyPDFLoader(pdf_path)
        docs.extend(loader.load())
    elif f.endswith('.docx') or f.endswith('.doc'):
        doc_path = "./input/multiple_docs/" + f
        loader = Docx2txtLoader(doc_path)
        docs.extend(loader.load())
    elif f.endswith('.txt'):
        text_path = "./input/multiple_docs/" + f
        loader = TextLoader(text_path)
        docs.extend(loader.load())

splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
docs = splitter.split_documents(docs)

# Convert the document chunks to embedding and save them to the vector store
vectorstore = Chroma.from_documents(docs, embedding=OpenAIEmbeddings(), persist_directory="./data")
vectorstore.persist()

chain = ConversationalRetrievalChain.from_llm(
    ChatOpenAI(temperature=0.7, model_name='gpt-3.5-turbo'),
    retriever=vectorstore.as_retriever(search_kwargs={'k': 6}),
    return_source_documents=True,
    verbose=False
)

chat_history = []

with gr.Blocks() as demo:
    chatbot = gr.Chatbot([("", "Hello, I'm Ben's chatbot, what can I tell you about Ben's CV?")],avatar_images=["./input/avatar/Guest.jpg","./input/avatar/Thierry Picture.jpg"])
    msg = gr.Textbox()
    clear = gr.Button("Clear")
    chat_history = []

    def user(query, chat_history):
        # print("User query:", query)
        # print("Chat history:", chat_history)

        # Convert chat history to list of tuples
        chat_history_tuples = []
        for message in chat_history:
            chat_history_tuples.append((message[0], message[1]))

        # Get result from QA chain
        result = chain({"question": query, "chat_history": chat_history_tuples})

        # Append user message and response to chat history
        chat_history.append((query, result["answer"]))
        # print("Updated chat history:", chat_history)

        return gr.update(value=""), chat_history

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False)
    clear.click(lambda: None, None, chatbot, queue=False)

demo.launch(debug=True)


