from dotenv import load_dotenv
import chainlit as cl
import asyncio
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import UnstructuredHTMLLoader, PyPDFLoader, CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import os
from chainlit.input_widget import Select
from groq import Groq

# Load environment variables
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable not set")

chat_model_instance = None  # Inicializar como None
value = None  # Está alineado con el modelo seleccionado

supported_file_types = [
    "text/plain",
    "text/html",
    "application/pdf",
    "text/csv",
]

current_chain = None
client = Groq(api_key=groq_api_key)  # Instanciar el cliente Groq

# Initialize utility variables
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
)
embeddings_model = HuggingFaceEmbeddings()  # Initialize HuggingFaceEmbeddings

welcome_message = """### Interact with TKM Technology
You can upload a file or start a chat.
I support many types of files.
"""

@cl.on_chat_start
async def start():
    global value, chat_model_instance
    elements = [
        cl.Pdf(name="brochure", display="side", path="./docs/brochure.pdf"),
        cl.Video(name="video", url="https://www.youtube.com/watch?v=vyMkqxCOVPY", display="side")
    ]
    await cl.Message(content="I am ready for you... take a look at our brochure, or video", elements=elements).send()

    settings = await cl.ChatSettings(
        [
            Select(
                id="Model",
                label="Groq - Models",
                values=["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768", "gemma-7b-it"],
                initial_index=0,
            )
        ]
    ).send()
    value = settings["Model"]

    # Configurar ChatGroq después de que el modelo ha sido seleccionado
    chat_model_instance = ChatGroq(
        temperature=0,
        model=value  # Configurar el modelo aquí
    )

    await cl.Message(content=welcome_message).send()

    response = await cl.AskActionMessage(
        content="Would you like to start a chat or upload a file?",
        actions=[
            cl.Action(name="chat", value="chat", label="💬 Chat"),
            cl.Action(name="file", value="file", label="📁 File")
        ],
        timeout=180
    ).send()

    if response and response.get("value"):
        user_choice = response["value"].strip().lower()

        if user_choice == "chat":
            await start_chat()

        elif user_choice == "file":
            await ask_for_file()

        else:
            await cl.Message(content="Invalid option. Please select 'chat' or 'file'.").send()

async def start_chat():
    # Reiniciar memoria para chat libre
    global memory, current_chain
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    current_chain = None
    await cl.Message(content="You can now start chatting!").send()

async def ask_for_file():
    files = None
    while not files:
        try:
            files = await cl.AskFileMessage(
                content='Please upload a file:',
                accept=supported_file_types,
                timeout=180
            ).send()
            if files:
                await handle_file_upload(files[0])  # Toma el primer archivo de la lista
                return
        except asyncio.TimeoutError:
            await cl.Message(content="No file was uploaded. You can upload a file at any time.").send()

@cl.on_message
async def main(message: cl.Message):
    global value, current_chain, chat_model_instance

    if value is None:
        await cl.Message(content="The model is not selected.").send()
        return

    # Handle file analysis case
    if current_chain:
        response = await current_chain.ainvoke({"question": message.content})
        await cl.Message(content=response["answer"]).send()
    else:
        # Handle free chat case
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": message.content,
                }
            ],
            model=value,
        )
        await cl.Message(content=chat_completion.choices[0].message.content).send()

async def handle_file_upload(file):
    global current_chain, chat_model_instance, memory

    vectorstore = await create_vectorstore(file)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)  # Reiniciar la memoria
    current_chain = create_custom_chain(vectorstore)
    await cl.Message(content=f"`{file.name}` uploaded and processed! How can I assist you with this file?").send()

async def create_vectorstore(file):
    file_path = file.path
    file_type = file.type
    file_name = file.name

    with open(file_path, 'rb') as f:
        file_content = f.read()

    documents = []

    if file_type == "text/plain":
        text = file_content.decode("utf-8")
        texts = text_splitter.split_text(text)
        documents = [Document(page_content=t) for t in texts]
        embeddings = embeddings_model.embed_documents([d.page_content for d in documents])
    else:
        loader = None
        if file_type == "text/html":
            loader = UnstructuredHTMLLoader(file_path)
        elif file_type == "application/pdf":
            loader = PyPDFLoader(file_path)
        elif file_type == "text/csv":
            loader = CSVLoader(file_path)

        if loader:
            documents = loader.load()
            split_docs = text_splitter.split_documents(documents)
            embeddings = embeddings_model.embed_documents([d.page_content for d in split_docs])
            documents = split_docs

    vectorstore = Chroma.from_texts([d.page_content for d in documents], embeddings_model)
    return vectorstore

def create_custom_chain(vectorstore):
    global memory
    chain = ConversationalRetrievalChain.from_llm(
        llm=chat_model_instance,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=False
    )
    return chain