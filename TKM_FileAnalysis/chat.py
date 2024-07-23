import chainlit as cl
import os
from groq import Groq
from dotenv import load_dotenv
load_dotenv()
from chainlit.input_widget import Select

groq_api_key = os.getenv("GROQ_API_KEY")
LITERAL_API_KEY="lITERAL_API_KEY"

client = Groq(
    api_key=groq_api_key,
)
# Declarar la variable global
value = None

@cl.on_chat_start
async def start():
    global value
    elements = [
        cl.Pdf(name="brochure", display="side", path="./docs/brochure.pdf"),
        cl.Video(name="video", url="https://www.youtube.com/watch?v=vyMkqxCOVPY", display="side")
        ]
        # Reminder: The name of the pdf must be in the content of the message
    await cl.Message(content="I am ready to answer...or take a look at our brochure, or video", elements=elements).send()

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

@cl.on_message
async def main(message: cl.Message):
    global value
    if value is None:
        await cl.Message(content="The model is not selected.").send()
        return
    # Your custom logic goes here...
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": message.content,
            }
        ],
        model=value,
    )

    # Send a response back to the user
    await cl.Message(
        content=f"{chat_completion.choices[0].message.content}",
    ).send()