import json

from chitchat import ChitChatDialogue
from ai_engine.messages import DialogueMessage,BaseMessage
from uagents import Agent, Context, Model, Field
from groq import Groq


# define dialogue messages; each transition needs a separate message
class InitiateChitChatDialogue(Model):
    assistant_type : str = Field(
        description = 'you MUST ask user what type of assistant user is looking for? This is the type of assistant you are looking for.'
    )
    model : str = Field(
        description = "you MUST always ask user what llm model they want to choose from options :'gemma-7b-it','gemma2-9b-it','llama3-8b-8192','llama3-groq-70b-8192-tool-use-preview','llava-v1.5-7b-4096-preview','whisper-large-v3','mixtral-8x7b-32768'. Do remeber you MUST provide these as options and user has to select one from them."
    )


class AcceptChitChatDialogue(BaseMessage):
    type: str = "agent_message"

    # user messages, this is the text that the user wants to send to the agent
    agent_message: str


class ChitChatDialogueMessage(DialogueMessage):
    """ChitChat dialogue message"""

    pass


class ConcludeChitChatDialogue(Model):
    """I conclude ChitChat dialogue request"""

    pass


class RejectChitChatDialogue(Model):
    """I reject ChitChat dialogue request"""

    pass


async def generate_response(assistant, user_input,model):
    client = Groq(api_key="<your_groq_api_key>", )

    system_prompt = {
    "role": "system",
    "content":
    f"You are a {assistant}. You reply with very short answers."
    }

    # Initialize the chat history
    chat_history = [system_prompt]

    chat_history.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(model=model,
                                            messages=chat_history,
                                            max_tokens=100,
                                            temperature=1.2)
    # Append the response to the chat history
    chat_history.append({
        "role": "assistant",
        "content": response.choices[0].message.content
        })
    # Print the response
    return response.choices[0].message.content

Mailbox_key = "<Replace it with your mailbox key>"

agent = Agent(
    name="groq-example-agent",
    seed="<Your agent_address>",
    mailbox=f"{Mailbox_key}@https://agentverse.ai",
    log_level="DEBUG",
)


# instantiate the dialogues
chitchat_dialogue = ChitChatDialogue(
    version="0.66",
    storage=agent.storage,
)

@chitchat_dialogue.on_initiate_session(InitiateChitChatDialogue)
async def start_chitchat(
    ctx: Context,
    sender: str,
    msg: InitiateChitChatDialogue,
):
    ctx.logger.info(f"Received init message from {sender} Session: {ctx.session}")
    # do something when the dialogue is initiated
    ctx.storage.set('Assitant',msg.assistant_type)
    ctx.storage.set('Model',msg.model)
    await ctx.send(sender, AcceptChitChatDialogue(agent_message=f"Hello, I am your {msg.assistant_type} assistant and I am running on LLM Model {msg.model}"))


@chitchat_dialogue.on_start_dialogue(AcceptChitChatDialogue)
async def accepted_chitchat(
    ctx: Context,
    sender: str,
    _msg: AcceptChitChatDialogue,
):
    ctx.logger.info(
        f"session with {sender} was accepted. This shouldn't be called as this agent is not the initiator."
    )


@chitchat_dialogue.on_reject_session(RejectChitChatDialogue)
async def reject_chitchat(
    ctx: Context,
    sender: str,
    _msg: RejectChitChatDialogue,
):
    # do something when the dialogue is rejected and nothing has been sent yet
    ctx.logger.info(f"Received conclude message from: {sender}")


@chitchat_dialogue.on_continue_dialogue(ChitChatDialogueMessage)
async def continue_chitchat(
    ctx: Context,
    sender: str,
    msg: ChitChatDialogueMessage,
):
    # do something when the dialogue continues
    ctx.logger.info(f"Received message: {msg.user_message} from: {sender}")
    type = ctx.storage.get('Assitant')
    model = ctx.storage.get('Model')
    ctx.logger.info(f'User selected the {type} type of assistant with model: {model}.')
    response = await generate_response(msg.user_message, type,model)
    try:
        await ctx.send(
            sender,
            ChitChatDialogueMessage(
                type="agent_message",
                agent_message=response
            ),
        )
    except EOFError:
        await ctx.send(sender, ConcludeChitChatDialogue())


@chitchat_dialogue.on_end_session(ConcludeChitChatDialogue)
async def conclude_chitchat(
    ctx: Context,
    sender: str,
    _msg: ConcludeChitChatDialogue,
):
    # do something when the dialogue is concluded after messages have been exchanged
    ctx.logger.info(f"Received conclude message from: {sender}; accessing history:")
    ctx.logger.info(ctx.dialogue)


agent.include(chitchat_dialogue, publish_manifest=True)


if __name__ == "__main__":
    print(f"Agent address: {agent.address}")
    agent.run()
