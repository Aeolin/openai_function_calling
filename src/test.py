import string
import random
from function_calling import ChatHistory, ChatRole, description, CallableFunction
from api import get_chat_completion, setup_api


@description("Generate a password of a given length")
def generate_password(len: int):
    return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(len))


setup_api('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', lambda x: str(x))
history = ChatHistory('gpt-3.5-turbo', 4096, [CallableFunction(generate_password)])
history.add_message(ChatRole.SYSTEM, "You are a bot that generates passwords")
history.add_message(ChatRole.USER, "Generate a password of length 16")
completion = get_chat_completion(history)
print(completion.content)