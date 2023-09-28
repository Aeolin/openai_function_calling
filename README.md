# openai-function-calling
Is a simple library which adds convenience methods to query open ai's chat endpoint with support for function calling.

## Installation
`pip install openai-function-calling-aeolin`

## Basic Usage Example
```python
from typing import Any
from function_calling import setup_api, get_chat_compmetion, ChatHistory, ChatRole, description

def to_string(obj: Any) -> str:
    return str(obj) 

setup_api("openai api key", to_string) 

@description("Generates a password of the given length") 
def generate_password(length: int) -> str:
    return "".join([random.choice("123456789abcdefghijklmnopqrstuvwxyz") for x in range(length)])

functions = [generate_password] 
history = ChatHistory("gpt-3.5-turbo",4096,functions)
history.add_message(ChatRole.SYSTEM,"You're PasswordBot an ai that can generate passwords for people") 
history.add_message(ChatRole.USER, "can you generate me a password of length 16?")
response = get_chat_completion(history) 
print(response.content)
```
## setup_api
The setup api function takes two parameters, the first one is your openai api key, the second one is a function which can be used to convert objects to the type string. The string function is used by the library when ever a callable function returns something which is not of type string

## ChatHistory
The ChatHistory will keep track of all messages and function responses. With the last parameter of `add_message` a function can be marked as ephemeral. Per default all messages which are not of the role `SYSTEM` are ephemeral.
Ephemeral messages can and will be removed by the library if the context runs out of tokens.

## get_chat_completion
This function take a ChatHistory object as an argument and queries the openai chat endpoint with it. The resulting function responses and messages are automatically recorded in the history itself. Upon completion a `ChatMessage` object is returned with the assitant response. The message is stored in the field `content` inside the `ChatMessage`
The second parameter is an optional function which receives a list of responses if N was greated than 1 and returns one of the responses. If this function is not supplied the first response will be taken by default
The following optional parameters are passed through to the openai library (parameter name: default value)
- function_call: auto
- max_tokens: 1024
- temperature: 0.9
- top_p: 1.0
- n: 1
