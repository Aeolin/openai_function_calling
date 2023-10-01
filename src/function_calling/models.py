import json
from enum import Enum
from typing import List, Union, Any, Callable

import tiktoken
from pydantic import TypeAdapter, Field


class ChatRole(Enum):
    SYSTEM = 'system'
    ASSISTANT = 'assistant'
    USER = 'user'
    FUNCTION = 'function'


class ChatMessage:
    def __init__(self, content: str, role: ChatRole, token_count: int, ephemeral: bool = True):
        self.content = content
        self.role = role
        self.token_count = token_count
        self.ephemeral = ephemeral

    def to_dict(self):
        return {
            "content": self.content,
            "role": self.role.value,
        }


class FunctionChatMessage(ChatMessage):
    def __init__(self, function_name: str, function_result: str, token_count: int):
        super().__init__(function_result, ChatRole.FUNCTION, token_count, True)
        self.name = function_name

    def to_dict(self):
        return {
            "name": self.name,
            "content": self.content,
            "role": self.role.value,
        }


class CallableFunction:
    def __init__(self, func):
        self.func = func
        self.adapter = TypeAdapter(func)
        self.name = func.__name__
        schema = self.adapter.json_schema()
        schema.pop('additionalProperties', None)
        if func.__description__ is not None:
            self.description = func.__description__
        self.schema = schema

    def call(self, json_str: str) -> Any:
        if self.adapter.validate_json(json_str):
            return self.func(**json.loads(json_str))

    def get_schema(self):
        return self.schema

    def to_dict(self):
        res = {
            "name": self.name,
            "parameters": self.schema,
        }
        if self.description:
            res["description"] = self.description

        return res


def description(*args, **kwargs):
    def inner(func):
        func.__description__ = args[0]
        return func

    return inner


class ChatHistory:
    def __init__(self, model: str, max_tokens: int, functions: List[Callable]):
        self.messages = []
        self.model = model
        self.tokenizer = tiktoken.encoding_for_model(model)
        self.functions = {func.__name__: CallableFunction(func) for func in functions}
        self.max_tokens = max_tokens
        self.token_bias = 0

    def add_message(self, role: ChatRole, message: str, ephemeral: Union[bool | None] = None):
        message = message.strip()
        token_count = len(self.tokenizer.encode(message))
        self.trim_history(token_count)
        self.messages.append(ChatMessage(message, role, token_count, ephemeral or role == ChatRole.SYSTEM))

    def add_response_message(self, message: ChatMessage, total_tokens: int):
        self.trim_history(message.token_count)
        self.messages.append(message)

    def update_token_bias(self, bias: int):
        self.token_bias = bias - self.total_tokens()

    def total_tokens(self) -> int:
        return sum([message.token_count for message in self.messages])

    def tokens_left(self) -> int:
        return self.max_tokens - (self.total_tokens() + self.token_bias)

    def add_function_result(self, function_name: str, content: str):
        content = content.strip()
        token_count = len(self.tokenizer.encode(content))
        self.trim_history(token_count)
        self.messages.append(FunctionChatMessage(function_name, content, token_count))

    def trim_history(self, minimum_free_space: int):
        if self.tokens_left() < minimum_free_space:
            for x in range(len(self.messages)):
                if self.messages[x].ephemeral:
                    self.messages.pop(x)
                    if self.tokens_left() >= minimum_free_space:
                        return

            if self.tokens_left() >= minimum_free_space:
                raise Exception("Not enough space in chat history to add message")

    def get_functions(self) -> List[CallableFunction]:
        return self.functions.values()

    def get_function(self, name: str) -> CallableFunction:
        return self.functions.get(name)


@description("Chat history")
def test(msg: str, value: int):
    return f"{msg} {value}"