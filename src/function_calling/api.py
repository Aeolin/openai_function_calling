import openai
from function_calling.models import ChatHistory, ChatRole, ChatMessage
from typing import List, Any, Callable
import sys


def setup_api(api_key: str, string_transformer: str):
    openai.api_key = api_key
    sys.modules[__name__].string_transformer = string_transformer


def get_chat_completion(history: ChatHistory, selector: Callable[[List[Any]], Any] = None, **kwargs) -> ChatMessage:
    selector = selector or (lambda x: x[0])
    completion = openai.ChatCompletion.create(
        model=history.model,
        messages=[x.to_dict() for x in history.messages],
        functions=[x.to_dict() for x in history.get_functions()],
        function_call=kwargs.get('function_call', 'auto' if len(history.functions) > 0 else 'none'),
        max_tokens=kwargs.get('max_tokens', 1024),
        temperature=kwargs.get('temperature', 0.9),
        top_p=kwargs.get('top_p', 1.0),
        n=kwargs.get('n', 1),
        stream=False
    )

    msg = selector(completion['choices'])['message']
    function_call = msg.get('function_call')
    if function_call:
        name = function_call['name']
        arguments = function_call['arguments']
        result = history.get_function(name).call(arguments)
        if not isinstance(result, str):
            result = sys.modules[__name__].string_transformer(result)

        history.add_function_result(name, result)
        return get_chat_completion(history, selector, **kwargs)
    else:
        tokens = completion['usage']['completion_tokens'] if len(completion['choices']) > 0 else len(history.tokenizer.encode(msg['content']))
        total_tokens = completion['usage']['total_tokens']
        result = ChatMessage(msg['content'], ChatRole.ASSISTANT, tokens, True)
        history.add_response_message(result, total_tokens)
        return result

