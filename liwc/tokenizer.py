import re

def get_all_tokens(text:str) -> list:
    return re.findall(r"\w+(?:[-']\w+)*|\"|[-.(]+|\S\w*", text)

def get_word_only_token(text:str) -> list:
    all_tokens = get_all_tokens(text)
    word_pattern = re.compile(r"\w+(?:[-']\w+)*|\S\w+")
    
    return [token.lower() for token in all_tokens if word_pattern.match(token)]