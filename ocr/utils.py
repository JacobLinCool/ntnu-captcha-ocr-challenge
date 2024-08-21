import torch

char_dict = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "-": 10,
    "+": 11,
    "=": 12,
    "a": 13,
    "b": 14,
    "c": 15,
    "d": 16,
    "e": 17,
    "f": 18,
    "g": 19,
    "h": 20,
    "i": 21,
    "j": 22,
    "k": 23,
    "l": 24,
    "m": 25,
    "n": 26,
    "o": 27,
    "p": 28,
    "q": 29,
    "r": 30,
    "s": 31,
    "t": 32,
    "u": 33,
    "v": 34,
    "w": 35,
    "x": 36,
    "y": 37,
    "z": 38,
}

char_dict_rev = {v: k for k, v in char_dict.items()}


def text_to_tensor(text: str, max_length: int = 4) -> torch.Tensor:
    tensor = torch.zeros(max_length, dtype=torch.long)
    for i, c in enumerate(text):
        tensor[i] = char_dict[c]
    return tensor


def texts_to_tensors(texts: list[str], max_length: int = 4) -> torch.Tensor:
    tensors = torch.zeros(len(texts), max_length, dtype=torch.long)
    for i, text in enumerate(texts):
        tensors[i] = text_to_tensor(text, max_length)
    return tensors


def tensor_to_text(tensor: torch.Tensor) -> str:
    text = ""
    for i in tensor:
        text += char_dict_rev[torch.argmax(i).item()]
    return text


def tensors_to_texts(tensors: torch.Tensor) -> list[str]:
    texts = []
    for tensor in tensors:
        texts.append(tensor_to_text(tensor))
    return texts
