import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
pooling_type_dict = {  # cls or mean
    "BAAI/bge-base-en-v1.5": "cls",
    "default": "mean",
}
api_list = {
    "local": {
        "OPENAI_BASE_URL": "http://localhost:8000/v1/",
        "OPENAI_MODEL": "",
        "OPENAI_API_KEY": "PLACEHOLDER",
    },
    "openai": {
        "OPENAI_BASE_URL": "https://api.openai.com/v1/",
        "OPENAI_MODEL": "gpt-4o-2024-08-06",
        "OPENAI_API_KEY": "PLACEHOLDER",
    },
    "gpt-4o": {
        "OPENAI_BASE_URL": "https://api.openai.com/v1/",
        "OPENAI_MODEL": "gpt-4o-2024-08-06",
        "OPENAI_API_KEY": "PLACEHOLDER",
    },
}
