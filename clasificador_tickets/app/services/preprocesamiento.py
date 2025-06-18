def prepro_text(text: str) -> str:
    return text.strip().replace("\n", " ").lower()