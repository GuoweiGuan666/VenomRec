class CharTokenizer:
    name = "char"

    def encode(self, text: str):
        return list(text or "")

    def decode(self, tokens):
        # tokens are single-character strings
        return "".join(tokens or [])


__all__ = ["CharTokenizer"]