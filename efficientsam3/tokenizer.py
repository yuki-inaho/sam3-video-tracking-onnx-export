"""Torch-free CLIP BPE, following upstream tokenizer_ve.py (Meta/OpenCLIP)."""

import html
import json

import ftfy
import numpy as np
import regex

from contract import CONTEXT


class Tokenizer:
    def __init__(self, path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.encoder = data["encoder"]
        self.ranks = {tuple(pair): rank for rank, pair in enumerate(data["merges"])}
        values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        for byte in range(256):
            if byte not in values:
                values.append(byte)
        # Assign missing bytes consecutively from U+0100, as CLIP does.
        next_code = 256
        byte_encoder = {}
        for byte in values:
            if 33 <= byte <= 126 or 161 <= byte <= 172 or 174 <= byte <= 255:
                byte_encoder[byte] = chr(byte)
            else:
                byte_encoder[byte] = chr(next_code)
                next_code += 1
        self.byte_encoder = byte_encoder
        self.pattern = regex.compile(
            r"<start_of_text>|<end_of_text>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+",
            regex.IGNORECASE,
        )
        self.cache = {}

    def bpe(self, word):
        if word in self.cache:
            return self.cache[word]
        parts = tuple(word[:-1]) + (word[-1] + "</w>",)
        while len(parts) > 1:
            pair = min(zip(parts, parts[1:]), key=lambda p: self.ranks.get(p, float("inf")))
            if pair not in self.ranks:
                break
            merged = []
            i = 0
            while i < len(parts):
                if i + 1 < len(parts) and (parts[i], parts[i + 1]) == pair:
                    merged.append(parts[i] + parts[i + 1])
                    i += 2
                else:
                    merged.append(parts[i])
                    i += 1
            parts = tuple(merged)
        self.cache[word] = parts
        return parts

    def __call__(self, text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Text prompt must be a nonempty string")
        clean = " ".join(html.unescape(html.unescape(ftfy.fix_text(text))).split()).lower()
        ids = [self.encoder["<start_of_text>"]]
        for token in self.pattern.findall(clean):
            word = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            ids.extend(self.encoder[p] for p in self.bpe(word))
        ids = ids[: CONTEXT - 1] + [self.encoder["<end_of_text>"]]
        result = np.zeros((1, CONTEXT), dtype=np.int64)
        result[0, : len(ids)] = ids
        return result
