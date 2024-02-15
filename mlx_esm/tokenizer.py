import math
from typing import Tuple

import mlx.core as mx


class Tokenizer(object):
  def __init__(self):
    # Special tokens: start of sequence, masked token, end of sequence, unknown token, padding token
    self.special_toks: Tuple[str, ...] = ("^", "*", "$", "?", "%")
    self.protein_toks: Tuple[str, ...] = (
      "-",
      ".",
      "A",
      "B",
      "C",
      "D",
      "E",
      "F",
      "G",
      "H",
      "I",
      "K",
      "L",
      "M",
      "N",
      "O",
      "P",
      "Q",
      "R",
      "S",
      "T",
      "U",
      "V",
      "W",
      "X",
      "Y",
      "Z",
    )
    self.all_toks: list[str] = sorted(list(self.special_toks) + list(self.protein_toks))

    # Make padding_idx always be 0. This allows us to keep positional embeddings simple.
    tmp_idx = self.all_toks.index("%")
    zero_val = self.all_toks[0]
    self.all_toks[0] = "%"
    self.all_toks[tmp_idx] = zero_val

    assert self.all_toks[0] == "%"
    assert len(self.all_toks) == len(set(self.all_toks))

    self.idx_to_tok = dict(enumerate(self.all_toks))
    self.tok_to_idx = {tok: idx for idx, tok in enumerate(self.all_toks)}
    self.vocab_size = len(self.all_toks)

    self.unk_idx = self.tok_to_idx["?"]
    self.pad_idx = self.tok_to_idx["%"]
    self.cls_idx = self.tok_to_idx["^"]
    self.mask_idx = self.tok_to_idx["*"]
    self.eos_idx = self.tok_to_idx["$"]

    assert self.pad_idx == 0

  def tokenize(self, sequence: str) -> list[str]:
    # Example:
    # -> split_on_token("H", "XYXHADHKJXXX")
    # -> ['XYX', 'H', 'AD', 'H', 'KJXXX']
    def split_on_token(tok: str, text: str) -> list[str]:
      result: list[str] = []
      split_text = text.split(tok)
      for i, sub_text in enumerate(split_text):
        sub_text = sub_text.strip()

        if i == len(split_text) - 1:
          if sub_text:
            result.append(sub_text)
        else:
          if sub_text:
            result.append(sub_text)
          result.append(tok)

      assert text == "".join(result)
      assert all(s == tok or tok not in s for s in result)

      return result

    # Example:
    # -> split_on_tokens(["H", "Y"], "XYXHADHKJXXX")
    # -> ['X', 'Y', 'X', 'H', 'AD', 'H', 'KJXXX']
    def split_on_tokens(toks: list[str], text: str) -> list[str]:
      if text == "":
        return []

      curr_tokens: list[str] = [text]
      next_tokens: list[str] = []

      for tok in toks:
        for sub_text in curr_tokens:
          if sub_text not in toks:
            next_tokens.extend(split_on_token(tok, sub_text))
          else:
            next_tokens.append(sub_text)
        curr_tokens = next_tokens
        next_tokens = []

      return curr_tokens

    return split_on_tokens(self.all_toks, sequence.strip())

  def encode(self, seq: str) -> mx.array:
    toks = self.tokenize(seq)
    # If token is not present, treat it as "unknown" token.
    return mx.array([self.tok_to_idx.get(tok, self.unk_idx) for tok in toks], dtype=mx.int32)

  def decode(self, encoded: mx.array) -> list[str]:
    return [self.idx_to_tok[idx] for idx in encoded.tolist()]


class BatchTokenizer(object):
  def __init__(self, tokenizer: Tokenizer):
    self.tokenizer = tokenizer

  def encode(self, sequences: list[str]) -> mx.array:
    tokenizer = self.tokenizer

    batch_size = len(sequences)
    batch = [tokenizer.encode(seq) for seq in sequences]

    max_tok_len = max(len(toks) for toks in batch)
    # +2 for CLS and EOS tokens. Make it a multiple of 8 for performance.
    seq_len = math.ceil((max_tok_len + 2) / 8) * 8

    # B = size of batch, L = sequence length
    shape = (batch_size, seq_len)

    # (B, L) -> filled with padding token
    tokens = mx.full(shape, tokenizer.pad_idx, dtype=mx.int32)

    # Fill the tokens tensor with the actual protein sequence tokens, making
    # sure to add a "start of sequence" token at the beginning and an "end of
    # sequence" token at the end. We are using a "pad-right" strategy, because
    # BERT is a model with absolute position embeddings so it’s usually advised
    # to pad the inputs on the right rather than the left.
    #
    # https://huggingface.co/docs/transformers/model_doc/bert#usage-tips
    for idx, toks in enumerate(batch):
      # First token of each sequence is the "start of sequence" token.
      tokens[idx, 0] = tokenizer.cls_idx
      # Then fill in the actual protein sequence tokens.
      tokens[idx, 1 : len(toks) + 1] = mx.array(toks)
      # Finally, the last token of each sequence is the "end of sequence" token.
      tokens[idx, len(toks) + 1] = tokenizer.eos_idx

    return tokens
