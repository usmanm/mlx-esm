import gzip
import shutil
import tempfile
from os import path
from typing import Tuple

import mlx.core as mx
import requests

DATA_DIR = path.join(path.dirname(__file__), path.pardir, "data")
UNIPARC_DIR_URL = (
  "https://ftp.uniprot.org/pub/databases/uniprot/current_release/uniparc/fasta/active/"
)


def download_file(url: str, path: str):
  response = requests.get(url)
  response.raise_for_status()

  with open(path, "wb") as f:
    f.write(response.content)


def extract_gz_file(gz_path: str, dest_path: str):
  with gzip.open(gz_path, "rb") as f_in, open(dest_path, "wb") as f_out:
    shutil.copyfileobj(f_in, f_out)


def load_uniparc_dbs(ids: list[int]) -> list[str]:
  return [item for _id in ids for item in load_uniparc_db(_id)]


def load_uniparc_db(_id: int) -> list[str]:
  assert 0 < _id <= 200

  filename = "uniparc_active_p%d.fasta" % _id
  filepath = path.join(DATA_DIR, filename)

  if not path.exists(filepath):
    url = path.join(UNIPARC_DIR_URL, f"{filename}.gz")
    with tempfile.NamedTemporaryFile() as tmp:
      download_file(url, tmp.name)
      extract_gz_file(tmp.name, filepath)

  # Example:
  #
  # >UPI0000000563 status=active
  # MSGHKCSYPWDLQDRYAQDKSVVNKMQQKYWETKQAFIKATGKKEDEHVVASDADLDAKL
  # ELFHSIQRTCLDLSKAIVLYQKRICSF
  # >UPI00000005DE status=active
  # MGAQDRPQCHFDIEINREPVGRIMFQLFSDICPKTCKNFLCLCSGEKGLGKTTGKKLCYK
  # GSTFHRVVKNFMIQGGDFSEGNGKGGESIYGGYFKDENFILKHDRAFLLSMANRGKHTNG
  # SQFFITTKPAPHLDGVHVVFGLVISGFEVIEQIENLKTDAASRPYADVRVIDCGVLATKL
  # TKDVFEKKRKKPTCSEGSDSSSRSSSSSESSSESEVERETIRRRRHKRRPKVRHAKKRRK
  # EMSSSEEPRRKRTVSPEG
  with open(filepath, "r") as f:
    sequences: list[str] = []

    current_label = ""
    current_value_buf: list[str] = []

    def _flush_sequence():
      nonlocal current_label
      if current_label == "":
        return
      sequences.append("".join(current_value_buf))
      current_label = ""
      current_value_buf.clear()

    for idx, line in enumerate(f):
      line = line.strip()

      if line.startswith(">"):
        _flush_sequence()
        label = line[1:].strip()
        current_label = label if label != "" else f"sequence:{idx}"
      else:
        current_value_buf.append(line)

  return sequences


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
    assert len(self.all_toks) == len(set(self.all_toks))

    self.idx_to_tok = dict(enumerate(self.all_toks))
    self.tok_to_idx = {tok: idx for idx, tok in enumerate(self.all_toks)}
    self.vocab_size = len(self.all_toks)

    self.unk_idx = self.tok_to_idx["?"]
    self.pad_idx = self.tok_to_idx["%"]
    self.cls_idx = self.tok_to_idx["^"]
    self.mask_idx = self.tok_to_idx["*"]
    self.eos_idx = self.tok_to_idx["$"]

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
  def __init__(self, context_size: int = 128, dynamic_padding: bool = False):
    self.tokenizer = Tokenizer()
    self.context_size = context_size
    self.dynamic_padding = dynamic_padding

  def encode(self, sequences: list[str]) -> mx.array:
    tokenizer = self.tokenizer

    batch_size = len(sequences)
    tokenized_batch = [tokenizer.encode(seq) for seq in sequences]

    # Truncate the tokenized sequences such that they fit within the
    # context. For that, we need to subtract 2 to account for the CLS
    # and EOS tokens.
    max_trunc_len = self.context_size - 2

    if self.dynamic_padding:
      max_token_len = max(len(toks) for toks in tokenized_batch)
      trunc_len = min(max_trunc_len, max_token_len)
    else:
      trunc_len = max_trunc_len

    truncated_tokenized_batch = [toks[:trunc_len] for toks in tokenized_batch]

    # B = size of batch, L = max length of sequence in batch + CLS + EOS
    shape = (batch_size, trunc_len + 2)

    # (B x L) -> filled with padding token
    tokens = mx.full(shape, tokenizer.pad_idx, dtype=mx.int32)

    # Fill the tokens tensor with the actual protein sequence tokens, making
    # sure to add a "start of sequence" token at the beginning and an "end of
    # sequence" token at the end. We are using a "pad-right" strategy, because
    # BERT is a model with absolute position embeddings so it’s usually advised
    # to pad the inputs on the right rather than the left.
    #
    # https://huggingface.co/docs/transformers/model_doc/bert#usage-tips
    for idx, toks in enumerate(truncated_tokenized_batch):
      # First token of each sequence is the "start of sequence" token.
      tokens[idx, 0] = tokenizer.cls_idx
      # Then fill in the actual protein sequence tokens.
      tokens[idx, 1 : len(toks) + 1] = mx.array(toks)
      # Finally, the last token of each sequence is the "end of sequence" token.
      tokens[idx, len(toks) + 1] = tokenizer.eos_idx

    return tokens
