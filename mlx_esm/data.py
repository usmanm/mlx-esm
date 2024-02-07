import gzip
import shutil
import tempfile
from os import path
from typing import Tuple

import requests

DATA_DIR = path.join(path.dirname(__file__), path.pardir, "data")
UNIPARC_DIR_URL = "https://ftp.uniprot.org/pub/databases/uniprot/current_release/uniparc/fasta/active/"


def download_file(url: str, path: str):
  response = requests.get(url)
  response.raise_for_status()

  with open(path, "wb") as f:
    f.write(response.content)


def extract_gz_file(gz_path: str, dest_path: str):
  with gzip.open(gz_path, "rb") as f_in, open(dest_path, "wb") as f_out:
    shutil.copyfileobj(f_in, f_out)


class Sequence(object):
  def __init__(self, label: str, value: str):
    self.label = label
    self.value = value

  def __repr__(self) -> str:
    visible_label = self.label.split(" ")[0]
    return f"Sequence({visible_label}, {self.value[:15]}...)"


class SequenceDataset(object):
  def __init__(self, data: list[Sequence]):
    self.data = data

  def __iter__(self):
    self.index = 0
    return self

  def __next__(self):
    if self.index >= len(self.data):
      raise StopIteration
    result = self.data[self.index]
    self.index += 1
    return result

  def __len__(self) -> int:
    return len(self.data)

  def __getitem__(self, idx: int) -> Sequence:
    return self.data[idx]

  @staticmethod
  def for_uniparc_db(id: int) -> "SequenceDataset":
    assert 0 < id <= 200

    filename = "uniparc_active_p%d.fasta" % id
    filepath = path.join(DATA_DIR, filename)

    if not path.exists(filepath):
      url = path.join(UNIPARC_DIR_URL, f"{filename}.gz")
      with tempfile.NamedTemporaryFile() as tmp:
        download_file(url, tmp.name)
        extract_gz_file(tmp.name, filepath)

    with open(filepath, "r") as f:
      sequences: list[Sequence] = []

      current_label = ""
      current_value_buf: list[str] = []

      def _flush_sequence():
        nonlocal current_label
        if current_label == "":
          return
        sequences.append(Sequence(current_label, "".join(current_value_buf)))
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

    return SequenceDataset(sequences)


class Tokenizer(object):
  def __init__(self):
    self.prepend_toks: Tuple[str, ...] = ("<^>", "<.>", "<$>", "<?>")
    self.append_toks: Tuple[str, ...] = ("<*>",)
    self.protein_toks: Tuple[str, ...] = (
      "L",
      "A",
      "G",
      "V",
      "S",
      "E",
      "R",
      "T",
      "I",
      "D",
      "P",
      "K",
      "Q",
      "N",
      "F",
      "Y",
      "M",
      "H",
      "W",
      "C",
      "X",
      "B",
      "U",
      "Z",
      "O",
      ".",
      "-",
    )
    self.all_toks: list[str] = sorted(list(self.prepend_toks) + list(self.append_toks) + list(self.protein_toks))
    assert len(self.all_toks) == len(set(self.all_toks))

    self.idx_to_tok = dict(enumerate(self.all_toks))
    self.tok_to_idx = {tok: idx for idx, tok in enumerate(self.all_toks)}
    self.num_vocab = len(self.all_toks)

    self.unk_idx = self.tok_to_idx["<?>"]
    self.pad_idx = self.tok_to_idx["<.>"]
    self.cls_idx = self.tok_to_idx["<^>"]
    self.mask_idx = self.tok_to_idx["<*>"]
    self.eos_idx = self.tok_to_idx["<$>"]

  def tokenize(self, sequence: Sequence) -> list[str]:
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

      return [tok if tok in toks else tok.strip() for tok in curr_tokens]

    return split_on_tokens(self.all_toks, sequence.value.strip())

  def encode(self, toks: list[str]) -> list[int]:
    return [self.tok_to_idx.get(tok, self.unk_idx) for tok in toks]

  def decode(self, encoded: list[int]) -> list[str]:
    return [self.idx_to_tok[idx] for idx in encoded]
