import gzip
import random
import shutil
import tempfile
from os import path
from typing import Literal, Optional, Tuple

import mlx.core as mx
import requests

from mlx_esm.tokenizer import BatchTokenizer, Tokenizer

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
        current_label = label if label != "" else f"SEQ_{idx}"
      else:
        current_value_buf.append(line)

  return sequences


DataSplit = Literal["train", "validate"]


class Loader(object):
  def __init__(
    self,
    tokenizer: Tokenizer,
    dbs: list[int],
    batch_size: int,
    max_seq_len: int,
    mask_rate: float,
  ):
    self.dbs = sorted(dbs)
    self.batch_size = batch_size
    self.max_seq_len = max_seq_len
    self.mask_rate = mask_rate
    self.batch_tokenizer = BatchTokenizer(tokenizer)
    self.data: Optional[list[str]] = None
    self.train_validate_split = 0.9

  def load(self):
    if self.data is not None:
      return
    sequences = load_uniparc_dbs(self.dbs)
    self.data = [s for s in sequences if len(s) <= self.max_seq_len]
    random.shuffle(self.data)

  def next_batch(self, split: DataSplit) -> Tuple[mx.array, mx.array]:
    if self.data is None:
      raise Exception("data has not been loaded yet")

    split_idx = int(len(self.data) * self.train_validate_split)
    data = self.data[:split_idx] if split == "train" else self.data[split_idx:]

    batch: list[str] = random.sample(data, self.batch_size)

    encoded = self.batch_tokenizer.encode(batch)
    shape: list[int] = list(encoded.shape)

    tokenizer = self.batch_tokenizer.tokenizer
    pad_idx, mask_idx = tokenizer.pad_idx, tokenizer.mask_idx

    # We should not mask padding tokens because they do not form part of the
    # underlying protein sequence. We do include CLS & EOS tokens because they
    # are part of the sequence in so far that they do inform us about the structure
    # of the protein.
    can_mask = encoded != pad_idx

    # We should mask tokens with a probability of `mask_rate`. We will use a
    # uniform distribution to determine which tokens to mask. By multiplying
    # the result of the uniform distribution by `can_mask`, we ensure that we
    # do not mask tokens that are padding tokens.
    should_mask = (mx.random.uniform(0, 1, shape, dtype=mx.float32) < self.mask_rate) * can_mask
    should_not_mask = 1 - should_mask

    # BERT differs from ESM-1 in how it does masking. In ESM-1, we mask tokens
    # with the mask token only, while in BERT the masking strategy is a bit more
    # complex. We will implement the ESM-1 masking strategy here.
    masked = (encoded * should_not_mask) + (should_mask * mask_idx)

    return (masked, encoded)
