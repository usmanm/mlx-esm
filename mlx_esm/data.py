import gzip
import shutil
import tempfile
from os import path

import requests

DATA_DIR = path.join(path.dirname(__file__), path.pardir, "data")
UNIPARC_DIR_URL = "https://ftp.uniprot.org/pub/databases/uniprot/current_release/uniparc/fasta/active/"


def download_file(url: str, path: str) -> None:
  response = requests.get(url)
  response.raise_for_status()

  with open(path, "wb") as f:
    f.write(response.content)


def extract_gz_file(gz_path: str, dest_path: str):
  with gzip.open(gz_path, "rb") as f_in, open(dest_path, "wb") as f_out:
    shutil.copyfileobj(f_in, f_out)


class Sequence(object):
  def __init__(self, label: str, value: str) -> None:
    self.label = label
    self.value = value

  def __repr__(self) -> str:
    visible_label = self.label.split(" ")[0]
    return f"Sequence({visible_label}, {self.value[:15]}...)"


class SequenceDataset(object):
  def __init__(self, sequences: list[Sequence]) -> None:
    self.sequences = sequences

  def __len__(self) -> int:
    return len(self.sequences)

  def __getitem__(self, idx: int) -> Sequence:
    return self.sequences[idx]

  @staticmethod
  def from_uniparc(id: int):
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
