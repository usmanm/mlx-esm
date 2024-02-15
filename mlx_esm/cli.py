from datetime import datetime
from os import path
from typing import Optional

import click

from mlx_esm.data import Tokenizer
from mlx_esm.infer import generate, unmask
from mlx_esm.model import ESM1
from mlx_esm.train import Config, Trainer


def esm1_model() -> ESM1:
  return ESM1(Tokenizer())


@click.command("generate")
@click.option(
  "--weights-file",
  type=click.Path(exists=True, dir_okay=False, file_okay=True, readable=True),
  required=True,
)
@click.option("--length", type=int)
@click.option("--max-prob-only", is_flag=True)
def generate_cmd(
  weights_file: str,
  length: Optional[int] = None,
  max_prob_only: bool = False,
) -> None:
  """
  Generate a random protein sequence.
  """
  m = esm1_model()
  m.load_weights(weights_file)

  generate(m, length=length, max_prob_only=max_prob_only)


@click.command("unmask")
@click.option(
  "--weights-file",
  type=click.Path(exists=True, dir_okay=False, file_okay=True, readable=True),
  required=True,
)
@click.option("--seq", type=str, required=True)
@click.option("--max-prob-only", is_flag=True)
def unmask_cmd(weights_file: str, seq: str, max_prob_only: bool = False) -> None:
  """
  Unmask a masked protein sequence.
  """
  m = esm1_model()
  m.load_weights(weights_file)

  unmask(m, seq, max_prob_only=max_prob_only)


@click.command("train")
@click.option(
  "--weights-dir",
  type=click.Path(exists=True, dir_okay=True, file_okay=False),
)
@click.option(
  "--weights-file",
  type=click.Path(exists=True, dir_okay=False, file_okay=True),
)
def train_cmd(weights_dir: Optional[str], weights_file: Optional[str]) -> None:
  """
  Train a new/existing model and save/updates the weights in a file.
  """
  if (weights_dir and weights_file) or (not weights_dir and not weights_file):
    raise click.BadParameter("You must provide exactly one of --weights-dir and --weights-file.")

  m = esm1_model()
  if weights_file:
    m.load_weights(weights_file)

  c = Config()
  t = Trainer(m, c)

  t.load()
  t.train()
  t.validate()

  if weights_file:
    file_path = weights_file
    # Clear the file
    with open(file_path, "w"):
      pass
  else:
    now = datetime.now()
    time_str = now.strftime("%Y%m%d%H%M")
    file_path = path.join(f"{weights_dir}/{time_str}.npz")

  m.save_weights(file_path)
  print(f"💾 weights saved to {file_path}")


@click.group()
def main():
  pass


main.add_command(generate_cmd)
main.add_command(unmask_cmd)
main.add_command(train_cmd)

if __name__ == "__main__":
  main()
