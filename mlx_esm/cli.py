from datetime import datetime
from enum import Enum
from os import path
from typing import Optional

import click

from mlx_esm.infer import generate, unmask
from mlx_esm.model import ESM1, MLP, Base
from mlx_esm.train import Config, Trainer


class ModelName(Enum):
  MLP = "mlp"
  ESM1 = "esm1"


def get_model(name: str) -> Base:
  v = ModelName[name.upper()]
  if v == ModelName.MLP:
    return MLP()
  elif v == ModelName.ESM1:
    return ESM1()
  else:
    raise KeyError(f"unknown model name: {name}")


@click.command("generate")
@click.option(
  "--model",
  type=click.Choice([e.value for e in ModelName], case_sensitive=False),
  required=True,
)
@click.option(
  "--weights-file",
  type=click.Path(exists=True, dir_okay=False, file_okay=True, readable=True),
  required=True,
)
def generate_cmd(model: str, weights_file: str) -> None:
  """
  Generate a random protein sequence.
  """
  m = get_model(model)
  m.load_weights(weights_file)

  generate(m)


@click.command("unmask")
@click.option(
  "--model",
  type=click.Choice([e.value for e in ModelName], case_sensitive=False),
  required=True,
)
@click.option(
  "--weights-file",
  type=click.Path(exists=True, dir_okay=False, file_okay=True, readable=True),
  required=True,
)
@click.argument("input")
def unmask_cmd(model: str, weights_file: str, input: str) -> None:
  """
  Unmask a masked protein sequence.
  """
  m = get_model(model)
  m.load_weights(weights_file)

  unmask(m, input)


@click.command("train")
@click.option(
  "--model",
  type=click.Choice([e.value for e in ModelName], case_sensitive=False),
  required=True,
)
@click.option(
  "--weights-dir",
  type=click.Path(exists=True, dir_okay=True, file_okay=False),
)
@click.option(
  "--weights-file",
  type=click.Path(exists=True, dir_okay=False, file_okay=True),
)
def train_cmd(model: str, weights_dir: Optional[str], weights_file: Optional[str]) -> None:
  """
  Train a new/existing model and save/updates the weights in a file.
  """
  if (weights_dir and weights_file) or (not weights_dir and not weights_file):
    raise click.BadParameter("You must provide exactly one of --weights-dir and --weights-file.")

  m = get_model(model)
  if weights_file:
    m.load_weights(weights_file)

  c = Config()
  t = Trainer(m, c)
  t.load()
  t.run()

  if weights_file:
    file_path = weights_file
    # Clear the file
    with open(file_path, "w"):
      pass
  else:
    now = datetime.now()
    time_str = now.strftime("%Y%m%d%H%M")
    file_path = path.join(f"{weights_dir}/{model}-{time_str}.npz")

  m.save_weights(file_path)

  print(f"💾 model weights saved to {file_path}")


@click.group()
def main():
  pass


main.add_command(generate_cmd)
main.add_command(unmask_cmd)
main.add_command(train_cmd)

if __name__ == "__main__":
  main()
