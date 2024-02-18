import mlx.core as mx
import numpy as np
from tqdm.auto import tqdm

from mlx_esm.data import Tokenizer
from mlx_esm.model import ESM1


def generate(
  model: ESM1,
  length: int,
  max_iters: int = 256,
  max_prob_only: bool = False,
):
  start_seq = "^" + "*" * length + "$"
  return impl(model, start_seq, max_iters, max_prob_only)


def unmask(
  model: ESM1,
  masked_seq: str,
  max_iters: int = 256,
  max_prob_only: bool = False,
):
  return impl(model, f"^{masked_seq}$", max_iters, max_prob_only)


def impl(model: ESM1, input: str, max_iters: int, max_prob_only: bool):
  tokenizer = Tokenizer()

  model.eval()
  mx.eval(model.parameters())

  toks = tokenizer.encode(input)
  x = mx.array([toks], dtype=mx.int32)

  total = (toks == tokenizer.mask_idx).sum().item()
  loop = tqdm(
    total=total,
    ncols=120,
    desc="🌱 generating",
  )
  for _ in range(max_iters):
    toks = x[0]

    if is_sequence_legit(tokenizer, toks):
      break

    # forward the model
    logits = model(x)
    x = compute_next_x(tokenizer, x, logits, max_prob_only)
    loop.update()

  loop.close()

  emoji = "🌳" if is_sequence_legit(tokenizer, toks) else "🍂"
  s = "".join(tokenizer.decode(toks)).strip().rstrip("%").rstrip("$").lstrip("^")
  print(emoji + " hello world: " + s)


def compute_next_x(
  tokenizer: Tokenizer,
  x: mx.array,
  logits: mx.array,
  max_prob_only: bool = False,
) -> mx.array:
  probs = np.array(mx.softmax(logits, axis=-1))

  # This is equivalent to multinomial in PyTorch.
  if max_prob_only:
    samples = np.array([np.argmax(prob) for prob in probs.reshape(-1, probs.shape[-1])])
  else:
    samples = np.array(
      [
        np.random.choice(range(probs.shape[-1]), p=prob)
        for prob in probs.reshape(-1, probs.shape[-1])
      ]
    )

  sample = mx.array(samples.reshape(probs.shape[0], probs.shape[1]))

  # We only swap out the first mask token to generate proetins using
  # an autoregressive style. My theory is that this will lead to
  # more realistic sequences because the model sees it grow rather
  # than guessing the entire sequence in a single shot.
  mask_all = x == tokenizer.mask_idx
  mask_first = mx.zeros_like(mask_all)
  mask_first[mx.arange(mask_all.shape[0]), mask_all.argmax(axis=1)] = 1

  sample = sample * mask_first
  x = x * (1 - mask_first)

  return x + sample


def is_sequence_legit(tokenizer: Tokenizer, toks: mx.array) -> bool:
  tokens: list[int] = toks.tolist()

  # Any invalid tokens in the sequence?
  invalid_toks = [tokenizer.mask_idx, tokenizer.unk_idx]
  if any(invalid_tok in tokens for invalid_tok in invalid_toks):
    return False

  # Remove any padding
  while tokens and tokens[-1] == tokenizer.pad_idx:
    tokens.pop()

  # Does the sequence start and end with the correct tokens?
  if tokens[0] != tokenizer.cls_idx or tokens[-1] != tokenizer.eos_idx:
    return False

  # Are only protein tokens in middle of the sequence?
  protein_toks: list[int] = tokenizer.encode("".join(tokenizer.protein_toks)).tolist()
  return all(tok in protein_toks for tok in tokens[1:-1])
