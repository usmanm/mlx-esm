# mlx-esm

This is an implementation of Meta's [ESM-1](https://huggingface.co/docs/transformers/model_doc/esm) using Apple's [MLX](https://ml-explore.github.io/mlx/build/html/index.html) library.

## Backstory

I've been learning about deep learning and neural nets over the last few months. The two best teachers in this space are [Jeremy Howard](https://twitter.com/jeremyphoward) and [Andrej Karpathy](https://twitter.com/karpathy). Their intuitive understanding of neural nets and their ability to clearly articulate complex concepts are both incredible. To get started, watch these lectures (1.5x speed recommended):
- [Practical Deep Learning for Coders](https://course.fast.ai/)
- [Neural Networks: Zero to Hero](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)

Recently, I've been learning about [TechBio](https://www.nfx.com/post/biotech-to-techbio), and came across Meta's research papers:
- [Biological structure and function emerge from scalingunsupervised learning to 250 million protein sequences](https://www.pnas.org/doi/epdf/10.1073/pnas.2016239118)
- [Evolutionary-scale prediction of atomic level protein structure
with a language model](https://www.biorxiv.org/content/10.1101/2022.07.20.500902v3.full.pdf)

Like all of Meta's AI research, the architecture, source code and weights for these protein language models is [open-source](https://github.com/facebookresearch/esm).

I'd been wanting to implement and train neural nets which are more than a toy example. So, I'm going to reimplement ESM-1 from the research paper, but in MLX. I'll use the ESM-1 PyTorch [implementation](https://github.com/facebookresearch/esm/blob/main/esm/model/esm1.py) and Bert MLX [implementation](https://github.com/ml-explore/mlx-examples/blob/main/bert/model.py) as a reference. Figured this will provide enough copy-pasta that I can do this quickly, but the tinkering needed will me understand of the low-level concepts of building neural net architectures. ESM-1 is a fork of the [BERT](https://huggingface.co/docs/transformers/model_doc/bert) language model and uses the [masked language modeling](https://huggingface.co/docs/transformers/main/tasks/masked_language_modeling) objective.

## Hacking It

Generally followed Karpathy's approach:
- Data loading and tokenizing first
- Build the training loop with a noop model
- Build the neural net module-by-module TODO
- Use `.shape()` generously to debug dimensionality issues of tensors

Though, I ended up using VS Code + Terminal more, because notebooks have never been my DX and I wanted to use ChatGPT, which is really fucking good for asking questions. It's knowledge of `mlx` is rather not great, but it can understand questions and answer in `pytorch` equivalent code. Converting from `mlx` to `pytorch` is fairly straightforward, 90% of stuff matches 1:1 to `pytorch`, the remainder is JAX inspired.

## Take Aways

Open-source ML feels like it's in a really dope spot. We have suddenly unlocked this ability to train really large neural nets, thanks to Transformers and Nvidia. We have both the compute and an architecture that scales with compute. Meta's decision to open-source their AI work allows anyone really to start playing around with these models. 

Neural net architectures have a lego block type feel. They're made of "modules" wrapped behind common interface making them composable. Composing them together in code isn't super straight-forward though. I believe this is because they use a mathematical structure called [tensors](https://en.wikipedia.org/wiki/Tensor) (think of it as a N-dimensional matrix) to talk to each other. I wish I'd taken some linear algebra courses in college. It would be nice to find a more programming intuitive data structure instead of tensors.

```
➜ poetry run cli train --weights-dir=./weights
📥 loading data
🚂 training: 100%|████████████████████████████████████████████████████████████| 100000/100000 [1:44:43<00:00, 15.91it/s, loss=0.2758]
🔍 validating: 100%|████████████████████████████████████████████████████████████████| 10000/10000 [09:27<00:00, 17.63it/s, loss=0.2766]
💾 weights saved to ./weights/202402151405.npz

➜  poetry run cli generate --weights-file=./weights/202402151405.npz
🌱 generating: 100%|███████████████████████████████████████████████████████████████████| 67/67 [00:00<00:00, 311.70it/s]
🌳 hello world: RTAAEVGGGHPAGPGGRAEPQVAFGAGDRPGGGRKPYGGGSVSPQAGVQVCTAIYGVTHGAWRLPDK

➜  poetry run cli unmask --weights-file=./weights/202402151405.npz --seq="MRAGRGGVPGSGGLRAPPPPLL***LAMLPAAAPRSPALAAAPAGPSVSLYLSEDEVRRLL"
🌱 generating: 100%|█████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 170.82it/s]
🌳 hello world: MRAGRGGVPGSGGLRAPPPPLLAAALAMLPAAAPRSPALAAAPAGPSVSLYLSEDEVRRLL


>SequenceOne
AMDGMAGAGSTDAQAVAFVGEEAVAIALAVRAAIAARGA
>SequenceTwo
DMPVARGRNRSQTARGAQREIRQANSRAETGRVTIATERWAEASVDRSDEPADQEVQALRYAQQNVGWWLPSGSGAAQAGSRPAS
>SequenceThree
MKEVKERVPARSADDSLGVGVVEKIAAKARALEAKPRGAYHGIITVDTVTISTGLN
>SequenceFour
AMGIAAGLLERVAGDASYGGGVAVSQPWAIGGLAGTYERLASAVVRCTGEDEPLDVPIKRPRRRREVTEPRAAIPDIVQREREVRKRSEQQLGFRRALVTGTRVKGGTEFRLDCVGSEERIEVVGV
```

I ran these sequences through AlphaFold 2 to predict their molecular structure. The structure comes out in `pdb` files, which I assume are named after the [Protein World Bank](https://en.wikipedia.org/wiki/Protein_Data_Bank). Next was to figure out how to render these structures. I found [3Dmol.js](https://3dmol.csb.pitt.edu/), a free JavaScript library for visualizing molecular data which has [Python bindings](https://pypi.org/project/py3Dmol/) for notebooks. Using it is pretty straight forward, [here's](https://colab.research.google.com/drive/19RXVjA5BcnuyVTVsP6g2Ldx0hZO2VQ63?usp=sharing) a Colab notebook with reference code I used.
