# mlx-esm

This is an implementation of Meta's [ESM-1](https://huggingface.co/docs/transformers/model_doc/esm) using Apple's [MLX](https://ml-explore.github.io/mlx/build/html/index.html) library.

## Backstory

I've been learning about deep learning and neural nets over the last few months. The two best teachers in this space are [Jeremy Howard](https://twitter.com/jeremyphoward) and [Andrej Karpathy](https://twitter.com/karpathy). Their intuitive understanding of neural nets and their ability to articulate complex concepts are both incredible. To get started, watch these lectures (1.5x speed recommended):
- [Practical Deep Learning for Coders](https://course.fast.ai/)
- [Neural Networks: Zero to Hero](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)

I have also been reading up on [TechBio](https://www.nfx.com/post/biotech-to-techbio), and came across Meta's research papers:
- [Biological structure and function emerge from scalingunsupervised learning to 250 million protein sequences](https://www.pnas.org/doi/epdf/10.1073/pnas.2016239118)
- [Evolutionary-scale prediction of atomic level protein structure
with a language model](https://www.biorxiv.org/content/10.1101/2022.07.20.500902v3.full.pdf)

Like all of Meta's AI research, the architecture, source code and weights for these protein language models is [open-source](https://github.com/facebookresearch/esm).

I'd been wanting to implement and train a neural net which are more than a toy example. So, I'm decided to reimplement ESM-1 from the research paper, but in MLX. ESM-1 is a fork of the [BERT](https://huggingface.co/docs/transformers/model_doc/bert) language model and uses the [masked language modeling](https://huggingface.co/docs/transformers/main/tasks/masked_language_modeling) objective.

I used the ESM-1 PyTorch [implementation](https://github.com/facebookresearch/esm/blob/main/esm/model/esm1.py) and Bert MLX [implementation](https://github.com/ml-explore/mlx-examples/blob/main/bert/model.py) as a reference. I figured this will provide enough copy-pasta that I can do this quickly, but the tinkering needed to translate PyTorch to MLX will also expose to me some low-level concepts of building neural net architectures. 

## Hacking It

I generally followed Karpathy's approach:
- Data loading and tokenizing first. You should have a way to quickly get batches of input tensors to run your model on. Always set the seed to a constant so everything is reproducible.
- Build the training loop with a noop model. Include any helpful logging and plotting that you'll need to make sure when you run the real model, you can diagnose bugs quickly.
- Build the neural net one layer at a time. Typically, you want to start from the input embedding layer and go "deeper" into the neural net. At each layer, inspect the input and output tensor shapes to make sure the layer is doing what you expect it to do.
- Use `.shape()` generously to debug dimensionality issues of tensors. Libraries like PyTorch have magical reshaping capabilities, which mostly just works out of the most. Sometimes though when reshaping, transposing or squeezing you may want to test and validate with a simple input tensor.

Since I haven't really used notebooks much before, my development flow was more around VS Code + Terminal. I also finally understood why people love Github Copilot. It is really fucking good when you're not an expert and need help with explaining concepts, code and debugging. It's knowledge of `mlx` is not great, but it can understand questions and answer in `pytorch` equivalent code. Converting from `mlx` to `pytorch` is fairly straightforward, 90% of the API matches 1:1 with `pytorch`, the remainder is (I think) JAX inspired.

## Trying It Out

This project uses [Poetry](https://python-poetry.org/) to manage dependencies, so make sure to install it on your system first. Start by cloning the repository and installing all dependencies.

```
git clone git@github.com:usmanm/mlx-esm.git
cd mlx-esm
poetry install
```

You can now train your own ESM1 model. The training script will download [UniParc](https://www.uniprot.org/help/uniparc) dataset. By default, the script will train on only the first 3 partitions for 100K iterations. You can use `--num-iters` and `--dataset-partitions` CLI options to tweak these training parameters. On my Macbook Air M2, training with these default parameters took about 1 hour and 41 minutes. You can also skip this step and just use the weights from my training run directly for inference.

```
➜ poetry run cli train --weights-dir=./weights
📥 loading data
🚂 training: 100%|████████████████████████████████████████████████████████████| 100000/100000 [1:44:43<00:00, 15.91it/s, loss=0.2758]
🔍 validating: 100%|████████████████████████████████████████████████████████████| 10000/10000 [09:27<00:00, 17.63it/s, loss=0.2766]
💾 weights saved to ./weights/esm1-202402151405.npz
```

There are two inference modes:
- `generate`: This generates a new protein from scratch in an auto-regressive manner. You can specify `--length` to control the size of the protein. By default, a random length from the range `[32, 96)` will be picked.
- `unmask`: This takes a masked proteins sequence (some amino acids using a `*` character) and replaces the masked token with an amino acid.

```
➜  poetry run cli generate --weights-file=./weights/202402151405.npz
🌱 generating: 100%|████████████████████████████████████████████████████████████| 67/67 [00:00<00:00, 311.70it/s]
🌳 hello world: RTAAEVGGGHPAGPGGRAEPQVAFGAGDRPGGGRKPYGGGSVSPQAGVQVCTAIYGVTHGAWRLPDK

➜  poetry run cli unmask --weights-file=./weights/202402151405.npz --seq="MRAGRGGVPGSGGLRAPPPPLL***LAMLPAAAPRSPALAAAPAGPSVSLYLSEDEVRRLL"
🌱 generating: 100%|████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 170.82it/s]
🌳 hello world: MRAGRGGVPGSGGLRAPPPPLLAAALAMLPAAAPRSPALAAAPAGPSVSLYLSEDEVRRLL
```

Given, my GPU poor home infra, I only trained a small model (800K parameters) with ~1.5% of the protein dataset. But you can still do some cool stuff with it. I created a [FASTQ](https://knowledge.illumina.com/software/general/software-general-reference_material-list/000002211) file of 4 random proteins my model generated.

```
>SequenceOne
AMDGMAGAGSTDAQAVAFVGEEAVAIALAVRAAIAARGA
>SequenceTwo
DMPVARGRNRSQTARGAQREIRQANSRAETGRVTIATERWAEASVDRSDEPADQEVQALRYAQQNVGWWLPSGSGAAQAGSRPAS
>SequenceThree
MKEVKERVPARSADDSLGVGVVEKIAAKARALEAKPRGAYHGIITVDTVTISTGLN
>SequenceFour
AMGIAAGLLERVAGDASYGGGVAVSQPWAIGGLAGTYERLASAVVRCTGEDEPLDVPIKRPRRRREVTEPRAAIPDIVQREREVRKRSEQQLGFRRALVTGTRVKGGTEFRLDCVGSEERIEVVGV
```

I ran these sequences through [AlphaFold](https://github.com/google-deepmind/alphafold) to predict their molecular structure. The structure comes out in `pdb` files, which I assume are named after the [Protein World Bank](https://en.wikipedia.org/wiki/Protein_Data_Bank). Next I had to figure out how to render these 3D structures. I found [3Dmol.js](https://3dmol.csb.pitt.edu/), a free JavaScript library for visualizing molecular data which conveniently has [Python bindings](https://pypi.org/project/py3Dmol/) for notebooks. Using it is pretty straight forward, [here's](https://github.com/usmanm/mlx-esm/blob/main/notebooks/3dmol.ipynb) a Jupyter notebook with reference code I used.

Lo and behold, here's how these sequences look.

## Take Aways

Open-source ML feels like it's in a really dope spot. We have suddenly unlocked the ability to train really large neural nets, thanks to Nvidia and Transformers. We have both the compute and an architecture that scales with compute. Meta's decision to open-source their AI work allows anyone really to start playing around with these models. 

Neural net architectures have a lego block type feel. They're made of "modules" wrapped behind common interface making them composable. Composing them together in code sometimes isn't straight-forward though. I believe this is because they use a mathematical structure called [tensor](https://en.wikipedia.org/wiki/Tensor) (think of it as a N-dimensional matrix) to talk to each other. I wish I'd taken some linear algebra courses in college. It would be nice to find a more programming intuitive data structure instead of tensors.
