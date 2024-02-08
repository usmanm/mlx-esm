# mlx-esm

This is an implementation of Meta's [ESM-1](https://huggingface.co/docs/transformers/model_doc/esm) in Apple's [MLX](https://ml-explore.github.io/mlx/build/html/index.html) library.

I've been learning about deep learning and neural nets over the last few months. The two best teachers in this space are [Jeremy Howard](https://twitter.com/jeremyphoward) and [Andrej Karpathy](https://twitter.com/karpathy). Their intuitive understanding of neural nets and their ability to clearly articulate complex concepts are both incredible. To get started, watch these lectures (1.5x speed recommended):
- [Practical Deep Learning for Coders](https://course.fast.ai/)
- [Neural Networks: Zero to Hero](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)

During this time, I also became interested in [TechBio](https://www.nfx.com/post/biotech-to-techbio). While exploring it, I came across Meta's research papers:
- [Biological structure and function emerge from scalingunsupervised learning to 250 million protein sequences](https://www.pnas.org/doi/epdf/10.1073/pnas.2016239118)
- [Evolutionary-scale prediction of atomic level protein structure
with a language model](https://www.biorxiv.org/content/10.1101/2022.07.20.500902v3.full.pdf)

Like all of Meta's AI research, the source code and weights for these protein language models are [open-source](https://github.com/facebookresearch/esm) (what a great world we live in).

I'd been wanting to implement and train neural nets which are more than a toy example. And I believe there are no coincidences in life. So, I'm going to reimplement ESM-1 from the research paper, but in MLX. I'll use the ESM-1 PyTorch [implementation](https://github.com/facebookresearch/esm/blob/main/esm/model/esm1.py) and Bert MLX [implementation](https://github.com/ml-explore/mlx-examples/blob/main/bert/model.py) as a reference. Hoping this provides enough copy-pasta that I can actually do this, but tinkering needed to force me to grok some of the low-level concepts of building neural net architectures. ESM-1 is a fork of the [BERT](https://huggingface.co/docs/transformers/model_doc/bert) language model and uses the masked language modeling objective.
