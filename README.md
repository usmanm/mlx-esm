# mlx-esm

This is an implementation of Meta's [Evolutionary Scale Modeling (esm)](https://github.com/facebookresearch/esm) in Apple's [MLX](https://ml-explore.github.io/mlx/build/html/index.html) library.

I've been learning about deep learning and neural nets over the last few months. The two best teachers in this domain are [Jeremy Howard](https://twitter.com/jeremyphoward) and [Andrej Karpathy](https://twitter.com/karpathy). Their intuitive understanding of neural nets and their ability to clearly articulate complex concepts are both amazing. To get started, watch these lectures (1.5x speed recommended):
- [Practical Deep Learning for Coders](https://course.fast.ai/)
- [Neural Networks: Zero to Hero](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)

During this time, I also became interested in [TechBio](https://www.nfx.com/post/biotech-to-techbio). While exploring the space, I came across Meta's research paper: [Evolutionary-scale prediction of atomic level protein structure
with a language model](https://www.biorxiv.org/content/10.1101/2022.07.20.500902v3.full.pdf). Like all of Meta's AI research, these language models are fully [open-source](https://github.com/facebookresearch/esm) (what a great world we live in).

I'd been wanting to implement and train neural nets which are more than a toy example. I'm going to reimplement ESM-2 and ESMFold from the research paper, but in MLX. I'll use the PyTorch implementation as a reference, but at least converting it to a different library will force me to grok some of the concepts of the neural net architecture.
