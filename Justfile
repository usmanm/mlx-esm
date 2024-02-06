typecheck:
    poetry run pyright mlx_esm

format:
    poetry run ruff --fix mlx_esm

vscode:
    poetry run code .
