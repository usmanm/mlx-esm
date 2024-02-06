typecheck:
    poetry run pyright mlx_esm

format:
    poetry run ruff mlx_esm --fix

vscode:
    poetry run code .
