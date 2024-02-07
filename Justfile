typecheck:
    poetry run pyright mlx_esm

format:
    poetry run ruff format mlx_esm
    poetry run ruff check mlx_esm --fix

vscode:
    poetry run code .
