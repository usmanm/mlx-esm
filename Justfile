typecheck:
    poetry run pyright mlx_esm

format:
    poetry run ruff check mlx_esm --fix
    poetry run ruff format mlx_esm

vscode:
    poetry run code .
