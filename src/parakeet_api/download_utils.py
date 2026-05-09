import tarfile
import urllib.request
from pathlib import Path

from huggingface_hub import snapshot_download

SHERPA_DEFAULT_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2"
MLX_DEFAULT_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"


def _generate_bpe_vocab_from_tokens(target_path: Path):
    """Generate bpe.vocab from model's tokens.txt. Raises if tokens.txt not found."""
    bpe_vocab_path = target_path / "bpe.vocab"

    if bpe_vocab_path.exists():
        print(f"bpe.vocab already exists at {bpe_vocab_path}")
        return

    tokens_path = target_path / "tokens.txt"
    if not tokens_path.exists():
        raise FileNotFoundError(f"tokens.txt not found in {target_path}")

    with open(tokens_path, encoding="utf-8") as f:
        lines = f.readlines()

    if not lines:
        raise ValueError(f"tokens.txt is empty in {target_path}")

    with open(bpe_vocab_path, "w", encoding="utf-8") as out:
        for line in lines:
            token = line.strip().split()[0] if line.strip() else ""
            if token:
                out.write(f"{token} 0\n")

    print(f"bpe.vocab generated at {bpe_vocab_path}")


def extract_model_name_from_url(url: str) -> str:
    filename = url.split("/")[-1]
    if ".tar." in filename:
        return filename.split(".tar.")[0]
    elif filename.endswith(".tar"):
        return filename[:-4]
    return filename


def is_within_directory(directory: Path, target: Path):
    abs_directory = directory.resolve()
    abs_target = target.resolve()
    return abs_target.parts[: len(abs_directory.parts)] == abs_directory.parts


def safe_extract(tar, path: Path):
    for member in tar.getmembers():
        member_path = path / member.name
        if not is_within_directory(path, member_path):
            raise Exception("Attempted Path Traversal in Tar File")
    tar.extractall(path=path)


def download_sherpa(url: str, output_base: Path, generate_bpe_vocab: bool = False):
    sherpa_dir = output_base / "sherpa"
    sherpa_dir.mkdir(exist_ok=True, parents=True)

    model_name = extract_model_name_from_url(url)
    target_path = sherpa_dir / model_name

    model_exists = (
        (target_path / "model.onnx").exists()
        or (target_path / "model.int8.onnx").exists()
        or any(
            p.exists()
            for p in [target_path / "encoder.onnx", target_path / "encoder.int8.onnx"]
        )
    )

    if model_exists:
        print(f"Sherpa model already exists at {target_path}")
        if generate_bpe_vocab:
            _generate_bpe_vocab_from_tokens(target_path)
        return

    temp_archive = sherpa_dir / url.split("/")[-1]
    print(f"Downloading Sherpa model from {url}...")
    urllib.request.urlretrieve(url, temp_archive)

    print(f"Extracting {temp_archive}...")
    if temp_archive.name.endswith(".tar.bz2"):
        mode = "r:bz2"
    elif temp_archive.name.endswith(".tar.gz") or temp_archive.name.endswith(".tgz"):
        mode = "r:gz"
    else:
        mode = "r"

    try:
        with tarfile.open(temp_archive, mode) as tar:
            safe_extract(tar, sherpa_dir)
        print(f"Done! Model extracted to {target_path}")

        if generate_bpe_vocab:
            _generate_bpe_vocab_from_tokens(target_path)
    finally:
        if temp_archive.exists():
            temp_archive.unlink()


def download_mlx(repo_id: str, output_base: Path):
    model_name = repo_id.split("/")[-1]
    local_dir = output_base / "mlx" / model_name

    print(f"Downloading MLX model '{repo_id}' to {local_dir}...")
    snapshot_download(repo_id=repo_id, local_dir=local_dir)
    print(f"Done! MLX model saved to {local_dir}")
