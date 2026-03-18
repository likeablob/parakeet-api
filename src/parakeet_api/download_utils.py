import tarfile
import urllib.request
from pathlib import Path

from huggingface_hub import snapshot_download

SHERPA_DEFAULT_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2"
MLX_DEFAULT_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"


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


def download_sherpa(url: str, output_base: Path):
    sherpa_dir = output_base / "sherpa"
    sherpa_dir.mkdir(exist_ok=True, parents=True)

    filename = url.split("/")[-1]
    if ".tar." in filename:
        model_name = filename.split(".tar.")[0]
    elif filename.endswith(".tar"):
        model_name = filename[:-4]
    else:
        model_name = filename

    target_path = sherpa_dir / model_name
    if (target_path / "model.onnx").exists() or (
        target_path / "model.int8.onnx"
    ).exists():
        print(f"Sherpa model already exists at {target_path}")
        return

    temp_archive = sherpa_dir / filename
    print(f"Downloading Sherpa model from {url}...")
    urllib.request.urlretrieve(url, temp_archive)

    print(f"Extracting {temp_archive}...")
    if filename.endswith(".tar.bz2"):
        mode = "r:bz2"
    elif filename.endswith(".tar.gz") or filename.endswith(".tgz"):
        mode = "r:gz"
    else:
        mode = "r"

    try:
        with tarfile.open(temp_archive, mode) as tar:
            safe_extract(tar, sherpa_dir)
        print(f"Done! Model extracted to {target_path}")
    finally:
        if temp_archive.exists():
            temp_archive.unlink()


def download_mlx(repo_id: str, output_base: Path):
    model_name = repo_id.split("/")[-1]
    local_dir = output_base / "mlx" / model_name

    print(f"Downloading MLX model '{repo_id}' to {local_dir}...")
    snapshot_download(repo_id=repo_id, local_dir=local_dir)
    print(f"Done! MLX model saved to {local_dir}")
