import hashlib
from pathlib import Path


# -------------------------------------------------
# Calculate File Hash
# -------------------------------------------------

def calculate_file_hash(file_path: Path):
    """
    Calculate the SHA-256 hash of a file.

    Args:
        file_path (Path): Path to the document.

    Returns:
        str: SHA-256 hash.
    """

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:

        while True:

            chunk = file.read(8192)

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()