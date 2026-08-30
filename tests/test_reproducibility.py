from src.utils import sha256_file


def test_data_checksum_is_stable(data_path):
    checksum = sha256_file(data_path)
    assert checksum == sha256_file(data_path)
    assert len(checksum) == 64
