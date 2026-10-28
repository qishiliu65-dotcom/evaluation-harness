from eval_harness.reproducibility import compute_file_sha256


def test_compute_file_sha256_is_stable(tmp_path):
    file_path = tmp_path / "data.txt"
    file_path.write_text("hello", encoding="utf-8")

    first_hash = compute_file_sha256(str(file_path))
    second_hash = compute_file_sha256(str(file_path))

    assert first_hash == second_hash

def test_compute_file_sha256_changes_when_content_changes(tmp_path):
    file_path = tmp_path / "data.txt"

    file_path.write_text("hello", encoding="utf-8")
    first_hash = compute_file_sha256(str(file_path))

    file_path.write_text("hello world", encoding="utf-8")
    second_hash = compute_file_sha256(str(file_path))

    assert first_hash != second_hash