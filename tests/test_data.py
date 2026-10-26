"""Tests for the dataset loaders. They run on tiny hand-written inputs, so the CI
never downloads anything; the last two tests use the real files only if
scripts/download_data.py has already been run."""

from pathlib import Path

import pytest

from spamedge.data import (
    SMS_EXPECTED_ROWS,
    SMS_EXPECTED_SPAM,
    YOUTUBE_EXPECTED_ROWS,
    YOUTUBE_EXPECTED_SPAM,
    check_counts,
    load_sms,
    load_youtube,
    parse_sms_lines,
    read_youtube_csv,
    sha256_of,
    video_name,
)

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def expect_value_error(func, *args, fragment: str) -> None:
    with pytest.raises(ValueError) as info:
        func(*args)
    assert fragment in str(info.value), (
        f"ECHEC : message obtenu '{info.value}', attendu un message contenant '{fragment}'"
    )


def test_sha256_matches_the_fips_180_2_test_vector(tmp_path):
    # SHA-256("abc") is the first example of FIPS 180-2, appendix B.1.
    f = tmp_path / "abc.txt"
    f.write_bytes(b"abc")
    expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    got = sha256_of(f)
    assert got == expected, f"ECHEC : obtenu {got}, attendu {expected}"


def test_sms_lines_are_parsed_and_blank_lines_skipped():
    df = parse_sms_lines(["ham\tSee you at 6\n", "\n", "spam\tWIN a prize now\r\n"])
    # 3 lines, 1 blank -> 2 messages; ham -> 0, spam -> 1.
    expected_texts = ["See you at 6", "WIN a prize now"]
    expected_labels = [0, 1]
    got_texts, got_labels = list(df["text"]), list(df["label"])
    assert got_texts == expected_texts, f"ECHEC : obtenu {got_texts}, attendu {expected_texts}"
    assert got_labels == expected_labels, f"ECHEC : obtenu {got_labels}, attendu {expected_labels}"
    got_dtype = str(df["label"].dtype)
    assert got_dtype == "int64", f"ECHEC : dtype obtenu {got_dtype}, attendu int64"


def test_only_the_first_tab_separates_label_and_message():
    df = parse_sms_lines(["spam\tCall\tnow"])
    expected = "Call\tnow"
    got = df.loc[0, "text"]
    assert got == expected, f"ECHEC : obtenu {got!r}, attendu {expected!r}"


def test_sms_line_without_tab_is_rejected_with_its_line_number():
    expect_value_error(parse_sms_lines, ["ham\tok", "spam no tab here"], fragment="line 2")


def test_unknown_sms_label_is_rejected():
    # Labels are case-sensitive in the UCI file: 'Spam' would mean the format changed.
    expect_value_error(parse_sms_lines, ["Spam\thello"], fragment="unknown label 'Spam'")


def test_check_counts_rejects_a_wrong_spam_count():
    df = parse_sms_lines(["ham\ta", "spam\tb", "spam\tc"])  # 3 rows, 2 spam
    check_counts(df, 3, 2, "toy")
    expect_value_error(check_counts, df, 3, 1, "toy", fragment="got 3 rows / 2 spam, expected 3 / 1")


def test_video_name_is_taken_from_the_file_name():
    expected = "LMFAO"
    got = video_name(Path("Youtube03-LMFAO.csv"))
    assert got == expected, f"ECHEC : obtenu {got}, attendu {expected}"


def write_youtube_csv(path: Path, classes: list[int]) -> None:
    lines = ["COMMENT_ID,AUTHOR,DATE,CONTENT,CLASS"]
    for i, c in enumerate(classes):
        lines.append(f"id{i},Some Person {i},2013-11-07T06:20:48,\"comment {i}, with a comma\",{c}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_youtube_loader_keeps_only_text_label_and_video(tmp_path):
    csv = tmp_path / "Youtube01-Psy.csv"
    write_youtube_csv(csv, [1, 0])
    df = read_youtube_csv(csv)
    expected_columns = ["text", "label", "video"]
    expected_labels = [1, 0]
    expected_text = "comment 0, with a comma"  # the quoted comma must not split the field
    got_columns, got_labels, got_text = list(df.columns), list(df["label"]), df.loc[0, "text"]
    assert got_columns == expected_columns, f"ECHEC : obtenu {got_columns}, attendu {expected_columns}"
    assert got_labels == expected_labels, f"ECHEC : obtenu {got_labels}, attendu {expected_labels}"
    assert got_text == expected_text, f"ECHEC : obtenu {got_text!r}, attendu {expected_text!r}"
    # The author's name must not survive anywhere in the frame.
    flat = df.astype(str).to_string()
    assert "Some Person" not in flat, "ECHEC : le nom de l'auteur apparaît dans les données chargées"


def test_youtube_class_outside_0_1_is_rejected(tmp_path):
    csv = tmp_path / "Youtube02-KatyPerry.csv"
    write_youtube_csv(csv, [1, 2])
    expect_value_error(read_youtube_csv, csv, fragment="1 rows with CLASS outside 0/1")


@pytest.mark.skipif(not (RAW / "sms" / "SMSSpamCollection").exists(), reason="real data not downloaded")
def test_real_sms_file_matches_the_uci_counts():
    df = load_sms(RAW / "sms" / "SMSSpamCollection")
    expected = (SMS_EXPECTED_ROWS, SMS_EXPECTED_SPAM)
    got = (len(df), int(df["label"].sum()))
    assert got == expected, f"ECHEC : obtenu {got}, attendu {expected}"


@pytest.mark.skipif(not (RAW / "youtube").exists(), reason="real data not downloaded")
def test_real_youtube_files_match_the_uci_counts():
    df = load_youtube(RAW / "youtube")
    expected = (YOUTUBE_EXPECTED_ROWS, YOUTUBE_EXPECTED_SPAM)
    got = (len(df), int(df["label"].sum()))
    assert got == expected, f"ECHEC : obtenu {got}, attendu {expected}"
