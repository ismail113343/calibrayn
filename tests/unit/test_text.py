from calibrayn.text import STOPWORDS, content_words, sentences, shorten, words


def test_words_are_lowercase_and_keep_apostrophes():
    assert words("The Student's NOTES, 2nd edition!") == [
        "the",
        "student's",
        "notes",
        "2nd",
        "edition",
    ]


def test_content_words_drop_stopwords_and_tiny_words():
    assert content_words("The cat is on the mat in an old box") == ["cat", "mat", "old", "box"]


def test_content_words_of_nothing_is_empty():
    assert content_words("") == []
    assert content_words("the and of") == []


def test_stopwords_are_a_fixed_set():
    assert "the" in STOPWORDS
    assert isinstance(STOPWORDS, frozenset)


def test_sentences_split_on_full_stops_and_ignore_extra_space():
    text = "First one.  Second one?\nThird   one! Last"
    assert sentences(text) == ["First one.", "Second one?", "Third one!", "Last"]


def test_sentences_of_nothing_is_empty():
    assert sentences("   ") == []


def test_shorten_leaves_short_text_alone():
    assert shorten("a short line", 40) == "a short line"


def test_shorten_cuts_at_a_word_boundary():
    assert shorten("alpha beta gamma delta", 12) == "alpha beta..."


def test_shorten_handles_one_long_word():
    assert shorten("abcdefghijklmnop", 5) == "abcde..."
