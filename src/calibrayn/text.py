import re

WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

STOPWORDS = frozenset(
    """
    a about above after again against all also am an and any are as at be because been before
    being below between both but by can could did do does doing down during each few for from
    further had has have having he her here hers him his how i if in into is it its just me more
    most my no nor not now of off on once only or other our out over own same she should so some
    such than that the their them then there these they this those through to too under until up
    very was we were what when where which while who whom why will with would you your
    """.split()
)


def words(text):
    return WORD.findall(text.lower())


def content_words(text):
    return [word for word in words(text) if len(word) > 2 and word not in STOPWORDS]


def sentences(text):
    flat = " ".join(text.split())
    return [part.strip() for part in SENTENCE_END.split(flat) if part.strip()]


def shorten(text, limit):
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    cut = flat[:limit].rsplit(" ", 1)[0].rstrip(".,;:")
    return (cut or flat[:limit]) + "..."
