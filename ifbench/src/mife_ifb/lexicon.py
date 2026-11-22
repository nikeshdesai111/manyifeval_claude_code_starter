"""Word inventory for the witness builder.

Base: IFBench's own WORD_LIST (1,525 common English words, the list its keyword rules sample
from). Supplement: a small hand-written list of ordinary English words that WORD_LIST lacks but
some rules need (words starting with x, y, z, q, k, v, j for ``words:alphabet``; real
palindromes for ``words:palindrome``; short words for ``words:prime_lengths``; imperative verbs
for ``words:start_verb``). The supplement holds no names, numbers or made-up words.

Closed classes (pronouns, conjunctions, person names) are copied from the official verifiers
and checked against them in tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from . import source

SUPPLEMENT = """
xenon xylophone xylem xray yacht yarn yawn yeast yield yoga yolk yonder yearn zebra zero zest zigzag
zinc zoom zeal zany zipper quick quiet quilt quiz quest queen quote quartz quench kettle kernel kind kite
knot knee kayak koala vivid vast velvet vine violin vote vault jazz jelly jewel joke jolly jump jungle
oak oar ox owl orb ode elk elm emu egg ego ink imp ivy urn usher umpire utter
level radar civic rotor refer madam stats tenet racecar redder reviver sagas solos minim deified repaper
rotator redivider sexes shahs
an as at be by do go he if in is it me my no of on or so to up us we ox
ant ape arc ark art ash axe bag bat bay bed bee bin bow box bud bus cab cap cat cow cub cup dam den dew dig
dog dot dry dug ear eel fan fig fin fir fix fly fog fox fun fur gap gas gem gym hat hay hen hip hog hop hub
hug hut ice jam jar jaw jet jog jug key kit lab lad lap leg lid lip log map mat mix mob mop mud mug nap net
nut oil owl pad pan paw pea pen pet pie pig pin pit pod pot pub pug rag ram rat ray rib rim rod rug rum sap
saw sea sip sky sly spa spy sun tab tag tan tap tar tea tin tip toe top toy tub tug urn van vet wax web wig
yak yam yap zip
apple bread chair dance eagle fable grape honey igloo juice lemon mango nurse olive piano quail river snake
tiger uncle vapor whale yeast zebra
balloon cabbage dolphin feather giraffe harvest journey kitchen library monster octopus pancake rainbow
sparrow thunder unicorn volcano weather
""".split()

VERBS = ["write", "take", "consider", "imagine", "begin", "keep", "make", "build", "choose", "give", "hold",
         "start", "try", "use", "draw", "watch", "read", "think", "find", "bring"]

PRONOUNS = sorted({
    'i', 'me', 'we', 'us', 'you', 'he', 'him', 'she', 'her', 'it', 'they', 'them',
    'my', 'mine', 'our', 'ours', 'your', 'yours', 'his', 'hers', 'its', 'their', 'theirs',
    'myself', 'ourselves', 'yourself', 'yourselves', 'himself', 'herself', 'itself', 'themselves',
    'this', 'that', 'these', 'those', 'who', 'whom', 'whose', 'which', 'what',
    'whoever', 'whomever', 'whatever', 'whichever',
    'anybody', 'anyone', 'anything', 'everybody', 'everyone', 'everything',
    'nobody', 'nothing', 'somebody', 'someone', 'something',
    'each', 'either', 'neither', 'both', 'all', 'some', 'any', 'none'})

CONJUNCTIONS = ["and", "but", "for", "nor", "or", "so", "yet"]

PERSON_NAMES = ["Emma", "Liam", "Sophia", "Jackson", "Olivia", "Noah", "Ava", "Lucas", "Isabella", "Mason", "Mia",
                "Ethan", "Charlotte", "Alexander", "Amelia", "Benjamin", "Harper", "Leo", "Zoe", "Daniel", "Chloe",
                "Samuel", "Lily", "Matthew", "Grace", "Owen", "Abigail", "Gabriel", "Ella", "Jacob", "Scarlett",
                "Nathan", "Victoria", "Elijah", "Layla", "Nicholas", "Audrey", "David", "Hannah", "Christopher",
                "Penelope", "Thomas", "Nora", "Andrew", "Aria", "Joseph", "Claire", "Ryan", "Stella", "Jonathan"]

PALINDROMES = ["level", "radar", "civic", "kayak", "rotor", "refer", "madam", "stats", "tenet", "racecar",
               "redder", "reviver", "sagas", "solos", "minim", "deified", "repaper", "rotator", "redivider",
               "sexes", "shahs"]

EMOJI = ["\U0001F600", "\U0001F31E", "\U0001F333", "\U0001F30A", "\U0001F680", "\U0001F34E", "\U0001F3B5",
         "\U0001F4DA", "\U0001F431", "\U0001F525"]

CONSONANTS = set("bcdfghjklmnpqrstvwxyz")
PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23}


@dataclass(frozen=True)
class W:
    text: str           # lowercase
    first: str
    length: int
    vowels: frozenset
    cluster: bool
    parity: int         # syllapy syllable count mod 2
    stop: bool


def _cluster(w: str) -> bool:
    return any(a in CONSONANTS and b in CONSONANTS for a, b in zip(w, w[1:]))


@lru_cache(maxsize=None)
def feat(word: str) -> W:
    import syllapy

    w = word.lower()
    return W(w, w[0] if w else "", len(w), frozenset(c for c in w if c in "aeiou"), _cluster(w),
             syllapy.count(w) % 2, w in stopwords())


@lru_cache(maxsize=1)
def stopwords() -> frozenset:
    source._bootstrap()
    import nltk

    return frozenset(nltk.corpus.stopwords.words("english"))


@lru_cache(maxsize=1)
def vocabulary() -> tuple[str, ...]:
    """Filler vocabulary: WORD_LIST + supplement, lowercase alphabetic, de-duplicated, stop words removed."""
    base = source._bootstrap()[2].WORD_LIST
    seen, out = set(), []
    for w in list(base) + SUPPLEMENT:
        w = w.lower()
        if w.isalpha() and w not in seen and w not in stopwords() and w not in PRONOUNS:
            seen.add(w)
            out.append(w)
    return tuple(out)


@lru_cache(maxsize=1)
def stop_vocabulary() -> tuple[str, ...]:
    """Stop words usable as filler (only when no stop-word cap applies)."""
    return tuple(sorted(w for w in stopwords() if w.isalpha() and len(w) > 1 and w not in PRONOUNS))
