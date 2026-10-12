"""Content-word helpers, identical to ABLATION_AND_ERROR_ANALYSIS.md: lowercase tokens that are
not English/German function words and not digits."""
import re

EN_STOP = set("a an the and or but if of to in on at by for with from is are was were be been being it its this that these those he she they we you i me him her them us my his their our your not no so as do does did have has had will would can could shall should may might must there here then than when what which who whom how why all any some just also very up out into over about after before again once only own same too more most such am".split())
DE_STOP = set("der die das den dem des ein eine einen einem einer eines und oder aber ist sind war waren wird werden wurde es sie er wir ihr ich du in im an am auf aus bei mit nach von vor zu zum zur für über unter um bis durch auch nicht noch nur schon so wie als da dann dass doch sich hat haben hatte sein kann können soll sollen".split())


def words(s: str) -> list[str]:
    return re.findall(r"[\w']+", s.lower())


def content(s: str, lang: str = "en") -> set[str]:
    stop = DE_STOP if lang == "de" else EN_STOP
    return {w for w in words(s) if w not in stop and not w.isdigit() and w != "unknown"}


def content_tokens(s: str, lang: str = "en") -> list[str]:
    """content() as a multiset (repeats kept), for clipped-count precision / recall / F1. Use this for
    any F1: Counter(content(...)) silently counts each word once (2026-10-12 review)."""
    stop = DE_STOP if lang == "de" else EN_STOP
    return [w for w in words(s) if w not in stop and not w.isdigit() and w != "unknown"]


# Categories content() excludes, scored separately (review: "separately evaluate numbers and negation").
EN_NEG = set("not no never nothing none nobody neither nor nowhere cannot".split())
DE_NEG = set("nicht kein keine keinen keinem keiner keines nie niemals nichts niemand weder".split())
EN_NUM = set("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
             "seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand "
             "million billion first second third half".split())
DE_NUM = set("null eins zwei drei vier fünf sechs sieben acht neun zehn elf zwölf dreizehn vierzehn fünfzehn sechzehn "
             "siebzehn achtzehn neunzehn zwanzig dreißig vierzig fünfzig hundert tausend million grad".split())


def has_negation(s: str, lang: str = "en") -> bool:
    ws = words(s)
    neg = DE_NEG if lang == "de" else EN_NEG
    return any(w in neg or w.endswith("n't") for w in ws)


def number_tokens(s: str, lang: str = "en") -> list[str]:
    num = DE_NUM if lang == "de" else EN_NUM
    return [w for w in words(s) if w.isdigit() or w in num]
