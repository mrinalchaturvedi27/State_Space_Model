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
