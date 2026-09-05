"""Cross-lingual check: does a Slovak query find the right sentence in 26 languages?

Two topics, the same content in every language, one sentence each. A model with real
cross-lingual alignment must rank every capital-sentence above every cat-sentence,
whatever script or family it is written in.

The interesting result is not per-language accuracy - most multilingual models get
that right - but whether the groups stay separable *globally*. A model can answer
every language correctly on its own and still score some correct answer below some
distractor from another language, which is what breaks a mixed-language corpus.

    uv run python devtools/multilingual_check.py bge-m3 e5-small-int8

Models must be downloaded first (embedforge model download <id>).
"""

import sys
from pathlib import Path

import numpy as np

from embedforge.config import Settings
from embedforge.engine.base import TaskType, TextInput
from embedforge.engine.registry import create_backend

# (language, family, "capital of Slovakia" sentence, "cat sleeps on sofa" sentence)
DATA = [
    (
        "en",
        "Germanic",
        "The capital of Slovakia is Bratislava.",
        "The cat is sleeping on the sofa.",
    ),
    (
        "de",
        "Germanic",
        "Die Hauptstadt der Slowakei ist Bratislava.",
        "Die Katze schläft auf dem Sofa.",
    ),
    ("nl", "Germanic", "De hoofdstad van Slowakije is Bratislava.", "De kat slaapt op de bank."),
    ("sv", "Germanic", "Slovakiens huvudstad är Bratislava.", "Katten sover i soffan."),
    ("sk", "Slavic", "Hlavné mesto Slovenska je Bratislava.", "Mačka spí na gauči."),
    ("cs", "Slavic", "Hlavní město Slovenska je Bratislava.", "Kočka spí na gauči."),
    ("pl", "Slavic", "Stolicą Słowacji jest Bratysława.", "Kot śpi na kanapie."),
    ("ru", "Slavic", "Столица Словакии — Братислава.", "Кошка спит на диване."),
    ("uk", "Slavic", "Столиця Словаччини — Братислава.", "Кіт спить на дивані."),
    ("es", "Romance", "La capital de Eslovaquia es Bratislava.", "El gato duerme en el sofá."),
    ("fr", "Romance", "La capitale de la Slovaquie est Bratislava.", "Le chat dort sur le canapé."),
    ("it", "Romance", "La capitale della Slovacchia è Bratislava.", "Il gatto dorme sul divano."),
    ("ro", "Romance", "Capitala Slovaciei este Bratislava.", "Pisica doarme pe canapea."),
    ("hu", "Uralic", "Szlovákia fővárosa Bratislava.", "A macska a kanapén alszik."),
    ("fi", "Uralic", "Slovakian pääkaupunki on Bratislava.", "Kissa nukkuu sohvalla."),
    ("tr", "Turkic", "Slovakya'nın başkenti Bratislava'dır.", "Kedi kanepede uyuyor."),
    (
        "el",
        "Hellenic",
        "Η πρωτεύουσα της Σλοβακίας είναι η Μπρατισλάβα.",
        "Η γάτα κοιμάται στον καναπέ.",
    ),
    ("zh", "Sinitic", "斯洛伐克的首都是布拉迪斯拉发。", "猫在沙发上睡觉。"),
    ("ja", "Japonic", "スロバキアの首都はブラチスラバです。", "猫がソファで寝ています。"),
    (
        "ko",
        "Koreanic",
        "슬로바키아의 수도는 브라티슬라바입니다.",
        "고양이가 소파에서 자고 있습니다.",
    ),
    ("ar", "Semitic", "عاصمة سلوفاكيا هي براتيسلافا.", "القطة تنام على الأريكة."),
    ("he", "Semitic", "בירת סלובקיה היא ברטיסלבה.", "החתול ישן על הספה."),
    ("hi", "Indo-Aryan", "स्लोवाकिया की राजधानी ब्रातिस्लावा है।", "बिल्ली सोफे पर सो रही है।"),
    ("th", "Tai", "เมืองหลวงของสโลวาเกียคือบราติสลาวา", "แมวกำลังนอนอยู่บนโซฟา"),
    (
        "vi",
        "Austroasiatic",
        "Thủ đô của Slovakia là Bratislava.",
        "Con mèo đang ngủ trên ghế sofa.",
    ),
    ("sw", "Bantu", "Mji mkuu wa Slovakia ni Bratislava.", "Paka analala kwenye sofa."),
]
QUERY = "Aké je hlavné mesto Slovenska?"  # Slovak


def run(model_id: str) -> None:
    b = create_backend(Settings(model_dir=Path("data/models"), model_id=model_id))
    b.load()
    docs = [row[2] for row in DATA] + [row[3] for row in DATA]
    D = b.embed([TextInput(t) for t in docs], TaskType.DOCUMENT)
    q = b.embed([TextInput(QUERY)], TaskType.QUERY)[0]
    n = len(DATA)
    # Annotated because indexing an ndarray widens to Any, which hides the
    # element type from everything downstream.
    right: np.ndarray = np.asarray(q @ D[:n].T)
    wrong: np.ndarray = np.asarray(q @ D[n:].T)

    failures: list[str] = []
    print(f"\n=== {model_id} — Slovak query against 26 languages ===")
    print(f"{'lang':<5} {'family':<14} {'correct':>8} {'distractor':>11} {'margin':>8}")
    for i, (lang, fam, _, _) in enumerate(DATA):
        ok = right[i] > wrong[i]
        if not ok:
            failures.append(lang)
        print(
            f"{lang:<5} {fam:<14} {right[i]:8.3f} {wrong[i]:11.3f} {right[i] - wrong[i]:+8.3f} {'' if ok else '  <-- FAILS'}"
        )
    # Does the whole correct group outrank the whole distractor group?
    separable = right.min() > wrong.max()
    print(
        f"\n  languages resolved correctly : {n - len(failures)}/{n}"
        + (f"  (failed: {failures})" if failures else "")
    )
    print(
        f"  worst correct {right.min():.3f} vs best distractor {wrong.max():.3f} -> "
        f"{'fully separable' if separable else 'OVERLAP between groups'}"
    )
    print(f"  mean margin: {float(np.mean(right - wrong)):+.3f}")
    b.close()


for model_id in sys.argv[1:]:
    run(model_id)
