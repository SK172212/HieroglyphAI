from .base import render_learning_card


def render_how_hieroglyphs_work_card(
    advanced_mode: bool,
) -> None:
    """
    Render the How Hieroglyphs Work card.
    """

    if advanced_mode:
        render_learning_card(
            card_key="how-hieroglyphs-work-advanced",
            title="How Hieroglyphs Work",
            border_color="#2E8B57",
            level="Advanced",
            introduction=(
                "A single hieroglyph can function differently depending "
                "on its position and context inside a word. Select each "
                "option to explore the three principal sign functions "
                "used in Middle Egyptian writing."
            ),
            default_option="Phonograms",
            options={
                "Phonograms": {
                    "symbol": "𓅓",
                    "title": "Phonograms",
                    "text": (
                        "A phonogram represents one or more consonantal "
                        "sounds rather than the object it depicts. "
                        "Phonograms may be uniliteral, biliteral, or "
                        "triliteral depending on whether they represent "
                        "one, two, or three consonants. The owl, for "
                        "example, commonly represents <strong>m</strong>."
                    ),
                    "key_point": (
                        "<strong>Key idea:</strong> A phonogram contributes "
                        "sound to the written word, regardless of its picture."
                    ),
                },
                "Ideograms": {
                    "symbol": "𓇳",
                    "title": "Ideograms",
                    "text": (
                        "An ideogram, also called a word sign, represents "
                        "the complete word for the object or idea shown. "
                        "A sun disk can write the word for "
                        "<strong>sun</strong>. Ideograms are commonly "
                        "followed by a vertical stroke that marks the sign "
                        "as a complete word."
                    ),
                    "key_point": (
                        "<strong>Key idea:</strong> An ideogram represents "
                        "an entire word rather than only part of its sound."
                    ),
                },
                "Determinatives": {
                    "symbol": "𓀀",
                    "title": "Determinatives",
                    "text": (
                        "A determinative is normally an unpronounced sign "
                        "placed near the end of a word. It identifies the "
                        "word's semantic category and helps distinguish "
                        "words that share the same consonantal spelling. "
                        "A seated person, for example, can indicate that "
                        "the word refers to a human being."
                    ),
                    "key_point": (
                        "<strong>Key idea:</strong> A determinative clarifies "
                        "meaning but usually contributes no spoken sound."
                    ),
                },
            },
        )

    else:
        render_learning_card(
            card_key="how-hieroglyphs-work-basic",
            title="How Hieroglyphs Work",
            border_color="#2E8B57",
            level="Basic",
            introduction=(
                "Hieroglyphs are more than pictures. A sign can perform "
                "different jobs depending on how it is used inside a word. "
                "Select each option to explore the three main functions."
            ),
            default_option="Sound Signs",
            options={
                "Sound Signs": {
                    "symbol": "𓅓",
                    "title": "Sound Signs",
                    "text": (
                        "A sound sign represents one or more consonant sounds. "
                        "The picture does not necessarily describe the meaning "
                        "of the word. For example, the owl commonly represents "
                        "the sound <strong>m</strong>."
                    ),
                    "key_point": (
                        "<strong>Think of it like a letter:</strong> "
                        "the image is being used to write a sound."
                    ),
                },
                "Picture Signs": {
                    "symbol": "𓇳",
                    "title": "Picture Signs",
                    "text": (
                        "A picture sign represents the object or idea that it "
                        "depicts. A sun disk, for example, can stand for the "
                        "entire word meaning <strong>sun</strong>. Word signs "
                        "are often followed by a short vertical stroke."
                    ),
                    "key_point": (
                        "<strong>Think of it like a picture-word:</strong> "
                        "one sign can represent a complete word."
                    ),
                },
                "Meaning Signs": {
                    "symbol": "𓀀",
                    "title": "Meaning Signs",
                    "text": (
                        "A meaning sign appears at the end of a word and gives "
                        "a clue about its meaning. It is normally silent. A "
                        "seated person, for example, can show that a word refers "
                        "to a person."
                    ),
                    "key_point": (
                        "<strong>Think of it like a category label:</strong> "
                        "it helps explain what kind of word came before it."
                    ),
                },
            },
        )