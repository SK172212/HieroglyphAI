from .base import render_learning_card


def render_phonetic_complements_card() -> None:
    render_learning_card(
        card_key="phonetic-complements",
        title="Phonetic Complements",
        border_color="#5C6BC0",
        level="Advanced",
        introduction=(
            "Many Egyptian words contain extra sound signs that appear to repeat "
            "letters already written. These are called phonetic complements."
        ),
        default_option="What Are They?",
        options={
            "What Are They?": {
                "symbol": "𓄤 𓆑",
                "title": "Reinforcing a Reading",
                "text": (
                    "A phonetic complement is a one-consonant or two-consonant "
                    "sign added next to a larger phonogram that already contains "
                    "those same sounds. It reinforces the intended pronunciation "
                    "without changing how the word is read."
                ),
                "key_point": (
                    "<strong>Phonetic complements reinforce sounds—they do not "
                    "add new ones.</strong>"
                ),
            },
            "Why Use Them?": {
                "symbol": "𓄤 → 𓄤𓆑",
                "title": "Helping the Reader",
                "text": (
                    "Many biliteral and triliteral signs could be difficult to "
                    "recognize on their own. Egyptian scribes often added one-"
                    "consonant signs to make the intended reading clearer and to "
                    "reduce ambiguity."
                ),
                "key_point": (
                    "<strong>Think of them as confirmation signs rather than "
                    "additional spelling.</strong>"
                ),
            },
            "How to Read Them": {
                "symbol": "𓄤𓆑𓂋",
                "title": "Don't Pronounce Them Twice",
                "text": (
                    "When reading a word, pronounce the sounds of the main "
                    "phonogram only once. The complement repeats sounds already "
                    "present in the larger sign and should not be spoken again."
                ),
                "key_point": (
                    "<strong>Read the sound once, even if it appears multiple "
                    "times.</strong>"
                ),
            },
        },
    )