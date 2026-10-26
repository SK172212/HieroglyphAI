from .base import render_learning_card


def render_reading_direction_card() -> None:
    """
    Explain how to determine the reading direction of an
    Egyptian hieroglyphic inscription.

    This card is identical in Basic and Advanced Mode because
    the same explanation is useful for readers of all levels.
    """

    render_learning_card(
        card_key="reading-direction",
        title="Which Direction Do I Read?",
        border_color="#B7791F",
        level="Essential",
        introduction=(
            "Hieroglyphs can be written from left to right, "
            "from right to left, or vertically. The signs "
            "themselves usually show you where to begin."
        ),
        default_option="Look at the Faces",
        options={
            "Look at the Faces": {
                "symbol": "𓅓",
                "title": "Look at People and Animals",
                "text": (
                    "Find a sign showing a person, bird, or animal. "
                    "These figures normally face toward the beginning "
                    "of the inscription, so you read toward their faces. "
                    "If an owl faces left, begin on the left and read "
                    "toward it. If it faces right, begin on the right."
                ),
                "key_point": (
                    "<strong>Face the signs:</strong> Begin on the side "
                    "the people and animals are looking toward."
                ),
            },
            "Horizontal Text": {
                "symbol": "𓅓 𓈖 𓂋",
                "title": "Reading Horizontal Lines",
                "text": (
                    "Once you know where the inscription begins, move "
                    "across the line in that direction. Hieroglyphs may "
                    "therefore be read either from left to right or from "
                    "right to left. Unlike English, there is no single "
                    "direction used in every inscription."
                ),
                "key_point": (
                    "<strong>Follow the line:</strong> Start where the "
                    "signs face and continue across the inscription."
                ),
            },
            "Vertical Text": {
                "symbol": "𓅓<br>𓈖<br>𓂋",
                "title": "Reading Vertical Columns",
                "text": (
                    "Vertical inscriptions are generally read from top "
                    "to bottom. If there are several columns, first use "
                    "the direction of the people and animals to decide "
                    "whether to move through the columns from left to "
                    "right or from right to left."
                ),
                "key_point": (
                    "<strong>Read downward first:</strong> Then move to "
                    "the next column in the direction shown by the signs."
                ),
            },
        },
    )