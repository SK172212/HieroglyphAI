from .how_hieroglyphs_work import (
    render_how_hieroglyphs_work_card,
)
from .phonetic_complements import (
    render_phonetic_complements_card,
)
from .reading_direction import (
    render_reading_direction_card,
)
from .suffix_pronouns import (
    render_suffix_pronouns_card,
)


def render_learn_tab(
    advanced_mode: bool,
) -> None:

    render_how_hieroglyphs_work_card(
        advanced_mode=advanced_mode,
    )

    render_reading_direction_card()

    render_suffix_pronouns_card()

    if advanced_mode:
        render_phonetic_complements_card()