from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components


def render_suffix_pronouns_card() -> None:
    """Render the eight Middle Egyptian suffix pronouns."""

    card_key = "suffix-pronouns"
    visual_key = f"{card_key}-visual"
    key_point_key = f"{card_key}-key-point"
    border_color = "#0F766E"

    pronouns = {
        ".j": {
            "image_filename": "suffix_1s001.png",
            "grammar": "1st Person Singular",
            "meaning": "I, me, or my",
            "example": "pr.j — my house",
            "note": (
                "Often omitted in writing; its form could depend "
                "on the speaker."
            ),
        },
        ".k": {
            "image_filename": "suffix_2ms001.png",
            "grammar": "2nd Person Masculine Singular",
            "meaning": "you or your",
            "example": "pr.k — your house",
            "note": "Used when addressing one masculine person.",
        },
        ".ṯ": {
            "image_filename": "suffix_2fs001.png",
            "grammar": "2nd Person Feminine Singular",
            "meaning": "you or your",
            "example": "pr.ṯ — your house",
            "note": "Used when addressing one feminine person.",
        },
        ".f": {
            "image_filename": "suffix_3ms001.png",
            "grammar": "3rd Person Masculine Singular",
            "meaning": "he, him, or his",
            "example": "pr.f — his house",
            "note": (
                "The English translation depends on the word "
                "carrying the suffix."
            ),
        },
        ".s": {
            "image_filename": "suffix_3fs001.png",
            "grammar": "3rd Person Feminine Singular",
            "meaning": "she, her, or hers",
            "example": "pr.s — her house",
            "note": "Refers to one feminine person or noun.",
        },
        ".n": {
            "image_filename": "suffix_1pl001.png",
            "grammar": "1st Person Plural",
            "meaning": "we, us, or our",
            "example": "pr.n — our house",
            "note": (
                "Includes the speaker together with one or more "
                "other people."
            ),
        },
        ".ṯn": {
            "image_filename": "suffix_2pl.png",
            "grammar": "2nd Person Plural",
            "meaning": "you or your (plural)",
            "example": "pr.ṯn — your house",
            "note": (
                "Addresses multiple people without distinguishing "
                "gender."
            ),
        },
        ".sn": {
            "image_filename": "suffix_3pl001.png",
            "grammar": "3rd Person Plural",
            "meaning": "they, them, or their",
            "example": "pr.sn — their house",
            "note": (
                "The standard Middle Egyptian third-person "
                "plural suffix."
            ),
        },
    }

    st.markdown(
        f"""
        <style>
            .st-key-{card_key} {{
                min-height: 510px;
                padding: 24px 26px;
                margin-bottom: 24px;
                border: 2px solid {border_color};
                border-radius: 20px;
                background: #fffdf8;
                box-shadow: 0 6px 20px rgba(31, 41, 55, 0.08);
            }}

            .st-key-{card_key}
            div[data-testid="stVerticalBlock"] {{
                gap: 0.75rem;
            }}

            .st-key-{card_key} .learn-level {{
                margin: 0 0 8px;
                color: #6b7280;
                font-size: 12px;
                font-weight: 700;
                letter-spacing: 0.12em;
                text-transform: uppercase;
            }}

            .st-key-{card_key} .learn-card-title {{
                margin: 0 0 10px;
                color: {border_color};
                font-size: 29px;
                font-weight: 750;
                line-height: 1.2;
            }}

            .st-key-{card_key} .learn-introduction {{
                margin: 0 0 14px;
                color: #374151;
                font-size: 16px;
                line-height: 1.6;
            }}

            .st-key-{card_key}
            div[data-testid="stButtonGroup"] {{
                width: 100%;
                margin: 2px 0 10px;
            }}

            .st-key-{card_key}
            div[data-testid="stButtonGroup"]
            div[role="radiogroup"] {{
                display: flex !important;
                flex-wrap: nowrap !important;
                width: 100% !important;
                gap: 0 !important;
            }}

            .st-key-{card_key}
            div[data-testid="stButtonGroup"]
            div[role="radiogroup"] > label {{
                flex: 1 1 0 !important;
                width: 0 !important;
                min-width: 0 !important;
                max-width: none !important;
            }}

            .st-key-{card_key}
            div[data-testid="stButtonGroup"] button {{
                width: 100% !important;
                min-width: 0 !important;
                max-width: none !important;
                min-height: 40px !important;
                padding: 6px 1px !important;
                font-weight: 700;
                white-space: nowrap !important;
            }}

            .st-key-{card_key}
            div[data-testid="stButtonGroup"]
            button[aria-checked="true"],
            .st-key-{card_key}
            div[data-testid="stButtonGroup"]
            label:has(input:checked) button {{
                border-color: #ff4b4b !important;
                color: #ff4b4b !important;
                background: rgba(255, 75, 75, 0.10) !important;
            }}

            .st-key-{visual_key} {{
                display: flex;
                align-items: center;
                justify-content: center;
                min-height: 125px;
                margin: 8px 0 14px;
                padding: 10px;
                border-radius: 14px;
                background:
                    color-mix(
                        in srgb,
                        {border_color} 8%,
                        transparent
                    );
            }}

            .st-key-{visual_key}
            div[data-testid="stVerticalBlock"] {{
                display: flex;
                align-items: center;
                justify-content: center;
                width: 100%;
                min-height: 125px;
                gap: 0;
            }}

            .st-key-{visual_key}
            div[data-testid="stImage"] {{
                display: flex;
                justify-content: center;
                width: 100%;
            }}

            .st-key-{visual_key} img {{
                display: block;
                max-width: 100%;
                max-height: 125px;
                margin: 0 auto;
                object-fit: contain;
            }}

            .st-key-{visual_key} .glyph-placeholder {{
                color: {border_color};
                text-align: center;
            }}

            .st-key-{visual_key} .placeholder-form {{
                font-size: 48px;
                font-weight: 750;
                line-height: 1;
            }}

            .st-key-{visual_key} .placeholder-label {{
                margin-top: 8px;
                color: #6b7280;
                font-size: 12px;
            }}

            .st-key-{card_key} .option-title {{
                margin: 3px 0 5px;
                color: {border_color};
                font-size: 22px;
                font-weight: 750;
                line-height: 1.3;
            }}

            .st-key-{card_key} .option-text {{
                margin: 0;
                color: #1f2937;
                font-size: 16px;
                line-height: 1.65;
            }}

            .st-key-{card_key} .pronoun-example {{
                display: inline-block;
                margin-top: 8px;
                color: #374151;
            }}

            .st-key-{key_point_key} {{
                display: flex;
                align-items: center;
                width: 100%;
                min-height: 58px;
                margin-top: 8px;
                padding: 14px 18px;
                border-left: 4px solid {border_color};
                border-radius: 10px;
                background:
                    color-mix(
                        in srgb,
                        {border_color} 8%,
                        transparent
                    );
            }}

            .st-key-{key_point_key}
            div[data-testid="stVerticalBlock"],
            .st-key-{key_point_key}
            div[data-testid="stMarkdownContainer"] {{
                display: flex;
                align-items: center;
                width: 100%;
                min-height: 28px;
                gap: 0;
            }}

            .st-key-{key_point_key} p {{
                width: 100%;
                margin: 0;
                color: #1f2937;
                font-size: 15px;
                line-height: 1.5;
            }}

            @media (max-width: 600px) {{
                .st-key-{card_key} {{
                    min-height: auto;
                    padding: 20px 12px;
                    border-radius: 18px;
                }}

                .st-key-{card_key} .learn-card-title {{
                    font-size: 25px;
                }}

                .st-key-{card_key} .learn-introduction,
                .st-key-{card_key} .option-text {{
                    font-size: 15px;
                }}

                .st-key-{card_key} .option-title {{
                    font-size: 20px;
                }}

                .st-key-{card_key}
                div[data-testid="stButtonGroup"] button {{
                    min-height: 34px !important;
                    padding: 3px 0 !important;
                    font-size: 11px !important;
                    line-height: 1 !important;
                    white-space: nowrap !important;
                }}

                .st-key-{visual_key},
                .st-key-{visual_key}
                div[data-testid="stVerticalBlock"] {{
                    min-height: 110px;
                }}

                .st-key-{visual_key} img {{
                    max-height: 105px;
                }}

                .st-key-{visual_key} .placeholder-form {{
                    font-size: 40px;
                }}

                .st-key-{key_point_key} {{
                    padding: 12px 14px;
                }}

                .st-key-{key_point_key} p {{
                    font-size: 14px;
                }}
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Mobile-only layout patch for Streamlit's segmented control.
    # Streamlit gives each option a built-in minimum width on narrow screens,
    # which can push the final suffix onto a second row. This script runs
    # only on phone-sized displays and re-applies a strict, non-wrapping flex
    # layout (mirroring the desktop CSS above) to the actual rendered DOM,
    # leaving the desktop layout untouched.
    #
    # NOTE: an earlier version of this patch used CSS Grid
    # (display: grid + grid-auto-flow: column) to force eight columns.
    # With only grid-template-columns defined and no grid-template-rows,
    # column-major auto-flow left row placement ambiguous, which is what
    # caused the eighth button to wrap onto a second row on some phones.
    # Flex with nowrap can't create a second row at all — children simply
    # shrink toward min-width: 0 to fit — so that failure mode is gone.
    components.html(
        f"""
        <script>
        (() => {{
            const MOBILE_MAX_WIDTH = 600;

            function fixSuffixPronounControl() {{
                if (window.parent.innerWidth > MOBILE_MAX_WIDTH) {{
                    return;
                }}

                const card = window.parent.document.querySelector(
                    ".st-key-{card_key}"
                );

                if (!card) {{
                    return;
                }}

                const control = card.querySelector(
                    'div[data-testid="stButtonGroup"]'
                );

                if (!control) {{
                    return;
                }}

                const group = control.querySelector(
                    '[role="radiogroup"]'
                );

                if (!group) {{
                    return;
                }}

                group.style.setProperty(
                    "display",
                    "flex",
                    "important"
                );
                group.style.setProperty(
                    "flex-wrap",
                    "nowrap",
                    "important"
                );
                group.style.setProperty(
                    "width",
                    "100%",
                    "important"
                );
                group.style.setProperty(
                    "max-width",
                    "100%",
                    "important"
                );
                group.style.setProperty(
                    "gap",
                    "2px",
                    "important"
                );

                Array.from(group.children).forEach((item) => {{
                    item.style.setProperty(
                        "flex",
                        "1 1 0",
                        "important"
                    );
                    item.style.setProperty(
                        "min-width",
                        "0",
                        "important"
                    );
                    // width: 0 combined with flex-grow forces every
                    // item to end up the same size regardless of its
                    // label's natural content width.
                    item.style.setProperty(
                        "width",
                        "0",
                        "important"
                    );
                    item.style.setProperty(
                        "max-width",
                        "none",
                        "important"
                    );
                    item.style.setProperty(
                        "margin",
                        "0",
                        "important"
                    );
                }});

                group.querySelectorAll("*").forEach((element) => {{
                    element.style.setProperty(
                        "box-sizing",
                        "border-box",
                        "important"
                    );
                }});

                group.querySelectorAll("button").forEach((button) => {{
                    button.style.setProperty(
                        "width",
                        "100%",
                        "important"
                    );
                    button.style.setProperty(
                        "min-width",
                        "0",
                        "important"
                    );
                    button.style.setProperty(
                        "padding-left",
                        "0",
                        "important"
                    );
                    button.style.setProperty(
                        "padding-right",
                        "0",
                        "important"
                    );
                    button.style.setProperty(
                        "font-size",
                        "11px",
                        "important"
                    );
                    button.style.setProperty(
                        "white-space",
                        "nowrap",
                        "important"
                    );
                }});
            }}

            fixSuffixPronounControl();

            const observer = new MutationObserver(() => {{
                fixSuffixPronounControl();
            }});

            observer.observe(
                window.parent.document.body,
                {{
                    childList: true,
                    subtree: true
                }}
            );

            window.parent.addEventListener(
                "resize",
                fixSuffixPronounControl
            );

            setTimeout(fixSuffixPronounControl, 100);
            setTimeout(fixSuffixPronounControl, 300);
            setTimeout(fixSuffixPronounControl, 700);
        }})();
        </script>
        """,
        height=0,
    )

    with st.container(key=card_key):
        st.markdown(
            """
<div class="learn-level">Essential Grammar</div>
<div class="learn-card-title">Suffix Pronouns</div>
<div class="learn-introduction">
    Suffix pronouns attach directly to the end of nouns, prepositions, and verbs.
    Select a form to see its meaning and grammatical use.
</div>
            """,
            unsafe_allow_html=True,
        )

        selected_form = st.segmented_control(
            "Choose a suffix pronoun",
            options=list(pronouns),
            default=".j",
            key="suffix_pronoun_selection_polished",
            label_visibility="collapsed",
            width="stretch",
        )

        selected_form = selected_form or ".j"
        selected = pronouns[selected_form]

        image_filename = selected["image_filename"]

        image_path = (
            Path(__file__).resolve().parent
            / "card_images"
            / image_filename
        )

        if not image_path.is_file():
            image_path = None

        with st.container(key=visual_key):
            if image_path is not None:
                st.image(
                    str(image_path),
                    width="content",
                )

            else:
                st.markdown(
                    f"""
<div class="glyph-placeholder" role="img" aria-label="Glyph image unavailable for {selected_form}">
    <div class="placeholder-form">{selected_form}</div>
    <div class="placeholder-label">Glyph image unavailable</div>
</div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown(
            f"""
<div class="option-title">{selected["grammar"]}</div>
<div class="option-text">
    <strong>{selected_form}</strong> means {selected["meaning"]}.<br>
    <span class="pronoun-example">
        <strong>Example:</strong> {selected["example"]}
    </span>
</div>
            """,
            unsafe_allow_html=True,
        )

        with st.container(key=key_point_key):
            st.markdown(
                selected["note"],
                unsafe_allow_html=True,
            )