from __future__ import annotations

from typing import Any

import streamlit as st


def render_learning_card(
    *,
    card_key: str,
    title: str,
    border_color: str,
    level: str,
    introduction: str,
    options: dict[str, dict[str, Any]],
    default_option: str,
    min_height: int = 510,
) -> None:
    """
    Render a reusable interactive learning card.

    Each option may contain:
        image_path: optional local image path
        symbol: optional Unicode symbol used when no image is supplied
        title: heading for the selected option
        text: body text, which may contain simple HTML
        key_point: optional highlighted takeaway
    """

    if not options:
        raise ValueError(
            "The learning card must have at least one option."
        )

    if default_option not in options:
        raise ValueError(
            f"default_option must be one of: {list(options.keys())}"
        )

    selector_key = f"{card_key}-selector"
    visual_key = f"{card_key}-visual"
    key_point_key = f"{card_key}-key-point"

    st.markdown(
        f"""
        <style>
            .st-key-{card_key} {{
                min-height: {min_height}px;

                padding: 24px 26px;
                margin-bottom: 24px;

                border: 2px solid {border_color};
                border-radius: 20px;

                background: #fffdf8;

                box-shadow:
                    0 6px 20px
                    rgba(31, 41, 55, 0.08);
            }}

            .st-key-{card_key}
            div[data-testid="stVerticalBlock"] {{
                gap: 0.75rem;
            }}

            .st-key-{card_key} .learn-level {{
                margin: 0 0 8px 0;

                color: #6b7280;

                font-size: 12px;
                font-weight: 700;
                letter-spacing: 0.12em;
                text-transform: uppercase;
            }}

            .st-key-{card_key} .learn-card-title {{
                margin: 0 0 10px 0;

                color: {border_color};

                font-size: 29px;
                font-weight: 750;
                line-height: 1.2;
            }}

            .st-key-{card_key} .learn-introduction {{
                margin: 0 0 14px 0;

                color: #374151;

                font-size: 16px;
                line-height: 1.6;
            }}

            .st-key-{card_key}
            div[data-testid="stSegmentedControl"] {{
                margin-top: 2px;
                margin-bottom: 10px;
            }}

            .st-key-{visual_key} {{
                display: flex;
                align-items: center;
                justify-content: center;

                min-height: 145px;

                margin: 8px 0 14px 0;
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
                min-height: 105px;

                gap: 0;
            }}

            .st-key-{visual_key}
            div[data-testid="stImage"] {{
                display: flex;
                align-items: center;
                justify-content: center;

                width: 100%;
            }}

            .st-key-{visual_key} img {{
                display: block;

                max-width: 100%;
                max-height: 105px;

                margin: 0 auto;

                object-fit: contain;
            }}

            .st-key-{visual_key} .sign-symbol {{
                margin: 0;

                color: {border_color};

                font-size: 62px;
                font-weight: 700;
                line-height: 1;
                text-align: center;
            }}

            .st-key-{card_key} .option-title {{
                margin: 3px 0 5px 0;

                color: {border_color};

                font-size: 22px;
                font-weight: 750;
                line-height: 1.3;
            }}

            .st-key-{card_key} .option-text {{
                min-height: 105px;
                margin: 0;

                color: #1f2937;

                font-size: 16px;
                line-height: 1.65;
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
            div[data-testid="stVerticalBlock"] {{
                display: flex;
                align-items: center;
                justify-content: center;

                width: 100%;
                min-height: 28px;

                gap: 0;
            }}

            .st-key-{key_point_key}
            div[data-testid="stMarkdownContainer"] {{
                display: flex;
                align-items: center;

                width: 100%;
                min-height: 28px;
            }}

            .st-key-{key_point_key}
            div[data-testid="stMarkdownContainer"] p {{
                width: 100%;

                margin: 0;
                padding: 0;

                color: #1f2937;

                font-size: 15px;
                line-height: 1.5;
            }}

            @media (max-width: 600px) {{
                .st-key-{card_key} {{
                    min-height: auto;

                    padding: 20px 18px;

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

                .st-key-{key_point_key} {{
                    min-height: 54px;
                    padding: 12px 14px;
                }}

                .st-key-{key_point_key}
                div[data-testid="stMarkdownContainer"] p {{
                    font-size: 14px;
                }}

                .st-key-{card_key}
                div[data-testid="stSegmentedControl"] {{
                    width: 100%;
                    overflow: visible;
                }}

                .st-key-{card_key}
                div[data-testid="stSegmentedControl"] > div {{
                    display: flex !important;
                    flex-wrap: nowrap !important;

                    width: 100% !important;
                    min-width: 0 !important;

                    gap: 2px !important;
                }}

                .st-key-{card_key}
                div[data-testid="stSegmentedControl"] label {{
                    flex: 1 1 0 !important;

                    width: auto !important;
                    min-width: 0 !important;
                }}

                .st-key-{card_key}
                div[data-testid="stSegmentedControl"] button {{
                    width: 100% !important;
                    min-width: 0 !important;

                    padding: 6px 2px !important;
                }}

                .st-key-{card_key}
                div[data-testid="stSegmentedControl"] p,
                .st-key-{card_key}
                div[data-testid="stSegmentedControl"] span {{
                    margin: 0 !important;

                    font-size: 10px !important;
                    line-height: 1.1 !important;
                    white-space: nowrap !important;
                }}
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.container(key=card_key):
        st.markdown(
            f"""
            <div class="learn-level">
                {level}
            </div>

            <div class="learn-card-title">
                {title}
            </div>

            <div class="learn-introduction">
                {introduction}
            </div>
            """,
            unsafe_allow_html=True,
        )

        selected_option = st.segmented_control(
            "Choose a topic",
            options=list(options.keys()),
            default=default_option,
            label_visibility="collapsed",
            key=selector_key,
            width="stretch",
        )

        if selected_option is None:
            selected_option = default_option

        selected_content = options[selected_option]

        image_path = selected_content.get("image_path")
        symbol = selected_content.get("symbol", "")
        option_title = selected_content.get(
            "title",
            selected_option,
        )
        text = selected_content.get("text", "")
        key_point = selected_content.get("key_point")

        with st.container(key=visual_key):
            if image_path:
                st.image(
                    image_path,
                    width="content",
                )

            elif symbol:
                st.markdown(
                    f"""
                    <div class="sign-symbol">
                        {symbol}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            else:
                st.markdown(
                    """
                    <div class="sign-symbol">
                        —
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown(
            f"""
            <div class="option-title">
                {option_title}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="option-text">
                {text}
            </div>
            """,
            unsafe_allow_html=True,
        )

        if key_point:
            with st.container(key=key_point_key):
                st.markdown(
                    key_point,
                    unsafe_allow_html=True,
                )