from pathlib import Path
import streamlit as st


def display_glyph_metadata(label, info, advanced_mode=False):
    name = info.get("name", label)
    image_path = info.get("image_path")
    category = info.get("category")
    common_meanings = info.get("common_meanings", [])
    common_words = info.get("common_words", [])

    if not advanced_mode:
        st.markdown(f"### {name}")

        if image_path and Path(image_path).exists():
            st.image(image_path, width=140)

        basic = info.get("basic", {})

        if basic.get("quick_summary"):
            st.write(basic["quick_summary"])

        if basic.get("how_it_is_used"):
            st.write("**How this sign is used:**")
            for use in basic["how_it_is_used"]:
                sign_type = use.get("type", "")
                explanation = use.get("explanation", "")

                if sign_type and explanation:
                    st.write(f"- **{sign_type}** — {explanation}")
                elif sign_type:
                    st.write(f"- **{sign_type}**")
                elif explanation:
                    st.write(f"- {explanation}")

        if common_meanings:
            st.write("**What this sign can mean:**")
            for meaning in common_meanings:
                st.write(f"- {meaning}")

        if common_words:
            st.write("**Example Egyptian words:**")
            for word in common_words:
                rough = word.get("rough_pronunciation", "")
                meaning = word.get("meaning", "")

                if rough and meaning:
                    st.write(f"- **{rough}** — {meaning}")
                elif rough:
                    st.write(f"- **{rough}**")
                elif meaning:
                    st.write(f"- {meaning}")

        if basic.get("did_you_know"):
            st.info(f"**Did you know?** {basic['did_you_know']}")

        st.divider()

    else:
        st.markdown(f"### {label} - {name}")

        if image_path and Path(image_path).exists():
            st.image(image_path, width=140)

        if category:
            st.write(f"**Category:** {category}")

        advanced = info.get("advanced", {})

        if advanced.get("phonetic_values"):
            st.write("**Phonetic Values:**")
            for value in advanced["phonetic_values"]:
                st.write(f"- {value}")

        if advanced.get("functions"):
            st.markdown("**Functions:**")
            for function in advanced["functions"]:
                st.markdown(f"- {function}")

        if common_meanings:
            st.write("**Common Meanings:**")
            for meaning in common_meanings:
                st.write(f"- {meaning}")

        if common_words:
            st.write("**Common Words:**")
            for word in common_words:
                written_word = word.get("word", "")
                rough = word.get("rough_pronunciation", "")
                meaning = word.get("meaning", "")

                if written_word and rough and meaning:
                    st.write(f"- **{written_word}** *(approx. {rough})* — {meaning}")
                elif written_word and meaning:
                    st.write(f"- **{written_word}** — {meaning}")
                elif written_word and rough:
                    st.write(
                        f"- **{written_word}** "
                        f"(roughly pronounced *{rough}*)"
                    )
                elif written_word:
                    st.write(f"- **{written_word}**")
                elif meaning:
                    st.write(f"- {meaning}")

        if advanced.get("notes"):
            st.write(f"**Notes:** {advanced['notes']}")

        st.divider()