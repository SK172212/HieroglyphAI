import base64

import json

import uuid

from datetime import datetime

from io import BytesIO

from pathlib import Path



import streamlit as st

import streamlit.components.v1 as components

from PIL import Image

from streamlit_local_storage import LocalStorage

from cards import render_learn_tab



from camera_component import hieroglyph_camera

from display import display_glyph_metadata

from pipeline import analyze_image

from supabase_utils import (

    delete_scan,

    get_public_image_url,

    get_user_scans,

    load_scan_results,

    save_scan_to_supabase,

)



# source /Users/sohankilaru/Desktop/hieroglyphProject/pipeline_env/bin/activate

# streamlit run app.py --server.address 0.0.0.0



st.set_page_config(

    page_title="Hieroglyph AI",

    page_icon=".streamlit/favicon.png",   

    layout="wide",

)



local_storage = LocalStorage()





def load_metadata():

    with open("metadataU.json", "r", encoding="utf-8") as f:

        return json.load(f)





def decode_component_image(data_url):

    """

    Convert the Base64 image returned by the custom camera/upload

    component into a standard RGB PIL image.

    """

    if not data_url or "," not in data_url:

        raise ValueError("The image component returned invalid image data.")



    _, encoded_image = data_url.split(",", 1)

    image_bytes = base64.b64decode(encoded_image)



    return Image.open(BytesIO(image_bytes)).convert("RGB")





metadata = load_metadata()



uid = local_storage.getItem("uid")



if uid is None:

    uid = str(uuid.uuid4())

    local_storage.setItem("uid", uid)



USER_OUTPUT_DIR = Path("outputs") / uid

USER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)





def analyze_and_save_scan(image, show_spinner=True):

    """

    Save an image locally, run the detector and classifier,

    save the scan to Supabase, and preserve the image and

    results in Streamlit session state.

    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    scan_name = f"scan_{timestamp}"



    scan_folder = USER_OUTPUT_DIR / scan_name

    scan_folder.mkdir(parents=True, exist_ok=True)



    image_path = scan_folder / f"{scan_name}.png"

    image.save(image_path)



    if show_spinner:

        with st.spinner("Analyzing image..."):

            results, annot_img = analyze_image(str(image_path))

    else:

        results, annot_img = analyze_image(str(image_path))



    json_path = scan_folder / f"{scan_name}_detcls.json"



    save_scan_to_supabase(

        user_id=uid,

        scan_name=scan_name,

        image_path=image_path,

        json_path=json_path,

    )



    # Store the exact cropped area submitted to the model.

    st.session_state.home_scan_image = image.copy()

    st.session_state.home_scan_results = results

    st.session_state.home_annotated_image = annot_img

    st.session_state.home_scan_name = scan_name





def clear_pending_scan():

    st.session_state.pop("pending_scan_image", None)

    st.session_state.pop("pending_scan_submission_id", None)





def open_home():

    st.session_state.home_view = "home"





def open_camera():

    st.session_state.home_view = "camera"





def open_upload():

    st.session_state.home_view = "upload"



def scroll_to_anchor(anchor_id: str):

    components.html(

        f"""

        <script>

            setTimeout(() => {{

                const target = window.parent.document.getElementById("{anchor_id}");



                if (target) {{

                    target.scrollIntoView({{

                        behavior: "smooth",

                        block: "start"

                    }});



                    // Scroll slightly farther down

                    setTimeout(() => {{

                        window.parent.scrollBy(0,275);

                    }}, 150);

                }}

            }}, 250);

        </script>

        """,

        height=0,

    )



if "home_view" not in st.session_state:

    st.session_state.home_view = "home"




if "selected_scan_id" not in st.session_state:

    st.session_state.selected_scan_id = None





# ============================================================

# FULL-SCREEN CAMERA MODE

# ============================================================



if st.session_state.home_view == "camera":

    st.markdown(

        """

        <style>

            header[data-testid="stHeader"],

            [data-testid="stToolbar"],

            [data-testid="stSidebar"],

            [data-testid="collapsedControl"],

            [data-testid="stDecoration"],

            footer {

                display: none !important;

            }



            html,

            body,

            [data-testid="stApp"],

            [data-testid="stAppViewContainer"],

            [data-testid="stMain"],

            section.main,

            .main,

            .block-container {

                width: 100% !important;

                height: 100% !important;

                min-height: 100dvh !important;



                margin: 0 !important;

                padding: 0 !important;



                overflow: hidden !important;

                overscroll-behavior: none !important;



                background: black !important;

            }



            [data-testid="stMainBlockContainer"] {

                width: 100% !important;

                max-width: 100% !important;

                min-height: 100dvh !important;



                margin: 0 !important;

                padding: 0 !important;



                background: black !important;

            }



            div[data-testid="stVerticalBlock"] {

                gap: 0 !important;

            }



            iframe[title*="hieroglyph_camera"] {

                position: fixed !important;

                inset: 0 !important;

                z-index: 999999 !important;



                display: block !important;



                width: 100vw !important;

                height: 100dvh !important;



                margin: 0 !important;

                padding: 0 !important;



                border: none !important;

                background: black !important;

            }

        </style>

        """,

        unsafe_allow_html=True,

    )



    camera_result = hieroglyph_camera(

        mode="camera",

        key="home_hieroglyph_camera",

    )



    if camera_result:

        action = camera_result.get("action")

        submission_id = camera_result.get("submission_id")



        if action == "cancel":

            open_home()

            st.rerun()



        cropped_data_url = camera_result.get("cropped_image")



        previous_submission_id = st.session_state.get(

            "last_home_camera_submission_id"

        )



        if (

            action == "confirm"

            and cropped_data_url

            and submission_id != previous_submission_id

        ):

            try:

                camera_crop = decode_component_image(

                    cropped_data_url

                )



                # Store the confirmed crop without analyzing it yet.

                # This lets the camera close immediately.

                st.session_state.pending_scan_image = (

                    camera_crop

                )



                st.session_state.pending_scan_submission_id = (

                    submission_id

                )



                # Mark this camera submission as received so it

                # cannot be processed repeatedly on later reruns.

                st.session_state.last_home_camera_submission_id = (

                    submission_id

                )



                st.session_state.scroll_to_scan_progress = True



                open_home()

                st.rerun()



            except Exception as error:

                st.error(

                    f"The camera image could not be prepared: {error}"

                )



    # Nothing from the normal Streamlit interface is rendered

    # while the full-screen camera is active.

    st.stop()





# ============================================================

# FULL-SCREEN UPLOAD MODE

# ============================================================



if st.session_state.home_view == "upload":

    st.markdown(

        """

        <style>

            header[data-testid="stHeader"],

            [data-testid="stToolbar"],

            [data-testid="stSidebar"],

            [data-testid="collapsedControl"],

            [data-testid="stDecoration"],

            footer {

                display: none !important;

            }



            html,

            body,

            [data-testid="stApp"],

            [data-testid="stAppViewContainer"],

            [data-testid="stMain"],

            section.main,

            .main,

            .block-container {

                width: 100% !important;

                height: 100% !important;

                min-height: 100dvh !important;



                margin: 0 !important;

                padding: 0 !important;



                overflow: hidden !important;

                overscroll-behavior: none !important;



                background: black !important;

            }



            [data-testid="stMainBlockContainer"] {

                width: 100% !important;

                max-width: 100% !important;

                min-height: 100dvh !important;



                margin: 0 !important;

                padding: 0 !important;



                background: black !important;

            }



            div[data-testid="stVerticalBlock"] {

                gap: 0 !important;

            }



            iframe[title*="hieroglyph_camera"] {

                position: fixed !important;

                inset: 0 !important;

                z-index: 999999 !important;



                display: block !important;



                width: 100vw !important;

                height: 100dvh !important;



                margin: 0 !important;

                padding: 0 !important;



                border: none !important;

                background: black !important;

            }

        </style>

        """,

        unsafe_allow_html=True,

    )



    upload_result = hieroglyph_camera(

        mode="upload",

        key="home_hieroglyph_upload",

    )



    if upload_result:

        action = upload_result.get("action")

        submission_id = upload_result.get("submission_id")



        if action == "cancel":

            open_home()

            st.rerun()



        cropped_data_url = upload_result.get("cropped_image")



        previous_submission_id = st.session_state.get(

            "last_home_upload_submission_id"

        )



        if (

            action == "confirm"

            and cropped_data_url

            and submission_id != previous_submission_id

        ):

            try:

                uploaded_crop = decode_component_image(

                    cropped_data_url

                )



                # Store the confirmed crop without analyzing it yet.

                # This closes the upload component immediately and

                # performs inference on the normal Home screen.

                st.session_state.pending_scan_image = uploaded_crop

                st.session_state.pending_scan_submission_id = (

                    submission_id

                )



                # Prevent the same component submission from being

                # processed again on a later Streamlit rerun.

                st.session_state.last_home_upload_submission_id = (

                    submission_id

                )



                st.session_state.scroll_to_scan_progress = True



                open_home()

                st.rerun()



            except Exception as error:

                st.error(

                    f"The uploaded image could not be prepared: {error}"

                )



    # Nothing from the normal Streamlit interface is rendered

    # while the full-screen upload component is active.

    st.stop()





# ============================================================

# NORMAL APP INTERFACE

# ============================================================



st.markdown("""
<style>
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stHeader"] { background: #edf5f7; }
    [data-testid="stMainBlockContainer"] {
        max-width: 1120px;
        padding-top: 2.5rem;
    }
    .glyph-brand {
        color: #17354a;
        font-family: Georgia, 'Times New Roman', serif;
        font-size: clamp(2.6rem, 6vw, 4.6rem);
        font-weight: 700;
        letter-spacing: -.04em;
        line-height: 1.1;
        margin: 0;
    }
    .glyph-intro {
        color: #47677a;
        font-size: clamp(1rem, 2vw, 1.2rem);
        line-height: 1.5;
        margin: .55rem 0 1.5rem;
        max-width: 38rem;
    }
    .glyph-eyebrow {
        color: #087b83;
        font-size: .78rem;
        font-weight: 700;
        letter-spacing: .13em;
        text-transform: uppercase;
        margin-bottom: .5rem;
    }
    .stButton > button[kind="primary"] {
        background: #087b83;
        border-color: #087b83;
        color: white;
    }
    .stButton > button[kind="primary"]:hover {
        background: #075f69;
        border-color: #075f69;
        color: white;
    }
    .glyph-step {
        text-align: center;
        padding: .45rem .75rem 1rem;
    }
    .glyph-step-number {
        display: inline-grid;
        place-items: center;
        width: 3rem;
        height: 3rem;
        border: 2px solid #087b83;
        border-radius: 50%;
        color: #075f69;
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: .75rem;
    }
    .glyph-step h3 { color: #17354a; font-size: 1.1rem; margin: 0 0 .4rem; }
    .glyph-step p { color: #47677a; margin: 0; line-height: 1.5; }
    .glyph-result-guide {
        border-top: 1px solid #c9dfe6;
        margin: 2.25rem 0 1rem;
        padding-top: 1.5rem;
        text-align: center;
    }
    .glyph-result-guide h2 { color: #17354a; font-size: 1.35rem; margin: 0 0 .4rem; }
    .glyph-result-guide p {
        color: #47677a;
        line-height: 1.55;
        margin: 0 auto;
        max-width: 44rem;
    }
    @media (max-width: 640px) {
        [data-testid="stMainBlockContainer"] { padding-top: 1.25rem; }
    }
</style>
""", unsafe_allow_html=True)





@st.dialog("How HieroglyphAI works")

def welcome_popup():



    # Only this section scrolls

    with st.container(height=400):

        st.markdown("""

### What is Middle Egyptian?



**Middle Egyptian** was the language of ancient Egypt during its Middle Kingdom (about **2000–1600 BCE**) and remained the standard language for literature, religion, and monumental inscriptions for over **2,000 years**. As a result, many hieroglyphic inscriptions displayed in museums today are written in Middle Egyptian.



Hieroglyphs are more than pictures. Depending on how a sign is used, it can:



- Represent a **sound**, similar to a letter.

- Represent an entire **word or object**.

- Provide a clue about a word's **meaning** without being pronounced.



### About This App



**HieroglyphAI** uses artificial intelligence to detect, identify, and explain hieroglyphs in photographs of inscriptions. It is designed for students, museum visitors, and anyone interested in learning more about ancient Egypt.



### How to Get Started



1. Go to the **Home** tab.

2. Upload an image or take a picture of an inscription.

3. Focus the image on a small group of clearly visible hieroglyphs.

4. Analyze the image.

5. Review the detected signs and their explanations.



### Tips for Better Results



- Take the picture as straight-on as possible.

- Make sure the inscription is in focus.

- Use good lighting and limit shadows, glare, and blur.

- Keep each hieroglyph fully visible.

- Fill most of the image with the signs you want to identify.



**Please note:** This is an experimental AI model and may occasionally miss or misidentify hieroglyphs.

""")



    # This stays below the scrolling area

    if st.button(

        "Back to exploring",

        type="primary",

        width="stretch",

        key="close_app_guide",

    ):

        st.rerun()





brand_col, guide_col = st.columns([5, 1], vertical_alignment="bottom")
with brand_col:
    st.markdown(
        '<div class="glyph-eyebrow">Explore ancient Egypt</div>'
        '<h1 class="glyph-brand">HieroglyphAI</h1>'
        '<p class="glyph-intro">Photograph a few hieroglyphs and discover '
        'the signs in front of you.</p>',
        unsafe_allow_html=True,
    )
with guide_col:
    if st.button("How it works", width="stretch", key="open_app_guide"):
        welcome_popup()


advanced_mode = st.toggle(

    "Advanced Mode",

    value=False,

)



tab1, tab2, tab3 = st.tabs(

    [

        "Home",

        "Previous Scans",

        "Learn",

    ]

)





# ============================================================

# HOME TAB

# ============================================================



with tab1:



    # --------------------------------------------------------

    # HOME VIEW

    # --------------------------------------------------------



    st.header("Analyze an Inscription")



    photo_col, upload_col = st.columns(2)



    with photo_col:

        if st.button(

            "📷 Take a Picture",

            type="primary",

            width="stretch",

            key="open_camera_view",

        ):

            open_camera()

            st.rerun()



    with upload_col:

        if st.button(

            "⬆️ Upload an Image",

            width="stretch",

            key="open_upload_view",

        ):

            open_upload()

            st.rerun()



    st.markdown(
        '<h2 style="text-align:center; margin:2rem 0 1.25rem;">Scan in three steps</h2>',
        unsafe_allow_html=True,
    )
    step_cols = st.columns(3, gap="large")
    steps = (
        ("1", "Photograph", "Take a picture of an inscription or choose one from your photos."),
        ("2", "Focus", "Crop to a small group of clear, complete signs."),
        ("3", "Explore", "See the identified signs and learn what they depict and how they are used."),
    )
    for col, (number, title, description) in zip(step_cols, steps):
        with col:
            st.markdown(
                f'<div class="glyph-step"><span class="glyph-step-number">{number}</span>'
                f'<h3>{title}</h3><p>{description}</p></div>',
                unsafe_allow_html=True,
            )

    with st.expander("Photo tips for a clearer scan"):
        st.markdown("""
- Hold the camera straight on to the inscription.
- Fill most of the image with a few hieroglyphs, keeping each sign fully visible.
- Avoid glare, strong shadows, and blur.
""")


    # ----------------------------------------------------

    # PENDING CAMERA OR UPLOAD SCAN

    # ----------------------------------------------------



    pending_scan_image = st.session_state.get(

        "pending_scan_image"

    )



    if pending_scan_image is not None:

        st.markdown(

            '<div id="scan-progress-anchor"></div>',

            unsafe_allow_html=True,

        )



        if st.session_state.pop(

            "scroll_to_scan_progress",

            False,

        ):

            scroll_to_anchor("scan-progress-anchor")

        st.divider()



        st.subheader("Scanned Image")



        st.image(

            pending_scan_image,

            caption="Image area being analyzed",

            width="stretch",

        )



        try:

            # The image and spinner appear on Home while

            # inference runs instead of leaving the user

            # waiting on the camera screen.

            with st.spinner("Analyzing image..."):

                analyze_and_save_scan(

                    pending_scan_image,

                    show_spinner=False,

                )



            clear_pending_scan()



            st.session_state.scroll_to_scan_results = True



            # Rerun once more so the temporary analyzing

            # section is replaced cleanly by final results.

            st.rerun()



        except Exception as error:

            st.error(

                f"The selected image could not be analyzed: {error}"

            )



            retry_col, dismiss_col = st.columns(2)



            with retry_col:

                if st.button(

                    "Try Again",

                    type="primary",

                    width="stretch",

                    key="retry_failed_camera_scan",

                ):

                    st.rerun()



            with dismiss_col:

                if st.button(

                    "Dismiss",

                    width="stretch",

                    key="dismiss_failed_camera_scan",

                ):

                    clear_pending_scan()

                    st.rerun()



    # ----------------------------------------------------

    # COMPLETED RESULTS

    # ----------------------------------------------------



    elif "home_scan_results" in st.session_state:

        st.markdown(

            '<div id="scan-results-anchor"></div>',

            unsafe_allow_html=True,

        )



        if st.session_state.pop(

            "scroll_to_scan_results",

            False,

        ):

            scroll_to_anchor("scan-results-anchor")



        st.divider()



        st.success(

            f"Scan saved: {st.session_state.home_scan_name}"

        )



        if "home_scan_image" in st.session_state:

            st.subheader("Scanned Image")



            st.image(

                st.session_state.home_scan_image,

                caption="Image area that was analyzed",

                width="stretch",

            )



        st.subheader("Results")



        shown_signs = set()

        displayed_sign_count = 0



        for result in st.session_state.home_scan_results:

            # Rejected detector boxes do not contain classification fields.

            if not result.get("accepted_box", True):

                continue



            label = result.get("shown_label", "UNK")



            if label == "UNK" or label in shown_signs:

                continue



            shown_signs.add(label)

            displayed_sign_count += 1



            info = metadata.get(label)



            if info is None:

                st.markdown(f"### {label}")



                st.info(

                    "No metadata has been added for this sign yet."

                )



                st.divider()



            else:

                display_glyph_metadata(

                    label,

                    info,

                    advanced_mode,

                )



        if displayed_sign_count == 0:

            st.info(

                "No recognizable hieroglyphs were detected. "

                "Try taking a clearer image or selecting a "

                "different area."

            )



        with st.expander(

            "View annotated image with detected glyph boxes"

        ):

            st.image(

                st.session_state.home_annotated_image,

                caption="Detection and classification output",

                width="stretch",

            )



    st.markdown("""
<section class="glyph-result-guide" aria-labelledby="result-guide-heading">
  <h2 id="result-guide-heading">Reading your results</h2>
  <p>HieroglyphAI identifies individual signs in your photo. A sign might represent a sound,
  depict an object, or help clarify a word's meaning. Explore each sign to learn more;
  the results are not a translation of the whole inscription.</p>
</section>
""", unsafe_allow_html=True)


# ============================================================

# PREVIOUS SCANS TAB

# ============================================================



with tab2:

    st.header("Previous Scans")



    scans = get_user_scans(uid)



    if st.button(

        "🗑️ Clear All Previous Scans",

        key="clear_all_scans",

    ):

        for scan in scans:

            delete_scan(scan)



        st.session_state.selected_scan_id = None

        st.rerun()



    if st.session_state.selected_scan_id is None:

        st.subheader("All Scans")



        if len(scans) == 0:

            st.info("No scans have been saved yet.")



        else:

            for scan in scans:

                scan_name = scan["scan_name"]



                image_url = get_public_image_url(

                    scan["image_path"]

                )



                with st.container(border=True):

                    col1, col2 = st.columns([1, 3])



                    with col1:

                        st.image(

                            image_url,

                            width="stretch",

                        )



                    with col2:

                        title_col, delete_col = st.columns(

                            [6, 1]

                        )



                        with title_col:

                            st.markdown(f"### {scan_name}")



                        with delete_col:

                            if st.button(

                                "🗑️",

                                key=f"delete_{scan['id']}",

                            ):

                                delete_scan(scan)

                                st.rerun()



                        if st.button(

                            "View Results",

                            key=f"view_{scan['id']}",

                        ):

                            st.session_state.selected_scan_id = (

                                scan["id"]

                            )

                            st.rerun()



    else:

        selected_scan = None



        for scan in scans:

            if (

                scan["id"]

                == st.session_state.selected_scan_id

            ):

                selected_scan = scan

                break



        if selected_scan is None:

            st.warning("This scan could not be found.")



            if st.button(

                "Back",

                key="back_missing_scan",

            ):

                st.session_state.selected_scan_id = None

                st.rerun()



        else:

            scan_name = selected_scan["scan_name"]



            image_url = get_public_image_url(

                selected_scan["image_path"]

            )



            col1, col2 = st.columns([5, 1])



            with col1:

                st.subheader(scan_name)



            with col2:

                if st.button(

                    "X",

                    key="close_selected_scan",

                ):

                    st.session_state.selected_scan_id = None

                    st.rerun()



            st.image(

                image_url,

                caption="Original scan",

                width="stretch",

            )



            scan_results = load_scan_results(

                selected_scan["json_path"]

            )



            st.subheader("Results")



            shown_signs = set()

            displayed_sign_count = 0



            for result in scan_results:

                # Rejected detector boxes do not contain classification fields.

                if not result.get("accepted_box", True):

                    continue



                label = result.get("shown_label", "UNK")



                if label == "UNK" or label in shown_signs:

                    continue



                shown_signs.add(label)

                displayed_sign_count += 1



                info = metadata.get(label)



                if info is None:

                    st.markdown(f"### {label}")



                    st.info(

                        "No metadata has been added for "

                        "this sign yet."

                    )



                    st.divider()



                else:

                    display_glyph_metadata(

                        label,

                        info,

                        advanced_mode,

                    )



            if displayed_sign_count == 0:

                st.info(

                    "No recognizable hieroglyphs were detected "

                    "in this scan."

                )





# ============================================================

# MIDDLE EGYPTIAN CONVENTIONS TAB

# ============================================================



with tab3:

    st.header("Learn Middle Egyptian")



    st.write(

        "Explore the essential conventions used to read and "

        "understand Middle Egyptian hieroglyphs."

    )



    render_learn_tab(advanced_mode)
