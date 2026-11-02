
import streamlit as st

from fastai.vision.all import *
from PIL import Image
import pandas as pd

# ------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------

st.set_page_config(
    page_title="Warsaw Skyline Detector",
    page_icon="🏙️",
    layout="centered"
)

st.title("🏙️ Warsaw Skyline Detector")

st.write(
    "Upload a photograph and let the AI model "
    "determine whether it shows the Warsaw skyline."
)

# ------------------------------------------------
# LOAD TRAINED MODEL
# ------------------------------------------------

@st.cache_resource
def load_model():
    return load_learner("warsaw_model.pkl", cpu=True)

try:
    learn = load_model()
    st.success("AI model successfully loaded!")
except Exception as e:
    st.error(f"Could not load the model: {e}")
    st.stop()

# ------------------------------------------------
# IMAGE UPLOAD
# ------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a photograph",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded photograph",
        use_container_width=True
    )

    # ------------------------------------------------
    # RUN PREDICTION
    # ------------------------------------------------

    if st.button("Analyze photograph", type="primary"):

        with st.spinner("Analyzing the photograph..."):

            img = PILImage.create(image)

            prediction, pred_idx, probabilities = learn.predict(img)

            predicted_class = str(prediction)

            confidence = float(probabilities[pred_idx])

            # ------------------------------------------------
            # DISPLAY RESULTS
            # ------------------------------------------------

            st.divider()

            st.subheader("Analysis results")

            if predicted_class == "warsaw":

                st.success("🇵🇱 Warsaw skyline detected!")

            else:

                st.warning("Warsaw skyline not detected.")

            st.metric(
                "Model confidence",
                f"{confidence * 100:.1f}%"
            )

            st.write("### Class probabilities")

            results = pd.DataFrame({
                "Category": learn.dls.vocab,
                "Probability": [
                    float(p) * 100 for p in probabilities
                ]
            })

            results = results.sort_values(
                "Probability",
                ascending=False
            )

            st.bar_chart(
                results.set_index("Category")
            )

            st.caption(
                "Probabilities are model outputs, not "
                "guaranteed measures of correctness."
            )

# ------------------------------------------------
# FOOTER
# ------------------------------------------------

st.divider()

st.caption(
    "Warsaw Skyline Detector | Built with fastai, "
    "PyTorch and Streamlit"
)
