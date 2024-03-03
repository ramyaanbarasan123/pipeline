# Python In-built packages
from pathlib import Path
import PIL
import pandas as pd

# External packages
import streamlit as st

# Local Modules
import settings
import helper


# Setting page layout
st.set_page_config(
    page_title="Object Detection using YOLOv8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.markdown("<h2 style='text-align:center; padding: 20px;'>Object Detection using YOLOv8</h2>",
            unsafe_allow_html=True)


# Sidebar
st.sidebar.header("Deep Learning Model ")

if st.sidebar.radio("View", ["Object Detection", "Performance Dashboard"]) == "Object Detection":
 

    confidence = float(st.sidebar.slider(
        "Select Model Confidence", 10, 100, 40)) / 100

    # Selecting Detection Or Segmentation
    model_path = Path(settings.DETECTION_MODEL)


    # Load Pre-trained ML Model
    try:
        model = helper.load_model(model_path)
    except Exception as ex:
        st.error(f"Unable to load model. Check the specified path: {model_path}")
        st.error(ex)


    st.sidebar.header("Image/Video Config")
    source_radio = st.sidebar.radio(
        "Select Source", settings.SOURCES_LIST)

    source_img = None
    # If image is selected
    if source_radio == settings.IMAGE:
        source_img = st.sidebar.file_uploader(
            "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

        col1, col2 = st.columns(2)

        with col1:
            try:
                if source_img is None:
                    default_image_path = str(settings.DEFAULT_IMAGE)
                    default_image = PIL.Image.open(default_image_path)
                    st.image(default_image_path, caption="Default Image",
                            use_column_width=True)
                else:
                    uploaded_image = PIL.Image.open(source_img)
                    st.image(source_img, caption="Uploaded Image",
                            use_column_width=True)
            except Exception as ex:
                st.error("Error occurred while opening the image.") 
                st.error(ex)

        with col2:
            if source_img is None:
                default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
                default_detected_image = PIL.Image.open(
                    default_detected_image_path)
                st.image(default_detected_image_path, caption='Detected Image',
                        use_column_width=True)
            else:
                if st.sidebar.button('Detect Objects'):
                    res = model.predict(uploaded_image,
                                        conf=confidence
                                        )
                    boxes = res[0].boxes
                    if not boxes:
                        st.markdown(
                            f"<div style='text-align:center;'><h3>Sorry, no Leaks were detected.</h3></div>",
                            unsafe_allow_html=True)
                    else:
                        res_plotted = res[0].plot()[:, :, ::-1]
                        st.image(res_plotted, caption='Detected Image',
                                use_column_width=True)
                        sizes, detections, speeds, preprocess_times, postprocess_times = helper.format_yolov8_output(res)
                        try:
                            st.markdown(
                                "<h4 style='text-align: center;'>Detection Results</h4>", unsafe_allow_html=True)
                            data = {
                                "Size": sizes,
                                "Detections": detections,
                                "Speeds": speeds,
                                "Preprocess Times": preprocess_times,
                                "Postprocess Times": postprocess_times
                            }
                            df = pd.DataFrame(data)
                            st.table(df)
                        except Exception as ex:
                            st.write("No image is uploaded yet!")

    elif source_radio == settings.VIDEO:
        helper.play_stored_video(confidence, model)

    elif source_radio == settings.WEBCAM:
        helper.play_webcam(confidence, model)


    elif source_radio == settings.YOUTUBE:
        
        df_detection_results = helper.play_youtube_video(confidence, model)
        if df_detection_results:
            st.markdown("<h3 style='text-align: center;'>Detection Results</h3>", unsafe_allow_html=True)
            st.table(df_detection_results)

    else:
        st.error("Please select a valid source type!")


else:
    helper.display_dashboard()