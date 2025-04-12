import streamlit as st
import os
from openai import OpenAI
from groq import Groq
from dotenv import load_dotenv
from PIL import Image
import io
import base64
from datetime import datetime
import re
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Fruit Quality Control System", page_icon="🍎", layout="wide"
)

# Load environment variables
load_dotenv()

# Models configuration
OPENAI_MODELS = ["gpt-4o", "gpt-4o-mini"]
GROQ_MODELS = ["meta-llama/llama-4-scout-17b-16e-instruct"]

# Sidebar for API provider and model selection
st.sidebar.title("API Configuration")

# Select API Provider
api_provider = st.sidebar.selectbox("Select API Provider", ["OpenAI", "Groq"])

# Select Model based on provider
if api_provider == "OpenAI":
    model = st.sidebar.selectbox("Select OpenAI Model", OPENAI_MODELS)

    # Get API key from environment or allow user input for OpenAI
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        openai_api_key = st.sidebar.text_input(
            "Enter your OpenAI API key:", type="password"
        )
        if not openai_api_key:
            st.warning("Please enter your OpenAI API key to proceed.")
            st.stop()

    # Initialize OpenAI client
    client = OpenAI(api_key=openai_api_key)

else:  # Groq
    model = st.sidebar.selectbox("Select Groq Model", GROQ_MODELS)

    # Get API key from environment or allow user input for Groq
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        groq_api_key = st.sidebar.text_input(
            "Enter your Groq API key:", type="password"
        )
        if not groq_api_key:
            st.warning("Please enter your Groq API key to proceed.")
            st.stop()

    # Initialize Groq client
    client = Groq(api_key=groq_api_key)


def check_image_authenticity(image, filename):
    """
    Check if the image is authentic (not from internet, taken recently, not AI-generated)
    Returns a list of detected issues
    """
    issues = []

    # Check filename for stock photo patterns
    stock_keywords = [
        "stock",
        "shutterstock",
        "adobe",
        "istockphoto",
        "getty",
        "pixabay",
        "unsplash",
        "pexels",
        "download",
        "freepik",
        "depositphotos",
        "dreamstime",
        "123rf",
    ]

    if filename:
        filename_lower = filename.lower()
        for keyword in stock_keywords:
            if keyword in filename_lower:
                issues.append(
                    f"Filename contains '{keyword}' suggesting a stock or downloaded image"
                )
                break

        # Check for patterns like IMG_20240101_123456.jpg from digital cameras
        if not re.search(r"(img|dsc|dcim|pic|photo)_\d{6,}", filename_lower):
            issues.append("Filename doesn't match typical camera-generated pattern")

    # Check image metadata for creation date
    try:
        exif_data = image.getexif()
        # Check for DateTimeOriginal (tag 36867) or DateTime (tag 306)
        datetime_original = exif_data.get(36867)
        datetime_fallback = exif_data.get(306)

        if datetime_original or datetime_fallback:
            date_str = datetime_original if datetime_original else datetime_fallback
            try:
                # Parse the EXIF date format (YYYY:MM:DD HH:MM:SS)
                captured_time = datetime.strptime(date_str, "%Y:%m:%d %H:%M:%S")
                current_time = datetime.now()
                time_diff = current_time - captured_time

                # Check if image is more than 10 minutes old
                if time_diff.total_seconds() > (10 * 60):
                    issues.append(
                        f"Image was taken {time_diff.days} days and {int(time_diff.seconds / 3600)} hours ago"
                    )
            except Exception:
                pass
        else:
            issues.append("No creation date found in image metadata")
    except Exception:
        issues.append("Unable to access image metadata")

    # Check for professional backgrounds or studio setup
    try:
        # Convert to numpy array for analysis
        img_array = np.array(image)

        # Check for uniform background (common in studio photos)
        if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
            # Sample edges of the image
            edges = [
                img_array[0, :, :],  # top edge
                img_array[-1, :, :],  # bottom edge
                img_array[:, 0, :],  # left edge
                img_array[:, -1, :],  # right edge
            ]

            # Check if edges have low variance (suggests uniform studio background)
            for edge in edges:
                if np.var(edge) < 500:  # Low variance threshold
                    issues.append(
                        "Image has uniform background suggesting studio photography"
                    )
                    break
    except Exception:
        pass

    return issues


def get_image_analysis(image_data, filename="", fruit_type=None):
    """Send the image to the selected model for analysis"""
    try:
        # Check for authenticity issues
        authenticity_issues = check_image_authenticity(image_data, filename)

        # Get the original image format
        img_format = image_data.format if image_data.format else "JPEG"
        content_type = f"image/{img_format.lower()}"

        # Convert PIL Image to bytes
        buffered = io.BytesIO()

        # If image is PNG with transparency, convert to RGB first to avoid issues
        if img_format.upper() == "PNG" and image_data.mode == "RGBA":
            image_data = image_data.convert("RGB")

        # Save the image in its original format
        image_data.save(buffered, format=img_format)
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # Prepare the prompt for analysis
        prompt = """
        Analyze this fruit image and provide:
        1. Fruit type identification
        2. Quality assessment (Excellent, Good, Fair, Poor)
        3. Detailed condition description (color, ripeness, visible defects, mold, bruises, etc.)
        4. A score from 0-100 (where 100 is perfect quality)
        5. Refund recommendation
        6. Reasoning for the recommendation
        
        CRITICAL RULE: Image authenticity must be categorized as one of the following:
        - "Original" - Appears to be an authentic photo taken by the user
        - "Stock Photo" - Appears to be a professional or stock photograph
        - "Contains Watermarks" - Has watermarks or text overlays
        - "AI-Generated" - Shows signs of AI generation
        - "From Internet" - Appears to be downloaded from the internet
        - "Studio Photo" - Has professional lighting, backdrop, or studio setup
        
        REFUND POLICY: If Image Authenticity is ANYTHING other than "Original", 
        the refund recommendation MUST be "No" regardless of fruit quality.
        
        Also carefully examine:
        - The background of the image (natural vs studio/professional setting)
        - Any digital artifacts suggesting internet download or AI generation
        - Consistent lighting and natural shadows vs professional lighting
        """

        # Include user-provided fruit type if available
        if fruit_type:
            prompt += f"\n\nNote: The user has indicated this is a {fruit_type}. Please consider this in your analysis."

        # Add detected issues to the prompt if any
        if authenticity_issues:
            issues_text = "\n".join([f"- {issue}" for issue in authenticity_issues])
            prompt += f"\n\nThe system has flagged these potential issues with the image:\n{issues_text}\n\nConsider these in your analysis and recommendation."

        prompt += """
        Format your response as:
        - Fruit Type: [type]
        - Quality: [quality level]
        - Condition: [detailed description]
        - Quality Score: [score]/100
        - Image Authenticity: [Original/Stock Photo/Contains Watermarks/AI-Generated/From Internet/Studio Photo]
        - Refund Recommended: [Yes/No]
        - Reasoning: [explanation including authenticity concerns if present]
        
        REMEMBER: If Image Authenticity is anything other than "Original", Refund Recommended MUST be "No".
        """

        # Call the appropriate model based on the selected provider
        if api_provider == "OpenAI":
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a fruit quality control expert. Your task is to analyze fruit images, detect authenticity issues (stock photos, watermarks, AI-generated images), and provide detailed quality assessment.",
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{content_type};base64,{img_str}"
                                },
                            },
                        ],
                    },
                ],
                max_tokens=1000,
            )
            return response.choices[0].message.content

        else:  # Groq
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{content_type};base64,{img_str}"
                                },
                            },
                        ],
                    },
                ],
                temperature=1,
                max_completion_tokens=1000,
            )
            return response.choices[0].message.content

    except Exception as e:
        return f"Error analyzing image: {str(e)}"


def parse_analysis(analysis_text):
    """Parse the analysis text to extract structured data"""
    lines = analysis_text.strip().split("\n")
    result = {}

    for line in lines:
        if ": " in line:
            key, value = line.split(": ", 1)
            key = key.strip("- ")
            result[key] = value

    # Enforce the refund policy: If image is not Original, refund must be No
    if "Image Authenticity" in result and "Refund Recommended" in result:
        if result["Image Authenticity"].lower() != "original":
            result["Refund Recommended"] = "No"
            if "Reasoning" in result:
                if not "non-original image" in result["Reasoning"].lower():
                    result["Reasoning"] += (
                        " (Note: Refund denied because the image is classified as non-original.)"
                    )

    return result


def main():
    st.title("🍎 Fruit Quality Control System")
    st.write(
        "Upload an image of a fruit to analyze its quality and get a refund recommendation"
    )

    # Initialize session state for analysis results and errors
    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None
    if "analysis_error" not in st.session_state:
        st.session_state.analysis_error = None

    # Add a reset button in the sidebar
    if st.sidebar.button("Reset Analysis"):
        st.session_state.analysis_result = None
        st.session_state.analysis_error = None
        st.rerun()

    # Optional fruit type input
    fruit_type = st.text_input("Fruit Type (Optional)", "")

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image file", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        filename = uploaded_file.name if hasattr(uploaded_file, "name") else ""

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)

        with col2:
            # Add a button to analyze the image
            if st.button("Analyze Fruit Quality"):
                # Clear previous results and errors
                st.session_state.analysis_result = None
                st.session_state.analysis_error = None

                with st.spinner(f"Analyzing image using {api_provider} {model}..."):
                    try:
                        # Get analysis from selected model, passing optional fruit type
                        analysis = get_image_analysis(image, filename, fruit_type)
                        if analysis.startswith("Error analyzing image:"):
                            st.session_state.analysis_error = analysis
                        else:
                            st.session_state.analysis_result = analysis
                    except Exception as e:
                        st.session_state.analysis_error = (
                            f"Error during analysis: {str(e)}"
                        )

            # Display error if there was one
            if st.session_state.analysis_error:
                st.error(st.session_state.analysis_error)
                st.info(
                    "Please try again or reset the analysis using the sidebar button."
                )

            # Display analysis results
            if st.session_state.analysis_result:
                st.subheader("Analysis Results")

                # Try to parse structured data
                try:
                    parsed = parse_analysis(st.session_state.analysis_result)

                    # Create nice UI for results
                    if "Fruit Type" in parsed:
                        st.info(f"**Fruit Type:** {parsed.get('Fruit Type')}")

                    if "Quality" in parsed:
                        st.write(f"**Quality:** {parsed.get('Quality')}")

                    if "Quality Score" in parsed:
                        score_text = parsed.get("Quality Score").split("/")[0]
                        try:
                            score = int(score_text)
                            st.progress(score / 100)
                            st.write(f"**Quality Score:** {score}/100")
                        except:
                            st.write(
                                f"**Quality Score:** {parsed.get('Quality Score')}"
                            )

                    if "Condition" in parsed:
                        st.write(f"**Condition:** {parsed.get('Condition')}")

                    if "Image Authenticity" in parsed:
                        auth_status = parsed.get("Image Authenticity")
                        if auth_status.lower() != "original":
                            st.warning(f"⚠️ **Image Authenticity Issue:** {auth_status}")
                            if "Refund Recommended" in parsed:
                                # Double-check and enforce the policy
                                if parsed.get("Refund Recommended").lower() != "no":
                                    parsed["Refund Recommended"] = "No"
                        else:
                            st.success("✅ **Image Authenticity:** Original")
                except Exception as e:
                    st.error(f"Error parsing analysis results: {str(e)}")
                    st.text("Raw analysis output:")
                    st.write(st.session_state.analysis_result)

                # Continue displaying remaining results
                try:
                    if "Refund Recommended" in parsed:
                        if parsed.get("Refund Recommended").lower() == "yes":
                            st.error("⚠️ **Refund Recommended: Yes**")
                        else:
                            st.success("✅ **Refund Recommended: No**")

                    if "Reasoning" in parsed:
                        st.write(f"**Reasoning:** {parsed.get('Reasoning')}")
                except:
                    pass  # We've already shown an error about parsing above


if __name__ == "__main__":
    main()
