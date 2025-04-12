# Fruit Quality Control System

A Streamlit web application that uses AI models to analyze fruit quality from uploaded images and provide refund recommendations.

## Project Description

This application is a proof-of-concept (POC) for integrating AI-based image analysis into customer support systems. The goal is to help determine if a customer's request for a return/refund based on fruit quality is genuine by analyzing the images they upload.

## Features

- Upload images of fruits
- Support for multiple AI providers (OpenAI and Groq)
- Optional fruit type specification
- Image authenticity verification
- AI-powered analysis of fruit quality
- Detailed assessment including:
  - Fruit identification
  - Quality level
  - Condition description
  - Quality score (0-100)
  - Image authenticity classification
  - Refund recommendation
  - Reasoning for the recommendation

## Setup Instructions

1. Clone this repository
2. Install required packages:
   ```
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the project root with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   GROQ_API_KEY=your_groq_api_key_here
   ```
   - Alternatively, you can enter your API key directly in the app

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```
2. Open the provided URL in your web browser
3. In the sidebar, select your preferred API provider (OpenAI or Groq) and model
4. Optionally specify the fruit type
5. Upload an image of a fruit
6. Click "Analyze Fruit Quality" to get the assessment
7. Use the "Reset Analysis" button in the sidebar to start over

## Available Models

### OpenAI
- gpt-4o
- gpt-4o-mini

### Groq
- meta-llama/llama-4-scout-17b-16e-instruct

## Image Authenticity

The system checks for image authenticity to prevent fraudulent refund claims:
- Analyzes image metadata
- Checks filename patterns
- Identifies studio/professional photography
- Detects stock photos, watermarks, and AI-generated images
- Automatically denies refunds for non-authentic images

## Notes

This is a proof-of-concept and may require further refinement for production use. The refund recommendation is based on both quality assessment and image authenticity verification.
