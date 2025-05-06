# YOLO Image Processing Web Application

This is a web application that uses YOLO (You Only Look Once) for image processing. Users can upload images through a web interface, and the application will process them using the YOLO model and display the results.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure you have the YOLO model file (`best.pt`) in the root directory of the project.

## Running the Application

1. Start the Streamlit server:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to:
```
http://localhost:5000
```

## Usage

1. Click the "Choose Image" button to select an image from your computer
2. The image will be automatically uploaded and processed
3. You'll see both the original and processed images side by side
4. The processed image will show the YOLO model's detection results

## Project Structure

- `app.py`: Main Flask application
- `templates/index.html`: Frontend web interface
- `best.pt`: YOLO model file
