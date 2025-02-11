# Emotion-Based Music Recommendation System

## Overview
This project utilizes facial emotion recognition to recommend songs that match the detected mood. It employs the DeepFace library for emotion detection and a machine learning model trained on song characteristics to enhance recommendation accuracy.

## Features
- Captures emotions using a webcam.
- Analyzes emotions using DeepFace.
- Loads and preprocesses a song dataset.
- Trains a RandomForestClassifier model to predict song emotions.
- Recommends songs based on detected emotions.

## Technologies Used
- Python
- OpenCV
- DeepFace
- NumPy
- Pandas
- Scikit-learn

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/Ashish-Ram0906/song_recommendation.git
   ```
2. Navigate to the project directory:
   ```sh
   cd song_recommendation
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
4. Ensure you have a dataset file named `song_dataset.csv` in the project directory.

## Dataset Format
The dataset should be a CSV file containing the following columns:
- `Title` - Song title
- `Artist` - Artist name
- `Emotion` - Emotion label
- `tempo`, `energy`, `danceability`, `loudness`, `valence` - Song attributes

## Usage
Run the script using:
```sh
python main.py
```
The program will capture emotions from the webcam, analyze them, and recommend songs accordingly.

## Example Output
```
Capturing the emotions...
Emotion capturing in process...
Dominant Emotion Detected: happy
Recommended Songs for happy:
  Title      | Artist      | Emotion | Predicted_Emotion
---------------------------------------------------------
  Song A     | Artist X    | happy   | happy
  Song B     | Artist Y    | happy   | happy
```

## Acknowledgments
- [DeepFace](https://github.com/serengil/deepface)
- [OpenCV](https://opencv.org/)
- [Scikit-learn](https://scikit-learn.org/)

## License
This project is open-source under the MIT License.

