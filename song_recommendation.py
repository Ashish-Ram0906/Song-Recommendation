import cv2
import numpy as np
import pandas as pd
from deepface import DeepFace
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from collections import Counter

def detect_emotions(image):
    try:
        result = DeepFace.analyze(image, actions=['emotion'], enforce_detection=False)
        if isinstance(result, list):
            dominant_emotion = result[0]['dominant_emotion']
        else:
            dominant_emotion = result['dominant_emotion']
        return dominant_emotion.lower()
    except Exception as e:
        print(f"Error detecting emotion: {e}")
        return None

def load_song_dataset(file_path):
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    le = LabelEncoder()
    df['Emotion'] = le.fit_transform(df['Emotion'])

    X = df[['tempo', 'energy', 'danceability', 'loudness', 'valence']]
    y = df['Emotion']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, le, scaler

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def recommend_songs(facial_emotion, song_df, model, scaler, le):
    # Predict emotions for all songs
    X = song_df[['tempo', 'energy', 'danceability', 'loudness', 'valence']]
    X_scaled = scaler.transform(X)
    predicted_emotions = le.inverse_transform(model.predict(X_scaled))
    
    # Add predicted emotions to the dataframe
    song_df['Predicted_Emotion'] = predicted_emotions
    
    # Filter songs based on facial emotion and predicted emotion
    recommended_songs = song_df[(song_df['Emotion'] == facial_emotion) | (song_df['Predicted_Emotion'] == facial_emotion)]
    
    if recommended_songs.empty:
        print(f"No songs found for {facial_emotion}. Here are some random recommendations:")
        recommended_songs = song_df.sample(n=1)
    else:
        print(f"Recommended Songs for {facial_emotion}:")
    
    print(recommended_songs[['Title', 'Artist', 'Emotion', 'Predicted_Emotion']].head(5))

def main():
    # Load the song dataset
    song_df = load_song_dataset('song_dataset.csv')
    
    # Preprocess data and train the model
    X_train, X_test, y_train, y_test, le, scaler = preprocess_data(song_df)
    model = train_model(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    #print(f"Model Accuracy: {accuracy:.2f}")
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    print("capturing the emotions")
    time.sleep(5)
    
    print("emotion capturing in process")
    start_time = time.time()
    emotions = []
    
    while time.time() - start_time < 2:
        ret, frame = cap.read()
        if ret:
            cv2.imshow('Capturing Emotions', frame)
            cv2.waitKey(1)
            
            emotion = detect_emotions(frame)
            if emotion:
                emotions.append(emotion)
        else:
            print("Failed to capture image from webcam.")
            break
    
    cv2.destroyAllWindows()
    
    if emotions:
        # Get the most common emotion
        dominant_emotion = Counter(emotions).most_common(1)[0][0]
        print(f"Dominant Emotion Detected: {dominant_emotion}")
        recommend_songs(dominant_emotion, song_df, model, scaler, le)
    else:
        print("No emotions could be detected.")

    # Release the capture
    cap.release()

if __name__ == "__main__":
    main()