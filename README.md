# Basic Music Playlist Organizer and Generator Documentation

## Project Overview
The **Basic Music Playlist Organizer and Generator** is a beginner-friendly machine learning project implemented in Google Colab. It organizes a small music dataset into playlists, predicts song genres, clusters users by preferences, and generates simple melodies. The project uses four machine learning algorithms: Decision Tree Classifier, K-Means Clustering, Logistic Regression, and a Simple Neural Network (MLP). It is designed for users new to Python and ML, leveraging small datasets and a command-line interface within Colab.

### Objectives
- Predict a song's genre (pop, rock, classical) based on features like tempo and loudness.
- Group users into clusters based on genre preferences for personalized playlist suggestions.
- Classify songs as high-energy or low-energy to refine playlist mood.
- Generate a short sequence of musical notes as a basic melody.

### Target Audience
- Beginners in Python and machine learning with basic familiarity (e.g., prior experience with K-Means, as per user context).
- Users working in Google Colab with no local setup required.

## Requirements
### Software
- **Google Colab**: Free cloud-based Python environment (access at [colab.research.google.com](https://colab.research.google.com/)).
- **Libraries**:
  - `scikit-learn`: For Decision Tree, K-Means, and Logistic Regression.
  - `pandas`, `numpy`: For data handling.
  - `librosa`: For potential audio processing (not used in MVP).
  - `tensorflow`: For Neural Network.
  - `midiutil`: For MIDI file generation.
- **Browser**: Any modern browser (e.g., Chrome, Firefox) to access Colab.

### Hardware
- Any computer with internet access (Colab runs in the cloud).
- Optional: MIDI player (e.g., VLC Media Player) to play generated melodies.

## Setup Instructions
1. **Open Google Colab**:
   - Navigate to [colab.research.google.com](https://colab.research.google.com/).
   - Create a new notebook (File > New Notebook).
   - Rename it to “Music Playlist Project” (click the title).

2. **Install Libraries**:
   - In the first cell, run:
     ```python
     !pip install scikit-learn pandas numpy librosa tensorflow midiutil
     ```
   - This installs all required libraries.

3. **Import Libraries**:
   - In the next cell, add:
     ```python
     import pandas as pd
     import numpy as np
     from sklearn.tree import DecisionTreeClassifier
     from sklearn.cluster import KMeans
     from sklearn.linear_model import LogisticRegression
     from sklearn.model_selection import train_test_split
     from sklearn.metrics import accuracy_score
     import tensorflow as tf
     from midiutil import MIDIFile
     ```

4. **Save Your Work**:
   - Colab autosaves to Google Drive.
   - Manually save a copy (File > Save a copy in Drive).
   - Download periodically (File > Download > Download .ipynb).

## Data Preparation
### Datasets
1. **Music Dataset** (for Decision Tree and Logistic Regression):
   - **Option 1: Kaggle Dataset**:
     - Download a small dataset like [Music Genre Classification](https://www.kaggle.com/datasets/insiyam/music-genre-classification).
     - Upload to Colab:
       - Click the folder icon (📁) in the left sidebar.
       - Drag and drop the CSV (e.g., `music.csv`).
   - **Option 2: Synthetic Dataset** (recommended for simplicity):
     - Create a small CSV with 10 songs:
       ```python
       music_data = pd.DataFrame({
           'tempo': [120, 80, 150, 90, 130, 110, 140, 70, 100, 160],
           'loudness': [-5, -10, -3, -8, -4, -6, -2, -12, -7, -1],
           'genre': ['pop', 'classical', 'rock', 'classical', 'pop', 'pop', 'rock', 'classical', 'pop', 'rock']
       })
       music_data.to_csv('/content/music.csv', index=False)
       ```
     - Features: `tempo` (beats per minute), `loudness` (decibels), `genre` (pop, rock, classical).

2. **User Preferences** (for K-Means):
   - Create a CSV with 5 users:
     ```python
     user_data = pd.DataFrame({
         'user_id': [1, 2, 3, 4, 5],
         'pop': [5, 2, 4, 1, 3],
         'rock': [2, 4, 1, 5, 3],
         'classical': [0, 1, 3, 2, 4]
     })
     user_data.to_csv('/content/users.csv', index=False)
     ```
     - Features: Ratings (0-5) for pop, rock, classical genres.

3. **Melody Data** (for Neural Network):
   - Use synthetic note sequences initially:
     ```python
     notes = [60, 62, 64, 65, 67] * 10
     ```
   - Later, replace with MIDI files from [Free MIDI](https://freemidi.org/) if desired.

### Preprocessing
- **Music Dataset**: Load with `pandas`, ensure `tempo`, `loudness`, `genre` columns are present.
- **User Preferences**: Load with `pandas`, use ratings as features.
- **Melody Data**: Convert notes to sequences (e.g., use 3 notes to predict the next).

## Implementation Details
The project consists of four ML algorithms implemented in separate Colab cells.

### 1. Decision Tree Classifier (Genre Prediction)
- **Purpose**: Predict a song’s genre (pop, rock, classical) based on `tempo` and `loudness`.
- **Algorithm**: `sklearn.tree.DecisionTreeClassifier`.
- **Input**: `tempo`, `loudness`.
- **Output**: Genre label (e.g., “pop”).
- **Code**:
  ```python
  # Load music data
  data = pd.read_csv('/content/music.csv')
  X = data[['tempo', 'loudness']]
  y = data['genre']
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Train Decision Tree
  model = DecisionTreeClassifier()
  model.fit(X_train, y_train)

  # Test
  predictions = model.predict(X_test)
  print(f"Accuracy: {accuracy_score(y_test, predictions)}")

  # Predict for a new song
  new_song = [[120, -5]]
  predicted_genre = model.predict(new_song)
  print(f"Predicted genre: {predicted_genre[0]}")
  ```
- **Expected Output**: Accuracy (0.5–0.8) and genre prediction (e.g., “pop”).

### 2. K-Means Clustering (User Grouping)
- **Purpose**: Group users into 3 clusters based on genre preferences.
- **Algorithm**: `sklearn.cluster.KMeans`.
- **Input**: Ratings for pop, rock, classical.
- **Output**: Cluster ID (0, 1, or 2).
- **Code**:
  ```python
  # Load user data
  users = pd.read_csv('/content/users.csv')
  X = users[['pop', 'rock', 'classical']]

  # Apply K-Means
  kmeans = KMeans(n_clusters=3, random_state=42)
  clusters = kmeans.fit_predict(X)
  users['cluster'] = clusters
  print(users[['user_id', 'cluster']])
  ```
- **Expected Output**: Table showing each user’s cluster (e.g., user 1 in cluster 0).

### 3. Logistic Regression (Energy Classification)
- **Purpose**: Classify songs as high-energy (tempo > 100) or low-energy.
- **Algorithm**: `sklearn.linear_model.LogisticRegression`.
- **Input**: `tempo`, `loudness`.
- **Output**: Energy label (0=low, 1=high).
- **Code**:
  ```python
  # Load music data
  data = pd.read_csv('/content/music.csv')
  data['energy'] = (data['tempo'] > 100).astype(int)
  X = data[['tempo', 'loudness']]
  y = data['energy']
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

  # Train Logistic Regression
  model = LogisticRegression()
  model.fit(X_train, y_train)

  # Test
  predictions = model.predict(X_test)
  print(f"Accuracy: {accuracy_score(y_test, predictions)}")

  # Predict energy
  new_song = [[120, -5]]
  energy = model.predict(new_song)
  print(f"Energy: {'High' if energy[0] == 1 else 'Low'}")
  ```
- **Expected Output**: Accuracy (0.7–1.0) and energy label (e.g., “High”).

### 4. Simple Neural Network (Melody Generation)
- **Purpose**: Generate a sequence of musical notes.
- **Algorithm**: `tensorflow.keras.Sequential` (MLP).
- **Input**: Sequence of 3 notes.
- **Output**: Next note (MIDI pitch).
- **Code**:
  ```python
  # Synthetic notes
  notes = [60, 62, 64, 65, 67] * 10
  X, y = [], []
  for i in range(len(notes) - 3):
      X.append(notes[i:i+3])
      y.append(notes[i+3])
  X, y = np.array(X), np.array(y)

  # Build MLP
  model = tf.keras.Sequential([
      tf.keras.layers.Dense(10, activation='relu', input_shape=(3,)),
      tf.keras.layers.Dense(1)
  ])
  model.compile(optimizer='adam', loss='mse')
  model.fit(X, y, epochs=50, verbose=0)

  # Generate note
  seed = np.array([[60, 62, 64]])
  new_note = model.predict(seed)[0][0]
  print(f"Next note: {int(new_note)}")

  # Save MIDI
  midi = MIDIFile(1)
  midi.addNote(0, 0, int(new_note), 0, 1, 100)
  with open('/content/melody.mid', 'wb') as f:
      midi.write(f)
  ```
- **Expected Output**: Predicted note (e.g., 65) and a `melody.mid` file.

## Usage
1. **Run the Notebook**:
   - Execute cells in order (Ctrl+Enter).
   - Start with library installation, then data creation/loading, followed by each algorithm.
2. **Interact with the System**:
   - Use a command-line-style interface:
     ```python
     print("Music Playlist Organizer")
     user_id = int(input("Enter user ID (1-5): "))
     song_tempo = float(input("Enter song tempo (e.g., 120): "))
     song_loudness = float(input("Enter song loudness (e.g., -5): "))

     # Genre prediction
     genre = model.predict([[song_tempo, song_loudness]])
     print(f"Predicted genre: {genre[0]}")

     # User cluster
     user_cluster = users[users['user_id'] == user_id]['cluster'].values[0]
     print(f"User {user_id} is in cluster {user_cluster}")

     # Energy prediction
     energy = model.predict([[song_tempo, song_loudness]])
     print(f"Energy: {'High' if energy[0] == 1 else 'Low'}")
     ```
   - Note: `input()` may require `!pip install ipywidgets` in Colab. Alternatively, hardcode values (e.g., `user_id = 1`).
3. **Check Outputs**:
   - View accuracy scores and predictions in the Colab output.
   - Download `melody.mid` (right-click in file panel > Download) and play in a MIDI player.

## Troubleshooting
- **“File not found”**: Ensure CSV files are in `/content/` (check with `!ls /content/`).
- **“Column not found”**: Verify column names with `print(data.columns)`.
- **Low accuracy**: Normal for small datasets; try a larger dataset or adjust `random_state`.
- **MIDI file doesn’t play**: Ensure the file downloaded correctly and use a compatible player (e.g., VLC).
- **Colab crashes**: Reduce dataset size or restart the runtime (Runtime > Restart runtime).

## Next Steps
- **Expand Dataset**:
  - Use a larger music dataset (e.g., FMA: Free Music Archive).
  - Add more users to `users.csv`.
- **Improve Models**:
  - Tune Decision Tree (e.g., set `max_depth=3`).
  - Increase K-Means clusters (e.g., `n_clusters=5`).
  - Add more features to Logistic Regression (e.g., spectral features from `librosa`).
  - Use MIDI files for Neural Network training.
- **Enhance Interface**:
  - Create a web app with `streamlit` (hostable in Colab).
  - Add visualizations (e.g., cluster scatter plots with `matplotlib`).
- **Learn More**:
  - Watch [StatQuest: Decision Trees](https://www.youtube.com/watch?v=7VeUPuFGJHk).
  - Read [scikit-learn documentation](https://scikit-learn.org/stable/).
  - Explore [TensorFlow tutorials](https://www.tensorflow.org/tutorials).

## License
This project is for educational purposes and uses open-source libraries under their respective licenses (e.g., MIT for `scikit-learn`, Apache 2.0 for `tensorflow`).


