
# AI For Sign Language Translation Project 👐

*Based on @ronvidev: modelo_lstm_lsp project (version showcased on YT). Added learning with video*

This project aims to translate sign language into text using AI models. The project leverages TensorFlow and Python 3.12.2 for training and running the models. Here's an overview of the project structure and the usage sequence.

## Project Structure 📂

-   **acciones/** 📁
    -   Folder for storing action data.
-   **datos/** 📁
    -   Folder for storing data.
-   **modelos/** 📁
    -   Folder for storing models.
-   **videos/** 📁
    -   Folder for storing video files.
-   **camara.py** 📄
    -   Script to run the trained model and translate sign language in real-time.
-   **capturadora.py** 📄
    -   Script to capture images or video frames for creating datasets.
-   **constants.py** 📄
    -   Script containing constant variables used across the project.
-   **entrenamiento.py** 📄
    -   Script to train the AI model using the captured data.
-   **helpers.py** 📄
    -   Script containing helper functions used across the project.
-   **keypoints.py** 📄
    -   Script to process captured data and extract keypoints.
-   **modelo.py** 📄
    -   Script defining the model architecture.
-   **Prueba.mp4** 🎥
    -   Sample video for testing the model.
-   **tts.py** 📄
    -   Script for text-to-speech conversion.

## Usage Sequence 🔄

Follow this sequence to use the project effectively:

1.  **Capture Data** 📸
    
    -   Run `capturadora.py` to capture images or video frames for the dataset.
    -   **Note:** Use your hand to trigger the capture. Keep your hand in the frame for about 10 seconds per letter. To stop the capture, press `q`.
2.  **Extract Keypoints** 🗺️
    
    -   Run `keypoints.py` to process the captured data and extract keypoints.
3.  **Train the Model** 🧠
    
    -   Run `entrenamiento.py` to train the AI model using the extracted keypoints.
    -   The trained model will be saved as a `.h5` file.
4.  **Run the Model** 🚀
    
    -   Run `camara.py` to use the trained model for real-time sign language translation.

## Additional Notes 📝

-   **Constants and Helpers**: `constants.py` and `helpers.py` contain reusable variables and functions that are utilized in other scripts.
-   **Test Video**: `Prueba.mp4` is a sample video for testing the model.
-   **First Usage**: It is advisable to empty the folders and generate the project from scratch.

By following this guide, you should be able to capture, process, train, and run your sign language translation model efficiently. Happy coding! 🖥️✨
