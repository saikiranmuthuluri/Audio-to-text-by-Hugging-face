# Audio-to-text-by-Hugging-face

**Audio-to-Text Conversion Using Wav2Vec2:**

This project demonstrates the implementation of automatic speech recognition (ASR) using the Wav2Vec2 model from Hugging Face's transformers library. The Wav2Vec2 model, developed by Facebook AI, is a state-of-the-art deep learning model that performs speech-to-text conversion. This repository provides a simple yet powerful solution for converting spoken language into text, enabling developers to integrate speech recognition capabilities into various applications, such as transcription services, voice assistants, and accessibility tools.


**Key Features:**

**Speech-to-Text Conversion:** The project leverages the Wav2Vec2 model, a robust ASR model trained on large datasets of speech. It transcribes audio files into text with high accuracy.

**Audio Processing with Librosa:** The librosa library is used to load, preprocess, and manage audio data. The audio files are resampled to 16 kHz to meet the model's input requirements.

**Real-Time Audio Playback:** The project also supports the playback of audio files directly within a Jupyter notebook environment using the IPython.display module.

**Pre-Trained Model:**  The Wav2Vec2 model used here comes pre-trained on large datasets, which allows the system to generate accurate transcriptions with minimal fine-tuning.

**Tokenization and Text Decoding:** After processing the audio, the model's output is decoded into human-readable text using the Wav2Vec2 tokenizer.


**Technologies Used:**

**Transformers:** Hugging Face's transformers library provides an easy interface for downloading, using, and fine-tuning state-of-the-art models like Wav2Vec2 for various natural language processing (NLP) tasks.

**Torch:** The deep learning library torch is used to perform tensor operations and run the pre-trained model.

**Librosa:**  A Python package for audio and music analysis, librosa is used to load audio files, preprocess them, and convert them into formats suitable for model inference.

**IPython:**  Used to display audio files within Jupyter notebooks, allowing real-time playback of the processed audio.

**Working Methodology :**

**Loading the Pre-Trained Model and Tokenizer:**

    The Wav2Vec2Tokenizer is used to process and tokenize the raw audio into a format suitable for the model. It converts the raw audio waveform into a sequence of input tokens that the model can understand.
    The Wav2Vec2ForCTC model is loaded with pre-trained weights, capable of generating logits, which are the raw predictions of the model.
  
**Processing Audio Input:**

    The librosa.load() function loads the audio file from the specified path and resamples it to the required sample rate of 16 kHz.
    The audio is then tokenized using the Wav2Vec2Tokenizer, converting the waveform into model-compatible input values.

**Running the Model:**

    The model predicts the speech transcription using the tokenized audio data. The output logits represent the probability distribution for each potential token (e.g., phonemes or words).
    We apply the argmax function to find the token with the highest probability at each time step, which corresponds to the most likely transcription.

**Decoding and Transcription:**

    The predicted tokens are decoded back into readable text using the tokenizer’s decode() function.

**Displaying and Playing Audio:**

    The IPython.display.Audio() function is used to play the audio directly in a Jupyter notebook environment, enabling a seamless audio transcription experience.







