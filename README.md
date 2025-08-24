# **Face Recognition for Access Control Systems**  
![Python](https://img.shields.io/badge/Python-3.12-blue)  ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)  ![OpenCV](https://img.shields.io/badge/OpenCV-Enabled-green)  ![License](https://img.shields.io/badge/License-MIT-lightgrey)  

A complete end-to-end face recognition system for **secure access control**, powered by **Siamese Neural Networks (SNN)**. This project bridges **deep learning**, **computer vision**, and **full-stack development** to create a real-world solution for biometric authentication.  

---

## **📌 Project Overview**  
The goal of this project was to develop a robust and scalable application that verifies user identities using facial features instead of traditional credentials like passwords or keycards. The system relies on a **Siamese Neural Network**, which learns a similarity function between two images rather than performing classification. This approach allows for easier scalability and adaptability, as new identities can be added without retraining the entire model.  

The project combines **a deep learning model**, a **Python backend** for image processing and verification, and a **simple frontend** for real-time interaction. It reflects the core principles of modern security systems: **accuracy**, **speed**, and **user convenience**.  

---

## **✨ Features**  
- **Deep Learning-based Face Verification** using Siamese Neural Networks.  
- **End-to-End System** with both backend and frontend components.  
- **Image Preprocessing** for better performance, including face detection and cropping.  
- **Interactive Web Interface** for capturing images and verifying identity in real time.  
- **Secure Embedding Storage** using MongoDB for fast similarity searches.  

---

## **🛠️ Tech Stack**  
- **Programming Language:** Python 3.12  
- **Deep Learning:** TensorFlow, Keras  
- **Computer Vision:** OpenCV  
- **Backend Framework:** Flask (REST API)  
- **Frontend:** HTML, JavaScript  
- **Database:** MongoDB for embedding storage  
- **Other Tools:** TensorBoard for training visualization, NumPy, Pandas  

---

## **📂 Project Structure**
```
face_recognition/
├── backend/
│   ├── app.py                # API endpoints for verification
│   ├── data_processor.py     # Handles preprocessing and dataset preparation
│   ├── face_recognizer.py    # Siamese Neural Network model definition
│   ├── train.py              # Training script
│   └── tests/                
├── frontend/
│   ├── index.html            # User interface
│   └── script.js             # Handles webcam and API calls
└── README.md
```

---

## **🔍 Model Architecture**  
The core of the system is a **Siamese Neural Network**, an architecture designed to compare two inputs and determine their similarity. Each branch of the network uses an identical **VGG16-based feature extractor**, pre-trained on ImageNet and fine-tuned for the specific task of face verification. Instead of predicting classes, the model generates a compact 512-dimensional embedding for each face.  

After extracting embeddings from two images, the network computes their **L1 distance**, capturing how similar or different they are. This difference is then passed through a sigmoid-activated dense layer, which outputs a probability score indicating whether the two images belong to the same identity.  

This architecture provides several advantages: it minimizes retraining needs when new identities are added, improves generalization through shared weights, and offers strong performance even with limited labeled data.  

---

## **📊 Results**  
The model was trained on a subset of the **DigiFace-1M dataset**, consisting of synthetic human faces with diverse poses and lighting conditions. Using **data augmentation** techniques such as rotation, zoom, and horizontal flipping, the network was trained for 10 epochs with **Adam optimizer** and a low learning rate for stability.  

The final model achieved **92% accuracy on the training set** and **90% on the validation set**, maintaining precision around **91%**. This demonstrates the model’s ability to distinguish similar and dissimilar pairs effectively. When evaluated on **real-world images** from the VGGFace2 dataset, performance dropped, highlighting the challenge of domain adaptation - an important consideration for future iterations.  

---

## **🚀 How It Works**  
The workflow begins when a user captures their image using the web interface. The image is sent to the backend, where **OpenCV** detects and crops the face to remove background noise. This processed image is passed to the Siamese Neural Network, which converts it into an embedding vector.  

To verify the user, the system retrieves embeddings of authorized users from a **vector database** and computes the similarity score. If the score exceeds a predefined threshold, the user is granted access; otherwise, access is denied. This process ensures high security without relying on passwords or physical tokens.  


