Sketchy – AI-Powered Forensic Sketch-to-Mugshot Matching System

Overview:

Sketchy is an AI-powered forensic investigation system that matches hand-drawn suspect sketches with mugshot photographs using deep learning and cross-modal image retrieval techniques.

The system combines facial identity recognition through ArcFace with visual-semantic understanding using CLIP to bridge the gap between sketches and real-world photographs. By leveraging hybrid feature embeddings, Sketchy enables fast and scalable suspect retrieval from large mugshot databases, assisting investigators in identifying potential matches more efficiently.

Key Features:

* Hybrid ArcFace + CLIP retrieval pipeline
* Sketch-specific CLIP Adapter for improved sketch-photo alignment
* Automatic fallback to CLIP-only retrieval when face detection fails
* Fast candidate retrieval using FAISS HNSW indexing
* MongoDB-based suspect and case management
* FastAPI-powered REST API
* Modular and scalable architecture for large databases
* Weighted score fusion for improved matching accuracy

Technologies Used:

* Python
* OpenCV
* ArcFace
* CLIP
* PyTorch
* FAISS
* FastAPI
* MongoDB

How It Works:

1. Suspect Registration

* Mugshot images and suspect metadata are stored in the database.
* ArcFace extracts facial identity embeddings.
* CLIP extracts visual-semantic embeddings.
* Embeddings are indexed using FAISS for efficient retrieval.

2. Sketch Processing

* A forensic sketch is submitted as a query.
* ArcFace generates facial identity features.
* CLIP processes sketch features after image preprocessing.
* A custom adapter network improves sketch-to-photo feature alignment.

3. Candidate Retrieval

* FAISS retrieves the most similar suspects from the database.
* If facial detection fails, the system automatically switches to CLIP-only retrieval.

4. Re-Ranking and Fusion

* ArcFace and CLIP similarity scores are calculated.
* Scores are normalized and fused using weighted ranking.
* The system generates a ranked list of likely suspect matches.

My Contributions:

* Backend development and API integration
* Deep learning model integration
* Feature embedding and retrieval pipeline implementation
* Database management and indexing workflows
* System testing and evaluation

Project Type:

Collaborative college project developed as part of an AI and Computer Vision research initiative.

Purpose:

Sketchy serves as a decision-support tool for forensic investigations by generating a ranked list of potential suspects from large mugshot databases, reducing manual search effort and improving retrieval efficiency.
