# 🧠 Functional Connectomes
This repository contains a Jupyter Notebook titled `FunctionalConnectomes.ipynb`, focusing on the analysis and modeling of functional connectomes within neurobiological networksFunctional connectomes represent the dynamic patterns of neural interactions and are crucial for understanding brain function and behavior

## 📖 Overview
The notebook provides an unsupervised approach to learn dynamic affinities between neurons in live, behaving animal. It employs pairwise non-linear affinities between neuronal traces from brain-wide calcium activity, organized by non-negative tensor factorization (NTF. Each factor specifies groups of neurons most likely interacting during inferred intervals, facilitating the revelation of dynamic functional connectome. This method allows for the identification of neural motifs active during different experimental stages, such as stimulus application or spontaneous behavior.

## ✨ Features

- **Dynamic Affinity Learning:* Utilizes unsupervised learning techniques to capture time-varying neuronal interaction.
- **Non-negative Tensor Factorization (NTF):* Applies NTF to organize neuronal activity data, highlighting functional motif.
- **Community Detection:* Implements weighted community detection to infer dynamic functional connectome.

## 🚀 Getting Started

To explore the analyses and models presented:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/joaommata/Functional-Connectomes.git
   ``



2. **Install Dependencies:**
   Ensure you have Python and Jupyter Notebook installed. Install necessary packages using:
   ```bash
   pip install -r requirements.txt
   ``



3. **Run the Notebook:**
   Navigate to the repository directory and launch Jupyter Notebook:
   ```bash
   jupyter notebook FunctionalConnectomes.ipynb
   ``



## 🛠️ Usae

The notebook is structured to guide you through the process of loading neuronal activity data, applying NTF, and interpreting the resulting functional connectoe. Detailed explanations and code cells are provided for each sep.

## 🤝 Contributng

Contributions are weloe. Please fork the repository and submit a pull request with your enhancemnts.

## 📄 Licnse

This project is licensed under the MIT Liense.

## 🙏 Acknowledgents

This work is inspired by methodologies for learning dynamic representations of functional connectomes in neurobiological newors.

For more information, refer to the associated researchpaper: 