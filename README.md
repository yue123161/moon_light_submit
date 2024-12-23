# IJCNN 2025 Competition: Learning with Noisy Graph Labels
 
Handling noisy data is a persistent issue in machine learning, but it takes on a unique complexity in the context of graph structures. In domains where graph data is prevalent—such as social networks, biological networks, and financial systems—noisy labels can significantly degrade model performance, leading to unreliable predictions.  

Despite its significance, this problem remains underexplored. This competition addresses this gap by investigating graph classification under label noise. We believe it could drive major advancements in both research and real-world applications.

---

### Rules for Participation  

1. Submissions must not use any copyrighted, proprietary, or closed-source data or content.  
2. Participants are restricted to using only the datasets provided for training.  
3. Submissions can be novel solutions or modifications of existing approaches, with clear references for prior work.  
4. Submissions must include:  
   - A valid Pull Request on this GitHub page.  
   - Predictions for all 4 test datasets uploaded to the [Hugging Face competition space](https://huggingface.co/spaces/NoisyGraphLabelsChallenge/LEARNINGWITHNOISYGRAPHLABELS). The submission must be a gz folder containing 4 csv files.
   - The name of the submission must be the same in both the platforms.
   #### Submission Details on GitHub 
   - `All submissions must follow the file and folder structure below:  
   
   - **`main.py`**  
      - The script must accept the following command-line arguments:  
        ```bash
        python main.py --test_path <path_to_test.json.gz> --train_path <optional_path_to_train.json.gz>
        ```
      - **Behavior**:  
        - If `--train_path` is provided, the script must train the model using the specified `train.json.gz` file.  
        - If `--train_path` is not provided, the script should **only generate predictions** using the pre-trained model checkpoints provided.  
        - The output must be a **CSV file** named as:  
          ```
          testset_<foldername>.csv
          ```  
          Here, `<foldername>` corresponds to the dataset folder name (e.g., `A`, `B`, `C`, or `D`).  
        - Ensure the correct mapping between test and training datasets:  
          - Example: If `test.json.gz` is located in `./datasets/A/`, the script must use the pre-trained model that was trained on `./datasets/A/train.json.gz`.  
   
   - **Folder and File Naming Conventions**  
     - `checkpoints/`: Directory containing trained model checkpoints. Use filenames such as:  
       ```
       model_<foldername>_epoch_<number>.pth
       ```
       Example: `model_A_epoch_10.pth`  
     - `source/`: Directory for all implemented code (e.g., models, loss functions, data loaders).  
     - `submission/`: Folder containing the predicted CSV files for the four test sets:  
       ```
       testset_A.csv, testset_B.csv, testset_C.csv, testset_D.csv
       ```  
     - `logs/`: Log files for **each training dataset**. Include logs of accuracy and loss recorded every **10 epochs**.  
     - `requirements.txt`: A file listing all dependencies and the Python version. Example:  
       ```
       python==3.8.5
       torch==1.10.0
       numpy==1.21.0
       ```  
     - `README.md`: A clear and concise description of the solution, including:  
       - Image teaser explaning the procedure
       - Overview of the method 

5. Winning models and code must be open-sourced and publicly available.  
6. Multiple submissions per group are allowed, but the top-performing model will determine leaderboard ranking.
7. Ensure that your solution is fully reproducible. Include any random seeds or initialization details used to ensure consistent results (e.g., `torch.manual_seed()` or `np.random.seed()`) and If using a pre-trained model, include the instructions for downloading or specifying the model path.
8. Multiple submissions per team or individual are allowed. However, only the **top-performing model** will count towards the leaderboard.
9. **Submission Limits**:
   - Teams or individuals can submit **up to 4 submissions per day**. 
   - Multiple submissions are allowed, but only the **best-performing** model will count toward the leaderboard.
10. **Note:** use the zipthe folder.py to create submission.gz from submission folder for submission to hugging face.
---

### Dataset Details  

The dataset used in this competition is a subset of the publicly available Protein-Protein Association (PPA) dataset. We have selected 30% of the original dataset, focusing on 6 classes out of the 37 available in the full dataset. For more information about the PPA dataset, including its source and detailed description, please visit the [Hugging Face competition space](https://huggingface.co/spaces/NoisyGraphLabelsChallenge/LEARNINGWITHNOISYGRAPHLABELS).

---

### Evaluation Criteria  

The evaluation is performed following a hierarchical approach, where the first criterion serves as the primary measure of success. The second criterion will only be applied if two or more models achieve equal results on the first criterion and so on. 
The criteria are the following:
1. F1 score on the test dataset provided without ground truth.
2. Accuracy  on the test dataset provided without ground truth.
3. F1 score on the test set for evaluation, not provided to participants.
4. Accuracy on the test set for evaluation, not provided to participants.
5. Inference time measured as the total time taken to generate predictions for all test graphs on our machines.

---

### How to Run the Code  

This code serves as an example of how to load a dataset and utilize it effectively for training and testing a GNN model:
1. The data set can be download from https://drive.google.com/drive/folders/1Z-1JkPJ6q4C6jX4brvq1VRbJH5RPUCAk?usp=drive_link
2. The `main` file contains the implementation of the GNN model.
3. It uses the traindataset located in one of the data folders (A, B, C, or D) based on the `path_train` argument.
4. The GNN model is trained on the specified traindataset from the folder corresponding to the `path_train` argument.
5. After training, the code generates a CSV file for the test dataset, named based on the `test_path` argument.
6. For example, if `test_path` points to folder B, the output file will be named `testset_B.csv`.
7. If only the `test_path` argument is provided , the code should generate the respective test dataset’s CSV file using the pre-trained model.( This functionality is for you to implement).

---

### Organizers  

- **Farooq Ahmad Wani (Sapienza University of Rome):** Third-year Ph.D. student focusing on noise-resilient neural networks.  
- **Maria Sofia Bucarelli (Sapienza University of Rome):** PostDoc researching generalization, noisy labels, and neural network properties.  
- **Giulia Di Teodoro (University of Pisa):** PostDoc specializing in recommendation systems and explainable AI.  
- **Andrea Giuseppe Di Francesco (Sapienza University of Rome):** Second-year Ph.D. student working on inductive biases for Graph Neural Networks.  

---

### Competition Timeline  

- **Submission Opens:** December 23, 2024  
- **Submission Deadline:** February 10, 2025  
- **Winners Announced:** February 15, 2025
- **Winners paper submission deadline:** March 1, 2025

---

We look forward to your participation in pushing the boundaries of graph learning under noisy labels. Let's innovate together!
