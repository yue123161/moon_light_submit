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
   - Predictions for all test datasets uploaded to the Hugging Face competition space.
   - The name of the submission must be the same in both the platforms.
#### Submission Details on GitHub 
- `main.py`: Script accepting the command `python main.py --test_path <path_to_testset>` and outputting `testset_name.csv`.  
- Folders and Files:
  - `checkpoints/`: Model checkpoints (e.g., `model_epoch_10.pth`).  
  - `source/`: All implemented files (e.g., model and loss).  
  - `solution_data/`: Predicted CSV files for the four test sets.  
  - `logs/`: Logs for each training dataset, including accuracy and loss logged every 10 epochs.  
  - `requirements.txt`: List of dependencies and Python version (e.g., `python==3.8.5`).  
  - `README.md`: Description of the solution, including model architecture, training details, and plots of training accuracy.  

5. Winning models and code must be open-sourced and publicly available.  
6. Multiple submissions per group are allowed, but the top-performing model will determine leaderboard ranking.
7. Ensure that your solution is fully reproducible. Include any random seeds or initialization details used to ensure consistent results (e.g., `torch.manual_seed()` or `np.random.seed()`) and If using a pre-trained model, include the instructions for downloading or specifying the model path.
---

### Dataset Details  

The dataset used in this competition is a subset of the publicly available Protein-Protein Association (PPA) dataset. We have selected 30% of the original dataset, focusing on 6 classes out of the 37 available in the full dataset. For more information about the PPA dataset, including its source and detailed description, please visit the official website.

---

### Evaluation Criteria  

The evaluation is performed following a hierarchical approach, with the following criteria:
1. F1 score on the test dataset provided without ground truth.
2. Accuracy  on the test dataset provided without ground truth.
3. F1 score on the test set for evaluation, not provided to participants.
4. Accuracy on the test set for evaluation, not provided to participants.
5. Inference time measured as the total time taken to generate predictions for all test graphs on our machines.


---

### Organizers  

- **Farooq Ahmad Wani (Sapienza University of Rome):** Third-year Ph.D. student focusing on noise-resilient neural networks.  
- **Maria Sofia Bucarelli (Sapienza University of Rome):** PostDoc researching generalization, noisy labels, and neural network properties.  
- **Giulia Di Teodoro (University of Pisa):** PostDoc specializing in recommendation systems and explainable AI.  
- **Andrea Giuseppe Di Francesco (Sapienza University of Rome):** Second-year Ph.D. student working on inductive biases for Graph Neural Networks.  

---

### Competition Timeline  

- **Submission Opens:** December 20, 2024  
- **Submission Deadline:** February 10, 2025  
- **Winners Announced:** February 15, 2025
- **Winners paper submission deadline:** March 1, 2025

---

We look forward to your participation in pushing the boundaries of graph learning under noisy labels. Let's innovate together!
