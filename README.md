# E-Verify: A Scalable Embedding-based Factuality Verification Framework

E-Verify is a lightweight framework designed to address the challenge of scalable factuality verification for Large Language Models (LLMs). Traditional approaches to factuality verification, such as the widely adopted *Decompose-Then-Verify* paradigm, rely on costly pairwise reasoning and natural language inference (NLI) models, which become computationally expensive and inefficient for long text generations.

We propose a novel *Decompose-Embed-Interact* paradigm that transforms factuality verification from text-level reasoning into efficient embedding-based alignment. By decomposing generated content into atomic facts, encoding these facts into dense vectors, and verifying them through lightweight interactions in embedding space, E-Verify significantly improves scalability and efficiency while maintaining competitive accuracy.

### Getting Started

####  **Setting up**

This project uses **Git LFS** to manage large files, so make sure to clone the repository using `git lfs`:

```bash
git lfs clone https://github.com/lzy248/e-verify.git
```

After cloning the repository, navigate to the project directory and install the necessary dependencies using `requirements.txt`:

```bash
cd e-verify
pip install -r requirements.txt
```

####  **Training the Decomposer**

The **Decomposer** is responsible for efficiently decomposing long-form text into atomic facts. To train the Decomposer, we use **LLaMAFactory**, a framework designed for efficient fine-tuning. The training data is located in `data/decompose_train_data/train_alpaca.json`.

For detailed instructions on training the Decomposer, please refer to [LLaMAFactory](https://github.com/hiyouga/LLaMA-Factory). 


#### **Training the Embedder and MFIM**

Once the Decomposer is trained, the next step is to train the **Embedder** and the **Multi-Feature Interaction Module (MFIM)**. These two modules are trained together to optimize factuality verification performance. To train both modules, execute the following command:

```bash
python train/train_sts_decom.py
```

### **Performance Testing**

After training, you can evaluate the performance of the **E-Verify** framework using the following scripts.


To evaluate the performance of the system, use the **pipeline.py** and **eval.py** script:

```bash
python evaluate/pipeline.py
python evaluate/eval.py
```


To test the efficiency and speed of the system, use the **eval\_speed.py** script:

```bash
python evaluate/eval_speed.py
```
