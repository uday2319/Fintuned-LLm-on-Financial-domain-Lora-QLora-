**Fintuned-LLm-on-Financial-domain-Lora-QLora**

# Financial Domain LLM Fine-Tuning

Fine-tuning **TinyLlama 1.1B Chat** for financial-domain question answering using **QLoRA**.

##  **Project Overview**

This project adapts a pretrained TinyLlama model to answer basic financial questions in a structured format.

The model was fine-tuned on a custom dataset covering topics such as:

* Inflation
* SIP
* Mutual Funds
* Stocks
* Bonds
* Budgeting
* Income Tax
* GST
* Insurance
* PPF
* NPS
* Financial Planning

##  **Approach**

The project uses **QLoRA (Quantized Low-Rank Adaptation)**:

```text
TinyLlama 1.1B
      ↓
4-bit NF4 Quantization
      ↓
Prepare for k-bit Training
      ↓
LoRA Adapters
      ↓
Fine-Tuning
      ↓
Financial Domain Model
```

## 🛠️ **Technologies**

* Python
* PyTorch
* Hugging Face Transformers
* Hugging Face Datasets
* PEFT
* BitsAndBytes
* Google Colab

##**Fine-Tuning Configuration**

| Parameter           | Value               |
| ------------------- | ------------------- |
| Base Model          | TinyLlama 1.1B Chat |
| Method              | QLoRA               |
| Quantization        | 4-bit               |
| Quantization Type   | NF4                 |
| Double Quantization | Enabled             |
| Compute Type        | FP16                |
| LoRA Rank           | 8                   |
| LoRA Alpha          | 16                  |
| Target Modules      | `q_proj`, `v_proj`  |
| LoRA Dropout        | 0.1                 |
| Batch Size          | 2                   |
| Epochs              | 4                   |
| Learning Rate       | 1e-4                |
| Max Sequence Length | 512                 |

## **Training**

The model was trained for **4 epochs** on a custom dataset containing **29 financial instruction-response examples**.

Training loss decreased during training:

```text
Step 20 → 13.29
Step 40 → 5.38
Step 60 → 1.73
```

## **🚀 How to Run**

### 1. Install dependencies

```bash
pip install torchao
pip install "bitsandbytes>=0.46.1"
pip install transformers datasets peft accelerate
```

### 2. Open the notebook

Open:

```text
Financial_LLM_Finetuning.ipynb
```

in Google Colab.

### 3. Run the notebook

Execute the cells in order to:

1. Load TinyLlama
2. Apply 4-bit quantization
3. Configure LoRA
4. Prepare the dataset
5. Tokenize the data
6. Fine-tune the model
7. Generate financial responses

## 📁 **Project Structure**

```text
financial-llm-finetuning/
│
├── Financial_LLM_Finetuning.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

##**Future Improvements**

* Increase the size and quality of the training dataset
* Add a validation/test dataset
* Compare base model vs fine-tuned model
* Add evaluation metrics
* Reduce hallucinations
* Experiment with different LoRA configurations
* Compare LoRA vs QLoRA performance

