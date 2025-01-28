# Embracing Diversity: A Multi-Perspective Approach with Soft Labels

## Abstract
Prior studies show that adopting the annotation diversity shaped by different backgrounds and life experiences and incorporating them into the model learning, i.e. multi-perspective approach, contribute to the development of more responsible models.
Thus, in this paper we propose a new framework for designing and further evaluating perspective-aware models on stance detection task,in which multiple annotators assign stances based on a controversial topic. We also share a new dataset established through obtaining both human and LLM annotations. 
Results show that the multi-perspective approach yields better classification performance (higher F1-scores), outperforming the traditional approaches that use a single ground-truth, while displaying lower model confidence scores, probably due to the high level of subjectivity of the stance detection task.

## Paper Link
You can read the full paper [here](<link will be uploaded soon).

## How to Read the Code
### 1. Setup Instructions
- Clone the repository:
  ```bash
  git clone <repository-link>
  cd <repository-directory>

- Install the requriements
  ```bash 
  pip install -r requirements.txt

The code is divided in their respective folders. 

**Baseline** and **muliperspective** folders contain the code for finetuning. The files are in ipynb, so for reusability, you only need to upload the dataset path and run the cells. Please run the **data preparation notebooks** before finetuning.  

**LLM annotations** folder contains the code for LLM annotations. We used **OLMO - 7B**,**LLAMA-3 8B**, and **Mistral - 7B v0.3**. 

**gpt_summarization** contains the summarization task from the paper.


