# Natural Language Q&A Chatbot

## Problem

Input:
* `data` - Example: `data/sample.xlsx`
* `question` - Example: "Tổng số hồ sơ chứng thực chữ ký vào ngày 12 tháng 1 năm 2024 là bao nhiêu?"

Expected extracted features:
* `features` - Example:
  *  "Tổng số HS chứng thực chữ ký"
  *  "12/01/2024"

Expected output:
* `answer`: Example: "165"

## Solution Approach

### Preprocessing `data`:

* Raw Data (`.XLSX`)
* ↳ Raw Dataframe (`Pandas DF`)
* ↳ Preprocessed Dataframe (`Pandas DF`)


### Feature Extracting `data` and `question`:

* Preprocessed Dataframe Data / Question (`String`)
* ↳ Embedding (`PyTorch Tensor`)

#### Model:
* Stable Model: [HF/XLM-ROBERTA-ME5-BASE](https://huggingface.co/baobuiquang/XLM-ROBERTA-ME5-BASE) (License: [MIT License](https://choosealicense.com/licenses/mit/))
  * Initialized from [xlm-roberta-base](https://huggingface.co/xlm-roberta-base) (License: [MIT License](https://choosealicense.com/licenses/mit/))


### Feature Map Down Sampling Method: [Mean Pooling](https://paperswithcode.com/method/average-pooling)

* Reduce computationally expensive -> Fast chatbot (Speed)
* Prevent overfitting -> Better answer (Accuracy)

### Measurement: [Cosine Similarity](https://en.wikipedia.org/wiki/Cosine_similarity)
* Input:
  * Embedding `a` (`PyTorch Tensor`)
  * Embedding `b` (`PyTorch Tensor`)
* Output:
  * Cosine Similarity: The cosine of the angle between the 2 non-zero vectors `a` and `b` in space.
```
cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

### Interactive UI

Chatbot's Web UI is currently built with [gradio](https://github.com/gradio-app/gradio)  (License: [Apache-2.0 License](https://choosealicense.com/licenses/apache-2.0/)).

## Example and Rough Explanation

Sample data: [sample.xlsx](https://github.com/baobuiquang/nlqna-chatbot/blob/main/data/sample.xlsx)

### Step 1. Input:
* `question` = "Tổng số hồ sơ chứng thực chữ ký vào ngày 12 tháng 1 năm 2024 là bao nhiêu?"
* `data` = `data/sample.xlsx`

|                                                         |       |                |                |                |       |
| :-----------------------------------------------------: | :---: | :------------: | :------------: | :------------: | :---: |
|                                                         |  ...  | **11/01/2024** | **12/01/2024** | **13/01/2024** |  ...  |
|                           ...                           |       |                |                |                |       |
|      **Tổng số HS chứng thực hợp đồng, giao dịch**      |       |      156       |      161       |      177       |       |
|            **Tổng số HS chứng thực chữ ký**             |       |      159       |      165       |      182       |       |
| **Tổng số HS chứng thực việc sửa đổi, bổ sung, hủy bỏ** |       |      162       |      169       |      187       |       |
|                           ...                           |       |                |                |                |       |

### Step 2. Feature Extraction:

* `question` -> `question_embedding` (`PyTorch Tensor`)
* `data` -> `data_embeddings` (Map of `PyTorch Tensors`)

|                     |       |                     |                     |                     |       |
| :-----------------: | :---: | :-----------------: | :-----------------: | :-----------------: | :---: |
|                     |  ...  | ***\<PT Tensor\>*** | ***\<PT Tensor\>*** | ***\<PT Tensor\>*** |  ...  |
|         ...         |       |                     |                     |                     |       |
| ***\<PT Tensor\>*** |       |         156         |         161         |         177         |       |
| ***\<PT Tensor\>*** |       |         159         |         165         |         182         |       |
| ***\<PT Tensor\>*** |       |         162         |         169         |         187         |       |
|         ...         |       |                     |                     |                     |       |

### Step 3. Measurement Calculation:

Calculate the Cosine Similarity between `question_embedding` and `data_embeddings`.

|                 |       |                 |                 |                 |       |
| :-------------: | :---: | :-------------: | :-------------: | :-------------: | :---: |
|                 |  ...  | ***{cos_sim}*** | ***{cos_sim}*** | ***{cos_sim}*** |  ...  |
|       ...       |       |                 |                 |                 |       |
| ***{cos_sim}*** |       |       156       |       161       |       177       |       |
| ***{cos_sim}*** |       |       159       |       165       |       182       |       |
| ***{cos_sim}*** |       |       162       |       169       |       187       |       |
|       ...       |       |                 |                 |                 |       |

### Step 4. Output:

Find the highest Cosine Similarity in horizontal and vertical axis to determine the cell for final answer.

|                                |       |             |                                |             |       |
| :----------------------------: | :---: | :---------: | :----------------------------: | :---------: | :---: |
|                                |  ...  | *{cos_sim}* | ***{best_cos_sim_x_axis}*** | *{cos_sim}* |  ...  |
|              ...               |       |             |                                |             |       |
|          *{cos_sim}*           |       |     156     |              161               |     177     |       |
| ***{best_cos_sim_y_axis}*** |       |     159     |           ***165***            |     182     |       |
|          *{cos_sim}*           |       |     162     |              169               |     187     |       |
|              ...               |       |             |                                |             |       |

Output the answer (cell value): "165"

## Demo

https://github.com/baobuiquang/nlqna-chatbot/assets/60503568/57621579-6a58-4638-9644-b4e482ac975e

## Instructions (Recommended workflow)

### Installation

Prerequisites:
* [Python 3.11.x](https://www.python.org/downloads/release/python-3117/)
* [Git](https://git-scm.com/downloads)

Clone [this repository](https://github.com/baobuiquang/nlqna-chatbot):
```
git clone https://github.com/baobuiquang/nlqna-chatbot.git
cd nlqna-chatbot
```

Create virtual environment:
```
python -m venv venv
```

Activate virtual environment:
```
venv\Scripts\activate
```

Upgrade `pip` command:
```
python.exe -m pip install --upgrade pip
```

Install [required packages/libraries](https://github.com/baobuiquang/nlqna-chatbot/blob/main/requirements.txt):
```
pip install -r requirements.txt
```

Deactivate virtual environment:
```
deactivate
```

### Start chatbot

Activate virtual environment:
```
venv\Scripts\activate
```

Run chatbot app:
```
python app.py
```

Wait until the terminal print something like this:
```
...\nlqna-chatbot> python app.py
Running on local URL:  http://127.0.0.1:7860
To create a public link, set `share=True` in `launch()`.
```

Now chatbot can be accessed from [http://127.0.0.1:7860](http://127.0.0.1:7860).

### Stop chatbot

Press `Ctrl + C` in the terminal to close the chatbot server.

Deactivate virtual environment:
```
deactivate
```
