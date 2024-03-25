# Natural Language Q&A Chatbot

## Problem

Input:
* `data` - Example: `data/sample.xlsx`
* `question` - Example: "Tổng số hồ sơ chứng thực chữ ký vào ngày 12 tháng 1 năm 2024 là bao nhiêu?"

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

## Example and Explanation

### 1. Input:
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

### 2. Feature Extraction

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

### 3. Measurement Calculation

Calculate the Cosine Similarity between `question_embedding` and `data_embeddings`.

|                 |       |                 |                 |                 |       |
| :-------------: | :---: | :-------------: | :-------------: | :-------------: | :---: |
|                 |  ...  | ***{cos_sim}*** | ***{cos_sim}*** | ***{cos_sim}*** |  ...  |
|       ...       |       |                 |                 |                 |       |
| ***{cos_sim}*** |       |       156       |       161       |       177       |       |
| ***{cos_sim}*** |       |       159       |       165       |       182       |       |
| ***{cos_sim}*** |       |       162       |       169       |       187       |       |
|       ...       |       |                 |                 |                 |       |

### 4. Output

Find the highest Cosine Similarity in horizontal and vertical axis to determine the cell for final answer.

|                                |       |             |                                |             |       |
| :----------------------------: | :---: | :---------: | :----------------------------: | :---------: | :---: |
|                                |  ...  | *{cos_sim}* | ***{highest_cos_sim_x_axis}*** | *{cos_sim}* |  ...  |
|              ...               |       |             |                                |             |       |
|          *{cos_sim}*           |       |     156     |              161               |     177     |       |
| ***{highest_cos_sim_y_axis}*** |       |     159     |           ***165***            |     182     |       |
|          *{cos_sim}*           |       |     162     |              169               |     187     |       |
|              ...               |       |             |                                |             |       |

Output the answer (cell value): "165"

## Installation

TODO

## Deployment

TODO