# !wget -nc https://raw.githubusercontent.com/baobuiquang/datasets/main/sample.xlsx >& /dev/null
# !pip install gradio==4.21.0 >& /dev/null

# ==============================
# ========== PACKAGES ==========
import gradio as gr # gradio==4.21.0
import pandas as pd
import numpy as np
import torch
import time
from transformers import AutoTokenizer, AutoModel
from datetime import datetime
# pd.options.mode.chained_assignment = None  # default='warn'

# ===========================
# ========== FILES ==========
FILE_NAME = "data/sample.xlsx"
df_map = pd.read_excel(FILE_NAME, header=None, sheet_name=None)
df_map_sheet_names = pd.ExcelFile(FILE_NAME).sheet_names

# ============================
# ========== MODELS ==========
MODEL_NAME = "baobuiquang/XLM-ROBERTA-ME5-BASE"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# ===============================
# ========== FUNCTIONS ==========

# Text -> Embedding
def text_to_embedding(text):
    lower_text = text.lower() # Lowercasing
    encoded_input = tokenizer(lower_text, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    embedding = mean_pooling(model_output, encoded_input['attention_mask'])
    return embedding[0]

# List of Texts -> List of Embeddings
def texts_to_embeddings(list_of_texts):
    list_of_lower_texts = [t.lower() for t in list_of_texts] # Lowercasing
    encoded_input = tokenizer(list_of_lower_texts, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)
    list_of_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    return list_of_embeddings

# Mean Pooling
# - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Cosine Similarity between 2 embeddings
def cosine_similarity(a, b):
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

# Find index of the max similarity when comparing an embedding to a list
def similarity(my_embedding, list_of_embeddings):
    list_of_sim = [0] * len(list_of_embeddings)
    max_sim = -1.0
    max_sim_index = 0
    for i in range(len(list_of_embeddings)):
        cos_sim = cosine_similarity(my_embedding, list_of_embeddings[i])
        list_of_sim[i] = cos_sim
        if cos_sim > max_sim:
            max_sim = cos_sim
            max_sim_index = i
    return {"max_index": max_sim_index, "max": max_sim, "list": list_of_sim}

# ===================================
# ========== PREPROCESSING ==========

# preprocessed_df_map ----------------------------------------------------------
# - A list of dataframes (preprocessed), each dataframe contains data from 1 sheet from the XLSX file

preprocessed_df_map = []

for sheet_name in df_map_sheet_names:

    # Get sheet data
    df = pd.DataFrame(df_map[sheet_name])

    # Setup header
    header_position = df[df[0] == "#"].index[0]
    new_header = []
    for e in df.loc[header_position]:
        if isinstance(e, datetime):
            new_header.append(
                f"\
                ngày {e.strftime('%d').lstrip('0')} tháng {e.strftime('%m').lstrip('0')} năm {e.strftime('%Y')} \
                {e.strftime('%d').lstrip('0')}/{e.strftime('%m').lstrip('0')}/{e.strftime('%Y')} \
                {e.strftime('%d')}/{e.strftime('%m')}/{e.strftime('%Y')} \
                "
            )
        else:
            new_header.append(e)
    df = df.rename(columns = dict(zip(df.columns, new_header)))
    df = df.iloc[header_position+1:]

    # # Preprocess column "#" values
    # df['#'] = df['#'].replace(to_replace = r'^\d+(\.\d+)?$', value = np.nan, regex=True)
    # df['#'] = df['#'].fillna(method = 'ffill')
    # df = df.dropna(thresh = df.shape[1] * 0.25, axis = 0) # Keep rows that have at least 25% values are not NaN
    # df = df.dropna(thresh = df.shape[1] * 0.25, axis = 1) # Keep cols that have at least 25% values are not NaN
    # df = df.rename(columns={'#': 'Nhóm chỉ số'})

    # # Move column "#" to the end
    # columns = list(df.columns)
    # columns.append(columns.pop(0))
    # df = df.reindex(columns=columns)

    # General Preprocess
    df = df.reset_index(drop=True)
    df = df.fillna('No data')
    df = df.astype(str)

    # Return the preprocessed sheet
    preprocessed_df_map.append(df)

# ========================================
# ========== FEATURE EXTRACTION ==========

# embeddings_map ---------------------------------------------------------------
# - A list of pre-calculated embeddings (vectors) of x/y axis in the corresponding dataframe in the `preprocessed_df_map`

x_list_embeddings_map = []
y_list_embeddings_map = []

for i in range(len(preprocessed_df_map)):

    df = preprocessed_df_map[i]

    # HARDCODE
    x_list = list(df['Tên chỉ số'])
    y_list = list(df.columns)

    # Only need to calculate once
    x_list_embeddings = texts_to_embeddings(x_list)
    y_list_embeddings = texts_to_embeddings(y_list)

    # Return the embeddings map
    x_list_embeddings_map.append(x_list_embeddings)
    y_list_embeddings_map.append(y_list_embeddings)

# ==========================
# ========== MAIN ==========

def chatbot_mechanism(message, history, additional_input_1):
    # Clarify namings
    question = message
    sheet_id = additional_input_1
    # Select the right data
    df = preprocessed_df_map[sheet_id]
    x_list_embeddings = x_list_embeddings_map[sheet_id]
    y_list_embeddings = y_list_embeddings_map[sheet_id]
    # Find the position of the needed cell
    question_embedding = text_to_embedding(question)
    x_sim = similarity(question_embedding, x_list_embeddings)
    y_sim = similarity(question_embedding, y_list_embeddings)
    x_index = x_sim['max_index']
    y_index = y_sim['max_index']
    x_score = x_sim['max']
    y_score = y_sim['max']
    # Just add some text to warn users
    eval_text = ""
    if x_score < 0.85 or y_score < 0.85:
        eval_text = "\n⚠️ Low Cosine Similarity ⚠️"
    # Cell value
    cell_value = df.iloc[x_index, y_index]
    final_output_message = f"**{cell_value}**\n<div style='color: gray; font-size: 80%; font-family: courier, monospace;'>[x={str(round(x_score,2))}, y={str(round(y_score,2))}]{eval_text}</div>"
    return final_output_message
    # for i in range(len(final_output_message)):
    #     time.sleep(0.1)
    #     yield final_output_message[: i+1]

textbox_input = gr.Textbox(
    label = "Câu hỏi",
    placeholder = "Hãy đặt một câu hỏi",
    container = False,
    scale = 7,
)

with gr.Blocks(
    title = "CHATBOT",
    theme = gr.themes.Base(
        primary_hue = "stone",
    ),
    css = '\
        footer { visibility: hidden; display: none; }\
        [data-testid="block-label"] { visibility: hidden; display: none; } \
    ',
        # .gradio-container { max-width: 1000px !important; }\
) as app:
    with gr.Row():
        with gr.Column(scale=1):
            additional_input_1 = gr.Radio(
                choices = df_map_sheet_names,
                value = "Tư pháp", # Default
                type = "index",    # Return index instead of value
                label = "Dữ liệu",
            )
            gr.Markdown(
                """
                File dữ liệu: [`sample.xlsx`](https://view.officeapps.live.com/op/view.aspx?src=https%3A%2F%2Fraw.githubusercontent.com%2Fbaobuiquang%2Fdatasets%2Fmain%2Fsample.xlsx&wdOrigin=BROWSELINK)
                """
            )
        with gr.Column(scale=2):
            gr.ChatInterface(
                fn = chatbot_mechanism,
                chatbot = gr.Chatbot(
                    bubble_full_width = False,
                    render = False,
                    height = 450,
                ),
                textbox = textbox_input,
                additional_inputs = [
                    additional_input_1,
                ],
                retry_btn = None,
                undo_btn = "Xoá lệnh chat gần nhất",
                clear_btn = "Xoá toàn bộ đoạn chat",
                submit_btn = "Gửi",
                stop_btn = "Dừng",
                autofocus = True,
            )
        with gr.Column(scale=1):
            gr.Examples(
                label = 'Câu hỏi ví dụ (Dữ liệu "Tư pháp")',
                examples_per_page = 100,
                examples = [
                    "Tổng số hồ sơ chứng thực bản sao từ bản chính tới ngày 10/1/2024 là bao nhiêu?",     # 100
                    "15 tháng 1 năm 2024, hãy tìm dữ liệu tổng số hồ sơ chứng thực hợp đồng, giao dịch.", # 219
                    "Tổng số hồ sơ chứng thực chữ ký vào ngày 12 tháng 1 năm 2024 là bao nhiêu?",         # 165
                    "Có bao nhiêu HS chứng thực việc sửa đổi, bổ sung, hủy bỏ ngày 14/01/2024?",          # 194
                    "Tính đến ngày 11 tháng 1, 2024, số hồ sơ đăng ký kết hôn là bao nhiêu?",             # 177
                ],
                inputs = [textbox_input],
            )
            gr.Markdown(
                """
                Câu trả lời đúng cho các ví dụ: 100, 219, 165, 194, 177
                """
            )
            gr.Examples(
                label = 'Câu hỏi ví dụ (Dữ liệu "Công an huyện")',
                examples_per_page = 100,
                examples = [
                    "Số vụ phạm tội công nghệ cao ngày 19 tháng 3 năm 2024 là bao nhiêu?", # 121
                    "Tới ngày 20/3/2024, có mấy vụ án đặc biệt nghiêm trọng?",             # 208
                    "Ngày 22 tháng 3 năm 2024, có bao nhiêu người chết do TNGT",           # 273
                    "Có bao nhiêu vụ cháy cho đến ngày 24/03/2024?",                       # 437
                    "Tìm thông tin số vụ tai nạn giao thông tại ngày 18/3 năm 2024.",      # 104
                ],
                inputs = [textbox_input],
            )
            gr.Markdown(
                """
                Câu trả lời đúng cho các ví dụ: 121, 208, 273, 437, 104
                """
            )

app.launch(debug = False, share = False)