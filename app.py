# !wget -nc https://raw.githubusercontent.com/baobuiquang/datasets/main/sample.xlsx >& /dev/null
# !pip install gradio==4.21.0 >& /dev/null

# ==============================
# ========== HARDCODE ==========
X_LIST_NAME = "Tên chỉ số"

# ==============================
# ========== PACKAGES ==========
import gradio as gr # gradio==4.21.0
import pandas as pd
import numpy as np
import torch
import time
import re
from transformers import AutoTokenizer, AutoModel
from datetime import datetime, timedelta
from dateparser.search import search_dates
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
                f"ngày {e.strftime('%d').lstrip('0')} tháng {e.strftime('%m').lstrip('0')} năm {e.strftime('%Y')} {e.strftime('%d').lstrip('0')}/{e.strftime('%m').lstrip('0')}/{e.strftime('%Y')} {e.strftime('%d')}/{e.strftime('%m')}/{e.strftime('%Y')}"
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
    x_list = list(df[X_LIST_NAME])
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

    # Small preprocess the message to handle unclear cases (ex: "tháng này")
    extra_information_for_special_cases = ""
    extra_information_for_special_cases_flag = False
    unclear_cases = [
        # Case 0: -> DAY, MONTH, YEAR
        [
            "ngày này" , "ngày hiện tại", "ngày hôm nay", "hôm nay", "hôm này", "ngày nay", "ngày hiện nay",
            "bây giờ", "hiện giờ", "hiện nay", "hiện tại", "thời điểm này", "thời gian này", "lúc này", "khi này",
        ],
        # Case 1: -> MONTH, YEAR
        ["tháng này", "tháng hiện tại", "tháng nay", "tháng bây giờ", "tháng đang diễn ra", "tháng hiện nay", "tháng hiện giờ"], 
        # Case 2: -> YEAR
        ["năm này", "năm hiện tại", "năm nay", "năm hiện nay"],

        # Case 3: -> DAY, MONTH, YEAR
        ["hôm qua", "hôm trước", "ngày qua", "ngày trước"],
        # Case 4: -> MONTH, YEAR
        ["tháng trước", "tháng qua", "tháng vừa rồi", "tháng đã qua"], 
        # Case 5: -> YEAR
        ["năm trước", "năm ngoái", "năm qua", "năm vừa rồi", "năm đã qua"],

        # Case 6: -> DAY, MONTH, YEAR
        ["ngày mai", "ngày sau", "ngày tới", "ngày tiếp theo", "ngày hôm sau", "ngày kế tiếp", "ngày sắp tới"],
        # Case 7: -> MONTH, YEAR
        ["tháng sau", "tháng tới", "tháng tiếp theo", "tháng kế tiếp", "tháng sắp tới"], 
        # Case 8: -> YEAR
        ["năm sau", "năm tới", "năm tiếp theo", "năm kế tiếp", "năm sắp tới"],
    ]
    for i in range(len(unclear_cases)):
        for u in range(len(unclear_cases[i])):
            if unclear_cases[i][u] in question:
                # Flag
                extra_information_for_special_cases_flag = True
                # Get the current time data
                current_time = datetime.now()
                target_time = datetime.now() # Just pre-define
                # Handle specific cases
                if i in [0, 1, 2]:
                    target_time = current_time # No change
                elif i == 3:
                    target_time = current_time - timedelta(days = 1)
                elif i == 4:
                    target_time = current_time - timedelta(days = 30)
                elif i == 5:
                    target_time = current_time - timedelta(days = 365)
                elif i == 6:
                    target_time = current_time + timedelta(days = 1)
                elif i == 7:
                    target_time = current_time + timedelta(days = 30)
                elif i == 8:
                    target_time = current_time + timedelta(days = 365)
                # Extract time to day, month, year
                day = str(target_time.strftime('%d').lstrip(''))
                month = str(target_time.strftime('%m').lstrip(''))
                year = str(target_time.strftime('%Y').lstrip(''))
                # Handle specific cases
                if i in [0, 3, 6]:
                    extra_information_for_special_cases = f"Ngày {day} tháng {month} năm {year}"
                elif i in [1, 4, 7]:
                    extra_information_for_special_cases = f"Tháng {month} năm {year}"
                elif i in [2, 5, 8]:
                    extra_information_for_special_cases = f"Năm {year}"
    if extra_information_for_special_cases_flag == True:
        question = extra_information_for_special_cases + " " + question

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
    x_text = str(df.loc[x_index, 'Tên chỉ số'])
    y_text = str(df.columns[y_index])

    # Not related but extract datetime in the question if any
    extracted_date = search_dates(message)[0][1].strftime('%d/%m/%Y') if search_dates(message) != None else ' '
    print_extracted_date = bool(re.search(r'\d', message))

    # Small adjustment for better print
    if y_text.count('/') == 4:
        y_text = y_text[-10:] # If y_text is preprocessed datetime format, trim it

    # Just add some text to warn users
    eval_text_1 = ""
    eval_text_color = ""
    eval_text_2 = ""
    eval_text_sub_title = ""
    if x_score <= 0.875 or y_score <= 0.875:
        eval_text_sub_title = "Cảnh báo:"
        eval_text_1 = "Độ tương quan thấp"
        eval_text_2 = "opacity: 0.9;"
        eval_text_color = "orange"
    if x_score <= 0.865 or y_score <= 0.865:
        eval_text_sub_title = "Cảnh báo:"
        eval_text_1 = "⚠️ Độ tương quan rất thấp ⚠️"
        eval_text_2 = "opacity: 0.8;"
        eval_text_color = "red"


    # Score display
    x_score_display = str(round((x_score - 0.85) / (1.0 - 0.85) * 100, 1))
    y_score_display = str(round((y_score - 0.85) / (1.0 - 0.85) * 100, 1))

    # Cell value
    cell_value = df.iloc[x_index, y_index]
    
    # Final print
    final_output_message = f"\
        <div style='{eval_text_2}'>\
            <div style='color: gray; font-size: 80%; font-family: courier, monospace; margin-top: 6px;'>\
                Đặc trưng trích xuất được (embedding):\
            </div>\
            • {x_text}<br>\
            • {y_text if extra_information_for_special_cases_flag == False else extra_information_for_special_cases}<br>\
            <div style='color: gray; font-size: 80%; font-family: courier, monospace; margin-top: 6px;'>\
                Đặc trưng trích xuất được (nội suy):\
            </div>\
            • {extracted_date if print_extracted_date == True else ''}<br>\
            <div style='color: gray; font-size: 80%; font-family: courier, monospace; margin-top: 6px;'>\
                Đánh giá:\
            </div>\
            Độ tương quan: [x={x_score_display}%, y={y_score_display}%]<br>\
        </div>\
        <div style='color: gray; font-size: 80%; font-family: courier, monospace; margin-top: 6px;'>\
            {eval_text_sub_title}\
        </div>\
        <div style='color: {eval_text_color}; font-weight: bold;'>\
            {eval_text_1}\
        </div>\
    "
        # <div style='color: gray; font-size: 80%; font-family: courier, monospace; margin-top: 6px;'>\
        #     Kết quả:\
        # </div>\
        # <div style='font-weight: bold;'>\
        #     {cell_value}\
        # </div>\
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