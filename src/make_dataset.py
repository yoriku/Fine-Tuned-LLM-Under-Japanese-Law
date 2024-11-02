import json
import glob
import os

def extract_text(contents, 
                 start_keywords=("当裁判所の判断", "その理由は，次のとおりである。", "その理由は、次のとおりである。"), 
                 end_keywords=("判決する", "決定する")):
    start_index = -1
    for start_keyword in start_keywords:
        temp_index = contents.find(start_keyword)
        if temp_index != -1:
            start_index = temp_index - len(start_keyword) 
            break

    if start_index == -1:
        return None 

    end_index = -1
    for end_keyword in end_keywords:
        temp_index = contents.find(end_keyword, start_index)
        if temp_index != -1:
            end_index = temp_index + len(end_keyword)
            break

    if end_index == -1:
        return None

    # 必要なテキストを抽出し、空白と改行を削除
    extracted_text = contents[start_index + len(start_keyword):end_index]
    return extracted_text.replace('\n', '').replace(' ', '')

def process_json_files(file_paths):
    output_data = []

    for dialogue_id, file_path in enumerate(file_paths):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if "contents" in data and data["contents"].strip() and "case_gist" in data and data["case_gist"].strip():
            contents = data["contents"]
            case_gist = data["case_gist"].replace('\n', '').replace(' ', '')

            match = extract_text(contents)
            # print(match)
            if match is None:
                continue
            extracted_text = match
            output_data.append({
                "dialogue_id": dialogue_id,
                "conversation": [
                    {"role": "user", "content": case_gist},
                    {"role": "assistant", "content": extracted_text}
                ]
            })
            print(dialogue_id, case_gist)

    with open('output.json', 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

def get_json_file_paths(directory):
    # datasetディレクトリ以下の全てのJSONファイルのパスを取得し、list.jsonを除外
    json_files = glob.glob(os.path.join(directory, '**', '*.json'), recursive=True)
    json_files = [f for f in json_files if os.path.basename(f) != 'list.json']
    
    return json_files

if __name__ == "__main__":
    directory = 'dataset'
    json_file_paths = get_json_file_paths(directory)
    process_json_files(json_file_paths)