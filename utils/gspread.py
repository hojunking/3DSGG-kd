import gspread
import re
from gspread_formatting import get_user_entered_format, format_cell_range


def read_experiment_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
        # 데이터 추출을 위한 변수 초기화
    data = {
        "Method": "",
        "Object": [],
        "Predicate": [],
        "Triplet": [],
        "Parameters": "",
        "GNN_parameters": "",
        "FLOPs": "",
        "Time": "",
        "Latency": ""
    }
    
    # 파일의 각 라인에서 필요한 정보를 추출
    for line in lines:
        if line.startswith("Experiment:"):
            data["Method"] = line.split(":")[-1].strip()
        elif "Average Latency" in line:
            data["Latency"] = line.split(":")[-1].strip()
        elif "3d obj Acc" in line:
            acc_value = round(float(line.split(":")[-1].strip()), 2)
            data["Object"].append(acc_value)
        elif "rel Acc" in line:
            acc_value = round(float(line.split(":")[-1].strip()), 2)
            data["Predicate"].append(acc_value)
        elif "3d triplet Acc" in line:
            acc_value = round(float(line.split(":")[-1].strip()), 2)
            data["Triplet"].append(acc_value)
        elif "Total Parameters:" in line and '_unst' not in data["Method"]:
            data["Parameters"] = line.split(":")[1].strip()
        elif "gnn_total" in line:
            data["GNN_parameters"] = line.split(":")[1].strip()
        elif "Total Parameters after" in line and '_unst' in data["Method"]:
            data["Parameters"] = line.split(":")[1].strip()
        elif "Total Flops:" in line:
            data["FLOPs"] = line.split(":")[1].split(" ")[1]
        elif "Total time:" in line:
            data["Time"] = line.split(":")[1].strip()
    
    return data

def extract_ratio(method_value):
        # 정규 표현식으로 처음 두 개의 숫자 추출
    numbers = re.findall(r'\d+', method_value)

    # 첫 번째와 두 번째 숫자만 추출
    if len(numbers) >= 2:
        b_value = numbers[0]  # 첫 번째 숫자
        c_value = numbers[1]  # 두 번째 숫자
    elif len(numbers) ==1:
        b_value = numbers[0]
        c_value = None
    else:
        b_value = None
        c_value = None
    
    return b_value, c_value

def copy_format_from_previous_row(sheet, dest_row):
    """빈 행 바로 앞 행의 서식을 복사하여 대상 행에 적용"""
    source_row = dest_row - 1  # 앞선 행 번호 계산
    
    # 'B'부터 'Z'까지의 단일 문자 열과 'AA'부터 'AW'까지의 두 자리 문자 열 생성
    columns = [chr(i) for i in range(ord('B'), ord('Z') + 1)]
    
    for col in columns:
        source_cell = f'{col}{source_row}'
        dest_cell = f'{col}{dest_row}'
        fmt = get_user_entered_format(sheet, source_cell)  # 원본 서식 가져오기
        if fmt:
            format_cell_range(sheet, dest_cell, fmt)  # 대상 셀에 서식 적용

def save_gspread(result_path, flag):
    # 데이터 추출
    result_data = read_experiment_data(result_path)
    # Google Sheet 연동
    gc = gspread.service_account()
    sh = gc.open("light-gnn-ex-results")
    sheet = sh.worksheet(flag)

    #sheet.add_cols(30)
    # 빈 행 찾기
    all_values_in_column_b = sheet.col_values(2)  # B열의 모든 값 가져오기
    values_from_second_row  = all_values_in_column_b[1:]  # 인덱스 6부터 가져옴 (7행부터 시작)

    row_number = 2 if not values_from_second_row  else len(values_from_second_row ) + 2
    
    # 서식 복사
    copy_format_from_previous_row(sheet, row_number)
    print(f"Insert in row number: {row_number}", end=" ")
    print(result_data["Method"])
    # 배치 업데이트를 위한 데이터 준비
    updates = []
    
    # 다른 데이터 입력
    updates.append({'range': f'B{row_number}', 'values': [[result_data["Method"]]]})
    updates.append({'range': f'C{row_number}', 'values': [[result_data["Object"][0]]]})
    updates.append({'range': f'E{row_number}', 'values': [[result_data["Object"][1]]]})
    updates.append({'range': f'G{row_number}', 'values': [[result_data["Object"][2]]]})
    updates.append({'range': f'I{row_number}', 'values': [[result_data["Predicate"][0]]]})
    updates.append({'range': f'K{row_number}', 'values': [[result_data["Predicate"][1]]]})
    updates.append({'range': f'M{row_number}', 'values': [[result_data["Predicate"][2]]]})
    updates.append({'range': f'O{row_number}', 'values': [[result_data["Triplet"][0]]]})
    updates.append({'range': f'Q{row_number}', 'values': [[result_data["Triplet"][1]]]})
    updates.append({'range': f'S{row_number}', 'values': [[result_data["Parameters"]]]})
    updates.append({'range': f'V{row_number}', 'values': [[result_data["GNN_parameters"]]]})
    updates.append({'range': f'Y{row_number}', 'values': [[result_data["FLOPs"]]]})
    updates.append({'range': f'Z{row_number}', 'values': [[result_data["Latency"]]]})

    # 한 번에 업데이트
    sheet.batch_update(updates)

    print("Data uploaded to Google sheet successfully.")


if __name__ == "__main__":
    result_path = '/path/to/your/result_file.txt'  # 실제 경로로 수정
    flag = 'KD'  # 구글 시트에서 사용할 워크시트 이름

    # save_gspread 함수 호출
    save_gspread(result_path, flag)