import os.path

import pandas as pd
import shutil
base_path = '/Volumes/thien/data_tc/data_raw/SongMai_TPBG/label/CÔ LÊ CÚC SM_00001-00197-1'
label_path = '/Volumes/thien/data_tc/data_raw/SongMai_TPBG/label/CÔ LÊ CÚC SM_00001-00197-1/TongHop_LaoCai  SM 01-197.xls'
output_folder = 'data'
df = pd.read_excel(label_path, sheet_name='Tailieudinhkem')

print(df.head())

for idx, row in df.iterrows():
    path = row['File đính kèm (PDF)\n13']
    print(row['Tên tài liệu\n2'].lower())
    if row['Tên tài liệu\n2'].lower() == 'Giấy chứng nhận quyền sử dụng đất'.lower():
        id_ = row['Mã hồ sơ \n1']
        path = path.replace('\\', '/')
        filename = os.path.basename(path)
        src_path = f'{base_path}{path}'
        des_path = os.path.join(output_folder, f'{id_}_{filename}')
        if os.path.exists(des_path):
            print(f'exises file {src_path}')
            continue
        try:
            shutil.copyfile(src_path, des_path)
        except:
            print(f"NOt found: {src_path}")


# /Users/tienthien/Downloads/CÔ LÊ CÚC SM_01137-01246/Dat_Dai/TP_BacGiang/Phuong_/Hop_so_/SM_01215/GCNQSDDD___01.pdf
# /Users/tienthien/Downloads/CÔ LÊ CÚC SM_01137-01246/Dat_Dai/TP_BacGiang/Phuong_/Hop_so_/SM_01215/GCNQSDD___01.pdf