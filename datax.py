# import os
# import cv2
# from typing import *
# try:
#     from data import *
# except ModuleNotFoundError as err:
#     from dl.data import *
    
# class ImageFolder(Dataset):
#     def __init__(self,path:str) -> None:
#         super().__init__()
#         # path = os.path.abspath(path).replace('\\','/')
#         if not path.endswith('/'):
#             path += '/'
#         self.samples = []
#         self.labels = []
#         self.idx_to_label_str=[]
#         for idx,sub_folder_name in enumerate(os.listdir(path)):
#             self.idx_to_label.append(sub_folder_name)
#             sub_folder_path = path+sub_folder_name+'/'
#             for image_name in os.listdir(sub_folder_path):
#                 img_array = cv2.imread(sub_folder_path+image_name)
                
        
#     def __len__(self) -> int:
#         pass
    
#     def __getitem__(self, index: int) -> tuple:
#         pass
    
    
# path = "./test_img_folder/"
# for s in os.listdir(path):
#     print(s)