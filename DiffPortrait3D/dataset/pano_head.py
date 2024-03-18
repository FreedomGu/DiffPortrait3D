
import cv2
from torch.utils.data import Dataset
import os
import torch
class pano_head_val_pose_sequence_batch_mm(Dataset):
    def __init__(self, image_folder, image_transform = None, sequence_path = None, fea_condition_root = None):
        #assert appearance_idx <= 90
        self.root_folder_path = image_folder
        self.infer_id_list = []
        self.condition_folder_list = []
        for path_1 in sorted(os.listdir(self.root_folder_path)):
            if path_1.endswith(".jpg") or path_1.endswith(".png"):
                self.infer_id_list.append(os.path.join(self.root_folder_path, path_1))
        self.transform = image_transform 
        if sequence_path is not None:
            self.sequence_path = sequence_path
        else:
            self.sequence_path = self.fea_condition_root
        self.fea_condition_root = fea_condition_root
        self.use_cameraasfea = False
    def __len__(self):
        return len(self.infer_id_list)

    def __getitem__(self, idx):
        # fix test reference
        # loading prompt 
        
        prompt = ''
        # loading appearance
        
        id_path  = self.infer_id_list[idx]
        #appearance_path = id_path#os.path.join(id_paths)
        appearance = cv2.resize(cv2.imread(id_path), (512,512))
        appearance = cv2.cvtColor(appearance, cv2.COLOR_BGR2RGB)
        print("appearanc_path", id_path, "size", appearance.shape, "!!!!!!!!!!!!!!")
        if self.transform is not None:
            appearance = self.transform(appearance)
        
        # loading camera sequence
    
        conditions  = []
        fea_conditions = []
        idx = id_path.split("/")[-1]
        idx_name = idx.split(".")[0]
        for num, condition_path in enumerate(sorted(os.listdir(os.path.join(self.sequence_path, 'camera')))):
            condition = cv2.imread(os.path.join(self.sequence_path, 'camera', condition_path))
            condition = cv2.cvtColor(condition, cv2.COLOR_BGR2RGB)
            #print("apperance_path:", os.path.join(self.sequence_path, 'camera', condition_path))
            if self.transform is not None:
                #target = self.transform(target)  
                condition = self.transform(condition)
            if self.fea_condition_root is not None:

                fea_condition_path = os.path.join(self.fea_condition_root, idx_name , "noise_fixcam" , condition_path) # change this from fix to nofix
                print("fea_condition_path!!!", fea_condition_path)
                fea_condition = cv2.imread(fea_condition_path)
                fea_condition = cv2.cvtColor(fea_condition, cv2.COLOR_BGR2RGB)
                if self.transform is not None:  
                    fea_condition = self.transform(fea_condition)
                fea_conditions.append(fea_condition)
            elif self.use_cameraasfea:
                fea_conditions.append(condition)
            else:
                fea_conditions.append(appearance)

            #print("apperance.maxafter", apperance.max(), apperance.min(), target.max(), target.min())
            conditions.append(condition)
        conditions = torch.stack(conditions)
        fea_conditions = torch.stack(fea_conditions)
        res = {'infer_img_name':str(idx), 'condition_image': appearance, 'image': appearance, 'text_blip': prompt, 'text_bg': prompt, 'condition': conditions, 'fea_condition': fea_conditions}

        
        return res #dict(appe    
