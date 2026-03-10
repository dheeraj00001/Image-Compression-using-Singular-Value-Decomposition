import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
data = []
image_folder = "images"
saved_folder = "svd_results"
os.makedirs(saved_folder, exist_ok = True)
print(f"Saved folder created :{saved_folder}")
if not os.path.isdir(image_folder):     #if image folder doesn't exists
    print("Images folder doesn't exist")
else:
    files = os.listdir(image_folder) #os.listdir is a list datatype
    if len(files)==0: 
        print("Files doesn't exist in the folder")
    else:
        for file in files:
            if file.lower().endswith((".jpg", ".png", "jpeg")): #endswith() accepts either single string or tuple of strings
                path = os.path.join(image_folder, file)
                image = cv2.imread(path)
                if image is None:
                    print(f"This image {file} can't be read")
                    continue
                dimension = image.shape
                pixels = dimension[0]*dimension[1]            #total number of pixels in original image
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                U, S, Vt = np.linalg.svd(gray)      
                fig, axes = plt.subplots(3, 4, figsize=(12,8))
                axes = axes.flatten()
                #original image
                axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                axes[0].set_title("Original")
                axes[0].axis("off")
                for idx,k in enumerate(range(10, 101, 10)):
                    U_k = U[:,:k]
                    S_k = np.diag(S[:k])
                    Vt_k = Vt[:k, :]
                    compressed = U_k @ S_k @ Vt_k
                    compressed = np.clip(compressed,0,255).astype(np.uint8)
                    axes[idx+1].imshow(compressed, cmap = "gray")
                    axes[idx+1].set_title(f"k = {k}")
                    axes[idx+1].axis("off")
                for ax in axes[11:]:
                    ax.axis("off")
                save_path = os.path.join(saved_folder, f"{file}_svd_result.png")
                plt.tight_layout()
                plt.savefig(save_path, dpi=300)
                plt.close()
                data.append({"file_name" : file, "pixels_num": pixels, "channel_BGR": dimension[2]})
            else:
                print(f"This file {file} isn't in image format") 
        df = pd.DataFrame(data)
        df.to_excel("image_data.xlsx", index = False)