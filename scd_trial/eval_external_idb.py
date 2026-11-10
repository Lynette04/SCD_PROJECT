import os
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from pytorch_training import load_model
from step2_data_pipeline import efficientnet_preprocess

def _resolve_dir(sub):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(r"..", "Version 2, erythrocytesIDB 2017", "Version 2, erythrocytesIDB", sub),
        os.path.join(r"..\..", "Version 2, erythrocytesIDB 2017", "Version 2, erythrocytesIDB", sub),
        os.path.join(script_dir, "..", "..", "Version 2, erythrocytesIDB 2017", "Version 2, erythrocytesIDB", sub),
        os.path.join(r"C:\Users\kaboh\OneDrive\Desktop\Project-prototype", "Version 2, erythrocytesIDB 2017", "Version 2, erythrocytesIDB", sub),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return candidates[1]

EXTERNAL_DIRS=[
    _resolve_dir("erythrocytesIDB2"), _resolve_dir("erythrocytesIDB3"),
]

def evaluate_external_smears(model_path="sickle_cell_effnet_lite0_ft8.pt"):
    print(f"Loading {model_path}...")
    model = load_model(model_path)
    model.eval()
    
    total_images = 0
    correct = 0
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    
    print("\n Evaluating on external full blood smear images...")
    
    with torch.no_grad():
        for base_dir in EXTERNAL_DIRS:
            for folder_name in sorted(os.listdir(base_dir)):
                smear_folder = os.path.join(base_dir, folder_name)
                source_path = os.path.join(smear_folder, "source.jpg")
                mask_path = os.path.join(smear_folder, "mask-elongated.jpg")
                
                if not os.path.isfile(source_path) or not os.path.isfile(mask_path):
                    continue
                
                mask_np = np.array(Image.open(mask_path))
                true_label = 1 if(mask_np > 128). sum() > 0 else 0
                
                img = Image.open(source_path).convert("RGB")
                pixels = np.array(img, dtype=np.float32)
                tensor = torch.from_numpy(pixels).permute(2,0,1).unsqueeze(0)
                tensor = F.interpolate(tensor, size = (224, 224), mode="bilinear", align_corners=False)
                tensor = efficientnet_preprocess(tensor)
                
                confidence = model(tensor).item()
                prediction = 1 if confidence >= 0.5 else 0
                
                total_images +=1
                if prediction == true_label:
                    correct +=1
                    
                if prediction == 1 and true_label ==1:
                    true_positives+=1
                elif prediction==1 and true_label==0:
                    false_positives+=1
                elif prediction==0 and true_label==0:
                    true_negatives+=1
                elif prediction==0 and true_label==1:
                    false_negatives+=1
                
                status = "✓"  if prediction == true_label else "✗"
                label_str = "Sickle" if true_label == 1 else "Normal"
                print(f"[{status}] {folder_name:<20} True: {label_str:<7} Pred: {confidence:.3f}")
                
    accuracy = correct / total_images if total_images else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) else 0
    
    print("\n" + "=" *45)
    print("EXTERNAL TEST RESULTS (erythrocytesIDB)")
    print("=" * 45)
    print(f"Total External Smears: {total_images}")
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total_images})")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"TP: {true_positives} | FP: {false_positives} | TN: {true_negatives} | FN: {false_negatives}")
    
if __name__ == "__main__":
    evaluate_external_smears("sickle_cell_effnet_lite0_ft8.pt")       
                    