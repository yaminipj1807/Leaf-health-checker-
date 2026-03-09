import cv2
import numpy as np
import matplotlib.pyplot as plt

def analyze_leaf(image_path):
    # 1. Image Load pannu
    img = cv2.imread(image_path)
    if img is None:
        return "Machi, file name thappa iruku! Check pannu."

    # Color convert panrom
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 2. Masks create panrom
    # Green (Healthy)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(img_hsv, lower_green, upper_green)

    # Brown/Yellow (Disease)
    lower_brown = np.array([10, 40, 20])
    upper_brown = np.array([30, 255, 200])
    mask_disease = cv2.inRange(img_hsv, lower_brown, upper_brown)

    # 3. Clean up (Noise removal)
    kernel = np.ones((5,5), np.uint8)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
    mask_disease = cv2.morphologyEx(mask_disease, cv2.MORPH_OPEN, kernel)

    # 4. Math Calculations
    leaf_area = cv2.countNonZero(mask_green) + cv2.countNonZero(mask_disease)
    disease_area = cv2.countNonZero(mask_disease)
    
    perc = (disease_area / leaf_area) * 100 if leaf_area > 0 else 0

    # 5. Status Decision
    if perc < 1:
        status, text_col = "HEALTHY", "green"
    elif perc < 5:
        status, text_col = "MODERATE DAMAGE", "orange"
    else:
        status, text_col = "DISEASED / INFECTED", "red"

    # 6. Display Result
    plt.figure(figsize=(15, 6))
    
    # Original Image
    plt.subplot(1, 3, 1)
    plt.imshow(img_rgb)
    plt.title("Original Leaf")
    plt.axis("off")

    # Disease Isolation (Red-ah highlight panrom)
    plt.subplot(1, 3, 2)
    res = img_rgb.copy()
    res[mask_disease > 0] = [255, 0, 0]
    plt.imshow(res)
    plt.title("Detected Disease (Red)")
    plt.axis("off")

    # Final Report
    plt.subplot(1, 3, 3)
    plt.text(0.5, 0.6, f"STATUS: {status}", fontsize=15, ha='center', color=text_col, fontweight='bold')
    plt.text(0.5, 0.4, f"Damage: {perc:.2f}%", fontsize=12, ha='center')
    plt.title("Health Report")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
    
    print(f"Analysis Done! Status: {status} ({perc:.2f}%)")

# --- Code Start ---
# Un folder-la irukura correct image name-ah inga kodu
analyze_leaf("temp_image.jpg")