# scripts/label_map.py

# This dictionary maps class indices (0–24) to ASL letters A–Z, skipping 'J'
def get_label_map():
    # Skips 'J' which is not in the dataset
    return {i: chr(65 + i if i < 9 else 66 + i) for i in range(25)}

