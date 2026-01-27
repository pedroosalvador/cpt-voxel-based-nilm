import numpy as np
from volumentations import *

augmentation_config = {
    "target_samples": {
      "CRITICAL": 220,
      "SMALL": 220,
      "MEDIUM": 240
    },

    "target_multiplier": {
      "LARGE": 1.2
    },

    "class_assignments": {
      "AIR CONDITIONER": "LARGE",
      "COFFEE MAKER": "CRITICAL",
      "COMPACT FLUORESCENT LAMP": "LARGE",
      "FAN": "LARGE",
      "FRIDGE": "SMALL",
      "HAIR IRON": "CRITICAL",
      "HAIRDRYER": "LARGE",
      "HEATER": "SMALL",
      "INCANDESCENT LIGHT BULB": "LARGE",
      "LAPTOP": "LARGE",
      "MICROWAVE": "LARGE",
      "SOLDERING IRON": "MEDIUM",
      "VACUUM": "SMALL",
      "WASHING MACHINE": "CRITICAL",
      "WATER KETTLE": "MEDIUM"
    }
}

def apply_augmentation(voxel, pipeline):
    augmented = pipeline(image=voxel)['image']
    return np.clip(augmented, 0, 1)

def augment_dataset(X, y):
    """
    Level conditions:
    - CRITICAL: samples < 85 → target_samples = 220
    - SMALL: 85 <= samples < 130 → target_samples = 220
    - MEDIUM: 130 <= samples < 200 → target_samples = 240
    - LARGE: samples >= 200 → target_multiplier = 1.2
    """
    print('Applying data augmentation...')

    aug = Compose([GaussianNoise(var_limit=(0.0, 0.001), p=0.7)])
    augmented_voxels, augmented_labels = [], []

    for class_name in np.unique(y):
        mask = y == class_name  
        voxels, labels = X[mask], y[mask]
        
        samples_len = len(voxels)
        level_assignments = augmentation_config['class_assignments'][class_name]
    
        if level_assignments in augmentation_config['target_samples']:
            target_samples = augmentation_config['target_samples'][level_assignments]
            print(f"target samples: {target_samples}")
        elif level_assignments in augmentation_config['target_multiplier']:
            target_multiplier = augmentation_config['target_multiplier'][level_assignments]
            target_samples = int(samples_len * target_multiplier)
            print(f"target samples: {target_samples}")
        else:
            print('class not found in dictionary')

        print(f"{class_name:30s}: {samples_len:3d} → {target_samples:3d} [{level_assignments}]")

        # Add original samples first
        augmented_voxels.extend(voxels)
        augmented_labels.extend(labels)

        # Augment to reach target
        needed = target_samples - samples_len
        for _ in range(needed):
            idx = np.random.randint(0, samples_len)
            augmented = apply_augmentation(voxels[idx], aug)
            augmented_voxels.append(augmented)
            augmented_labels.append(class_name)
    
    print(f"\nDataset: {len(X)} → {len(augmented_voxels)} samples")
    
    # Convert to numpy with float32 (critical for memory!)
    X_aug = np.array(augmented_voxels, dtype=np.float32)
    y_aug = np.array(augmented_labels)
    
    return X_aug, y_aug