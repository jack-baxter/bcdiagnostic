import os
from dotenv import load_dotenv

#load config from env file
def load_config() -> dict:
    load_dotenv()
    
    config = {
        'train_csv': os.getenv('TRAIN_CSV', './data/mass_case_description_train_set.csv'),
        'test_csv': os.getenv('TEST_CSV', './data/mass_case_description_test_set.csv'),
        'dicom_root': os.getenv('DICOM_ROOT', './data/dicom_files/'),
        'target_size': int(os.getenv('TARGET_IMAGE_SIZE', 224)),
        'normalize_method': os.getenv('NORMALIZE_METHOD', 'max'),
        'resnet_weights': os.getenv('RESNET_WEIGHTS', 'imagenet'),
        'freeze_base': os.getenv('FREEZE_BASE', 'true').lower() == 'true',
        'dense_units': int(os.getenv('DENSE_UNITS', 128)),
        'dropout_rate': float(os.getenv('DROPOUT_RATE', 0.5)),
        'learning_rate': float(os.getenv('LEARNING_RATE', 0.001)),
        'epochs': int(os.getenv('EPOCHS', 10)),
        'batch_size': int(os.getenv('BATCH_SIZE', 32)),
        'rf_estimators': int(os.getenv('RF_ESTIMATORS', 100)),
        'random_state': int(os.getenv('RANDOM_STATE', 42)),
        'model_save_dir': os.getenv('MODEL_SAVE_DIR', './models'),
        'checkpoint_dir': os.getenv('CHECKPOINT_DIR', './models/checkpoints'),
        'log_dir': os.getenv('LOG_DIR', './logs'),
        'viz_dir': os.getenv('VIZ_DIR', './visualizations')
    }
    
    return config

#validate required paths exist
def validate_config(config: dict) -> bool:
    required_files = ['train_csv', 'test_csv']
    missing = []
    
    for key in required_files:
        path = config.get(key)
        if not os.path.exists(path):
            missing.append(path)
    
    if missing:
        print(f"error: missing required files: {', '.join(missing)}")
        return False
    
    return True

#print config for debugging
def print_config(config: dict):
    print("current configuration:")
    print("-" * 50)
    for key, value in config.items():
        print(f"{key}: {value}")
    print("-" * 50)
