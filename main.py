#!/usr/bin/env python3

import warnings
warnings.filterwarnings("ignore")

from config import load_config, validate_config, print_config
from data_processing import load_metadata, prepare_data
from models import build_model, compile_model, extract_embeddings, save_model
from training import train_model, evaluate_model
from random_forest import train_random_forest, evaluate_random_forest, save_random_forest, get_classification_metrics
from visualization import plot_training_history, plot_confusion_matrix, plot_model_comparison

def main():
    #load config
    config = load_config()
    print_config(config)
    
    if not validate_config(config):
        print("\nerror: missing required data files")
        print("place csv files in ./data/ directory")
        return
    
    #load and encode metadata
    print("\nloading metadata...")
    train_df, test_df = load_metadata(config['train_csv'], config['test_csv'])
    print(f"train samples: {len(train_df)}, test samples: {len(test_df)}")
    
    #load and preprocess dicom images
    print("\nloading dicom images...")
    train_images, train_labels, test_images, test_labels = prepare_data(
        train_df, test_df, (config['target_size'], config['target_size'])
    )
    
    #build and compile resnet50 model
    print("\nbuilding model...")
    model = build_model(config)
    model = compile_model(model, config['learning_rate'])
    model.summary()
    
    #train model with validation
    print("\ntraining resnet50 model...")
    checkpoint_path = f"{config['checkpoint_dir']}/best_model.keras"
    history = train_model(
        model, train_images, train_labels, 
        test_images, test_labels, 
        config, checkpoint_path
    )
    
    #evaluate cnn
    print("\nevaluating resnet50...")
    cnn_metrics = evaluate_model(model, test_images, test_labels)
    print(f"resnet50 test accuracy: {cnn_metrics['accuracy'] * 100:.2f}%")
    
    #save trained model
    model_path = f"{config['model_save_dir']}/resnet50_mammography.keras"
    save_model(model, model_path)
    
    #extract embeddings for random forest
    print("\nextracting embeddings...")
    train_embeddings = extract_embeddings(model, train_images)
    test_embeddings = extract_embeddings(model, test_images)
    print(f"embedding shape: {train_embeddings.shape}")
    
    #train random forest on embeddings
    print("\ntraining random forest...")
    rf = train_random_forest(train_embeddings, train_labels, config)
    
    #evaluate random forest
    print("\nevaluating random forest...")
    rf_metrics = evaluate_random_forest(rf, test_embeddings, test_labels)
    print(f"random forest test accuracy: {rf_metrics['accuracy'] * 100:.2f}%")
    
    #save random forest
    rf_path = f"{config['model_save_dir']}/random_forest.pkl"
    save_random_forest(rf, rf_path)
    
    #get detailed metrics
    metrics = get_classification_metrics(test_labels, rf_metrics['predictions'])
    print("\nclassification report:")
    print(metrics['classification_report'])
    
    #generate visualizations
    print("\ngenerating visualizations...")
    plot_training_history(history, f"{config['viz_dir']}/training_history.png")
    plot_confusion_matrix(metrics['confusion_matrix'], f"{config['viz_dir']}/confusion_matrix.png")
    plot_model_comparison(cnn_metrics['accuracy'], rf_metrics['accuracy'], 
                         f"{config['viz_dir']}/model_comparison.png")
    
    print("\npipeline complete")
    print(f"models saved to {config['model_save_dir']}/")
    print(f"visualizations saved to {config['viz_dir']}/")

if __name__ == "__main__":
    main()
