import matplotlib.pyplot as plt
import numpy as np
import os

#plot training history showing loss and accuracy curves
def plot_training_history(history: dict, save_path: str = None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    #loss curves
    ax1.plot(history['loss'], label='train loss')
    ax1.plot(history['val_loss'], label='val loss')
    ax1.set_title('model loss')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax1.legend()
    ax1.grid(True)
    
    #accuracy curves
    ax2.plot(history['accuracy'], label='train accuracy')
    ax2.plot(history['val_accuracy'], label='val accuracy')
    ax2.set_title('model accuracy')
    ax2.set_xlabel('epoch')
    ax2.set_ylabel('accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"training curves saved to {save_path}")
    
    plt.close()

#plot confusion matrix heatmap
def plot_confusion_matrix(cm: np.ndarray, save_path: str = None):
    fig, ax = plt.subplots(figsize=(6, 6))
    
    im = ax.imshow(cm, cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['benign', 'malignant'])
    ax.set_yticklabels(['benign', 'malignant'])
    
    #annotate cells with values
    for i in range(2):
        for j in range(2):
            text = ax.text(j, i, cm[i, j],
                          ha="center", va="center", color="black")
    
    ax.set_title('confusion matrix')
    ax.set_ylabel('true label')
    ax.set_xlabel('predicted label')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"confusion matrix saved to {save_path}")
    
    plt.close()

#compare model accuracies side by side
def plot_model_comparison(cnn_accuracy: float, rf_accuracy: float, 
                         save_path: str = None):
    fig, ax = plt.subplots(figsize=(8, 5))
    
    models = ['resnet50', 'random forest']
    accuracies = [cnn_accuracy * 100, rf_accuracy * 100]
    
    bars = ax.bar(models, accuracies)
    bars[0].set_color('steelblue')
    bars[1].set_color('forestgreen')
    
    ax.set_ylabel('accuracy (%)')
    ax.set_title('model comparison')
    ax.set_ylim([0, 100])
    
    #add value labels on bars
    for i, v in enumerate(accuracies):
        ax.text(i, v + 2, f'{v:.2f}%', ha='center', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"model comparison saved to {save_path}")
    
    plt.close()
