import matplotlib.pyplot as plt
import pickle

try:
    with open('training_history.pkl', 'rb') as f:
        history = pickle.load(f)
        
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Train Acc', marker='.')
    plt.plot(history['val_accuracy'], label='Val Acc', marker='.')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Train Loss', marker='.')
    plt.plot(history['val_loss'], label='Val Loss', marker='.')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    print("그래프 출력 완료")
    
except FileNotFoundError:
    print("history 파일을 찾을 수 없습니다.")