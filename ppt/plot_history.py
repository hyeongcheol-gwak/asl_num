import matplotlib.pyplot as plt
import pickle

try:
    with open('../train/training_history.pkl', 'rb') as f:
        history = pickle.load(f)
    
    if 'train_acc' in history:
        acc_key = 'train_acc'
        val_acc_key = 'val_acc'
    elif 'accuracy' in history:
        acc_key = 'accuracy'
        val_acc_key = 'val_accuracy'
    else:
        print("오류: 학습 히스토리 형식을 인식할 수 없습니다.")
        exit()
    
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history[acc_key], label='Train Acc', marker='.')
    plt.plot(history[val_acc_key], label='Val Acc', marker='.')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss', marker='.')
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
except KeyError as e:
    print(f"오류: 히스토리 키를 찾을 수 없습니다 - {e}")
