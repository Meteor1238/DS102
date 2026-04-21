import os
import time
from data_prep import load_prep_data
from svm import SVM
from evaluate import evaluate_classification, plot_loss
from sklearn.svm import LinearSVC

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.join(base_dir, '..', 'data', 'train')
    test_dir = os.path.join(base_dir, '..', 'data', 'test')

    print("Đang nạp tập Train...")
    X_train, y_train = load_prep_data(train_dir, target_size=(128, 128))
    
    print("\nĐang nạp tập Test...")
    X_test, y_test = load_prep_data(test_dir, target_size=(128, 128))
    
    print(f"\nKích thước tập Train: X={X_train.shape}, y={y_train.shape}")
    print(f"Kích thước tập Test : X={X_test.shape}, y={y_test.shape}")

    sgd_svm = SVM(learning_rate=0.001, C=0.5, epochs=50)
    
    start_time = time.time()
    sgd_svm.fit(X_train, y_train)
    sgd_time = time.time() - start_time
    print(f"Thời gian train SGD SVM: {sgd_time:.2f} giây")

    plot_loss(sgd_svm.loss_history, title="Loss History - SVM (SGD)")

    y_pred_custom = sgd_svm.predict(X_test)
    evaluate_classification(y_test, y_pred_custom)

    sk_svm = LinearSVC(C=0.5, max_iter=10000, dual=True, loss="hinge")
    
    print("Training Sklearn SVM...")
    start_time = time.time()
    sk_svm.fit(X_train, y_train)
    sk_time = time.time() - start_time
    print(f"Thời gian train (Sklearn) : {sk_time:.2f} giây")

    y_pred_sk = sk_svm.predict(X_test)
    evaluate_classification(y_test, y_pred_sk)

if __name__ == "__main__":
    main()