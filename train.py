import argparse, os
from tensorflow.keras.datasets import mnist
from src.model import build_cnn

def main(args):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape((-1,28,28,1)).astype('float32')/255.0
    X_test = X_test.reshape((-1,28,28,1)).astype('float32')/255.0

    model = build_cnn((28,28,1), 10)
    model.fit(X_train, y_train, validation_data=(X_test,y_test),
              epochs=args.epochs, batch_size=args.batch_size)
    os.makedirs(args.output_dir, exist_ok=True)
    model.save(os.path.join(args.output_dir, "best_model.h5"))
    print("Model saved in", args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()
    main(args)
