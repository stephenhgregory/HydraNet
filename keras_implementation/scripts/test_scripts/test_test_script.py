import test
import time

if __name__ == "__main__":

    train_data = "data/subj1/train"

    print("Beginning getting training patches")
    start_time = time.time()

    # Get our training data to use for determining which denoising network to send each patch through
    training_patches = test.retrieve_train_data(train_data)

    print(f"Done getting training patches! Total time = {time.time() - start_time}")

