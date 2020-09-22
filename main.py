import argparse
import os
import random

import numpy as np

from datasets import load_dataset
from models import GaussianVAE, StudentsTVAE

if __name__ == '__main__':

    # Parser
    parser = argparse.ArgumentParser(description="Student-t Variational Autoencoder for Robust Density Estimation.")
    parser.add_argument("--dataset", type=str, default="SMTP", help="Dataset Name.")
    parser.add_argument("--decoder", type=str, default="normal", help="Decoder.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning Rate for VAE.")
    parser.add_argument("--seed", type=int, default=42, help="Random Seed.")

    args = parser.parse_args()

    dataset = args.dataset
    decoder = args.decoder
    learning_rate = args.learning_rate
    seed = args.seed

    print(f"Dataset: {dataset} / Decoder: {decoder} / Learning Rate: {learning_rate} / Seed: {seed}")

    # Set seed
    random.seed(seed)
    np.random.seed(seed)

    # Get dataset
    X_train, X_valid, X_test = load_dataset(key=dataset)

    # Train
    save_dir = f"save/{dataset}/{decoder}/"
    save_path = f"save/{dataset}/{decoder}/model_{learning_rate}_{seed}"
    os.makedirs(save_dir, exist_ok=True)

    # for binary data
    if decoder == "normal":
        model = GaussianVAE(n_in=X_train.shape[1], n_latent=2, n_h=500)
    else:
        model = StudentsTVAE(n_in=X_train.shape[1], n_latent=2, n_h=500)

    print(f"Model: {type(model)}")

    model.fit(X_train, k=1, batch_size=100,
              learning_rate=learning_rate, n_epoch=500,
              warm_up=False, is_stoppable=True,
              X_valid=X_valid, path=save_path)

    # Test
    test_score = model.importance_sampling(X_test, k=10)
    print("Test Score: ", np.mean(test_score))

    # Save numpy files
    os.makedirs("npy/", exist_ok=True)
    np.save(f"npy/exp_{dataset}_{decoder}_train_loss_{learning_rate}_{seed}.npy", np.array(model.train_losses))
    np.save(f"npy/exp_{dataset}_{decoder}_train_time_{learning_rate}_{seed}.npy", np.array(model.train_times))
    np.save(f"npy/exp_{dataset}_{decoder}_valid_loss_{learning_rate}_{seed}.npy", np.array(model.valid_losses))
    np.save(f"npy/exp_{dataset}_{decoder}_RE_{learning_rate}_{seed}.npy", np.array(model.reconstruction_errors))
    np.save(f"npy/exp_{dataset}_{decoder}_KL_{learning_rate}_{seed}.npy", np.array(model.kl_divergences))
    np.save(f"npy/exp_{dataset}_{decoder}_test_score_{learning_rate}_{seed}.npy", test_score)

    if decoder == "normal":
        mu, ln_var = model.reconstruct(X_train)
        np.save(f"npy/exp_{dataset}_{decoder}_mu_{learning_rate}_{seed}.npy", mu)
        np.save(f"npy/exp_{dataset}_{decoder}_ln_var_{learning_rate}_{seed}.npy", ln_var)
    else:
        ln_df, loc, ln_scale = model.reconstruct(X_train)
        np.save(f"npy/exp_{dataset}_{decoder}_ln_df_{learning_rate}_{seed}.npy", ln_df)
        np.save(f"npy/exp_{dataset}_{decoder}_loc_{learning_rate}_{seed}.npy", loc)
        np.save(f"npy/exp_{dataset}_{decoder}_ln_scale_{learning_rate}_{seed}.npy", ln_scale)
