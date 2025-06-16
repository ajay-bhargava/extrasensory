import os
import modal


####
# Dataset Path
####

DATASET_PATH = "/mnt/prepared/extrasensory"
BUCKET_NAME = "with-context-datasets"
TRAINED_MODEL_PATH = "/weights"


####
# Credentials
####

CREDENTIALS = modal.Secret.from_name(
    "aws-secret",
    environment_name="main",
    required_keys=[
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
    ]
)

WANDB_SECRET = modal.Secret.from_name(
    "wandb-secret",
    environment_name="main",
    required_keys=[
        "WANDB_API_KEY",
    ]
)

####
# Image
####

training_image = (
    modal.Image.debian_slim()
    .apt_install(["tree", "curl"])
    .pip_install(
        "pandas",
        "numpy",
        "matplotlib",
        "rich",
        "tsai",
        "fastcore==1.7.29",
        "fastai==2.7.19",
        "scikit-learn",
        "wandb",
        "ipykernel"
    )
)

####
# Imports
####

with training_image.imports():
    import numpy as np
    from tsai.all import TSClassifier, TSStandardize, TSClassification, accuracy
    from sklearn.model_selection import train_test_split
    import wandb
    from fastai.callback.wandb import WandbCallback
    
####
# Weights Storage Location
####

weights = modal.Volume.from_name("extrasensory-weights")
    
####
# App
####

app = modal.App(
    name="extrasensory-train",
    image=training_image,
)

@app.function(
    volumes={
        "/mnt/": modal.CloudBucketMount(
            bucket_name=BUCKET_NAME,
            secret=CREDENTIALS
        ),
        "/weights": weights,
    },
    timeout=60 * 60 * 3,
    secrets=[WANDB_SECRET],
    gpu="any"
)
def train_extrasensory():
    # Load dataset and Labels
    dataset = np.load(f"{DATASET_PATH}/dataset.npy")
    labels = np.load(f"{DATASET_PATH}/labels.npy")
    
    if not os.path.exists(f"{TRAINED_MODEL_PATH}"):
        os.makedirs(f"{TRAINED_MODEL_PATH}", exist_ok=True)
    
    # Configruation Dictionary
    config = {
        "tfms": [None, TSClassification],
        "batch_tfms": [TSStandardize(by_sample=True)],
        "metrics": accuracy,
        "arch": "InceptionTimePlus",
        "bs": [64,128],
        "lr": 1e-4,
        "epochs": 5,
    }

    with wandb.init(project="extrasensory-train", config=config):
        # Split dataset into training and validation sets
        train_index, val_index = train_test_split(np.arange(len(dataset)), test_size=0.2, random_state=42, stratify=labels)

        # Splits
        splits = (list(train_index), list(val_index))
        
        # Create Model
        learn = TSClassifier(
            X=dataset, 
            y=labels, 
            splits=splits, 
            tfms=config["tfms"],
            batch_tfms=config["batch_tfms"],
            metrics=config["metrics"],
            arch=config["arch"],
            bs=config["bs"],
            cbs=[WandbCallback(log_preds=False, log_model=False, dataset_name="extrasensory-train")],
        )

        # Train Model
        learn.fit_one_cycle(config["epochs"], config["lr"])
        
        # Save Weights
        learn.export(f"{TRAINED_MODEL_PATH}/{wandb.run.name}.pkl") # type: ignore
        
        # Save Weights hook for Modal Volume
        weights.commit()

    # Finish Weights and Biases
    wandb.finish()
            