import modal
from pydantic import BaseModel

#### 
# Weights Path
####

weights = modal.Volume.from_name("extrasensory-weights")
MODEL_ID = "wild-sky-12.pkl"

####
# Image
####

inference_image = (
    modal.Image.debian_slim()
    .apt_install(["tree", "curl"])
    .pip_install(
        "numpy",
        "tsai",
        "fastcore==1.7.29",
        "fastai==2.7.19",
        "wandb",
        "torch"
    )
)

asgi_image = (
    modal.Image.debian_slim()
    .apt_install(["tree", "curl"])
    .pip_install(
        "httpx",
        "pydantic",
        "fastapi",
        "starlette",
        "numpy",
        "torch"
    )
)

####
# Application
####

app = modal.App(name="extrasensory-inference")

####
# Volumes
####

weights_volume = modal.Volume.from_name("extrasensory-weights")

####
# Imports
####

with inference_image.imports():
    from tsai.inference import load_learner
    import numpy as np
    import wandb
    
with asgi_image.imports():
    from fastapi import FastAPI, Body
    from pydantic import BaseModel
    import base64
    import io
    import numpy as np
    
####
# GPU Application
####

@app.cls(
    image=inference_image, 
    gpu="any",
    scaledown_window=60 * 10,
    volumes={"/weights": weights_volume},
)
class Inference:
    @modal.enter()
    def start(self):
        wandb.init(mode="disabled")
        self.model = load_learner(f"/weights/{MODEL_ID}", cpu=False)
        
    @modal.method()
    async def predict(self, data: np.ndarray):
        return self.model.get_X_preds(data, with_input=False)
    
####
# Request Model
####

class Request(BaseModel):
    array: str
   
class Result(BaseModel):
    probabilities: list
    predicted_class: str
    message: str
    
####
# ASGI Application
####

@app.function(
    image=asgi_image,
    max_containers=2
)
@modal.asgi_app(label="extrasensory-inference")
def extrasensory_inference():
    application = FastAPI(title="Extrasensory Inference", description="Inference for the Extrasensory Model")
    
    # Inference endpoint
    @application.post("/predict", response_model=Result)
    async def predict(request: Request = Body()):
        
        bytes = base64.b64decode(request.array)
        array = np.load(io.BytesIO(bytes))
        
        if array.shape != (1, 5, 225):
            return {"error": "Invalid array shape"}

        # Make prediction - don't await the remote call
        response = Inference().predict.remote(array) # type: ignore
        
        # Unpack the tuple and convert tensor to numpy
        probs_tensor, _, class_label = response
        probabilities = probs_tensor.cpu().numpy().tolist()
        predicted_class = class_label[0]
        
        return Result(
            probabilities=probabilities,
            predicted_class=predicted_class,
            message="Success"
        )
    
    return application