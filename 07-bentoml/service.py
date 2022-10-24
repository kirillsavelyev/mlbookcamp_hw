import bentoml
from bentoml.io import JSON, NumpyNdarray
from pydantic import BaseModel

"""
Usage:
    bentoml serve service.py:service
    for production
    bentoml serve service.py:service --production
        Service will be runned on http://localhost:3000
"""

class UserProfile(BaseModel):
    name: str
    age: int
    country: str
    rating: float


# model_ref = bentoml.sklearn.get("mlzoomcamp_homework:latest")
model_ref = bentoml.sklearn.get("mlzoomcamp_homework:qtzdz3slg6mwwdu5")     # the first model
# model_ref = bentoml.sklearn.get("mlzoomcamp_homework:jsi67fslz6txydu5")   # the second model

model_runner = model_ref.to_runner()
service = bentoml.Service("mlzoomcamp_homework", runners=[model_runner])

@service.api(input=NumpyNdarray(), output=NumpyNdarray())
def classify(input_data):
    return model_runner.run(input_data)
