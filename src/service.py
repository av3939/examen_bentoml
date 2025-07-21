import bentoml
from bentoml.io import JSON
from pydantic import BaseModel

# Chargement du modèle BentoML
model_ref = bentoml.sklearn.get("admission_model:latest")
model_runner = model_ref.to_runner()

# Déclaration du service
svc = bentoml.Service("admission_service", runners=[model_runner])

# Modèle d'entrée
class AdmissionData(BaseModel):
    GRE: float
    TOEFL: float
    UniversityRating: float
    SOP: float
    LOR: float
    CGPA: float
    Research: int

# Endpoint sécurisé (login)
@svc.api(input=JSON(), output=JSON())
def login(input_data: dict) -> dict:
    if input_data.get("password") == "secret123":
        return {"success": True, "message": "Authentication successful"}
    else:
        return {"success": False, "message": "Authentication failed"}

# Endpoint de prédiction
@svc.api(input=JSON(pydantic_model=AdmissionData), output=JSON())
async def predict(data: AdmissionData) -> dict:
    input_df = [[
        data.GRE, data.TOEFL, data.UniversityRating,
        data.SOP, data.LOR, data.CGPA, data.Research
    ]]
    prediction = await model_runner.async_run(input_df)
    admission_chance = round(float(prediction[0]), 2)
    admitted = admission_chance >= 0.5
    return {
        "admission_chance": admission_chance,
        "admitted": admitted
    }
