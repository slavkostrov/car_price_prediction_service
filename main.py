import tempfile
from typing import List, Optional, Union

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI()
_BEST_MODEL = joblib.load("BEST_MODEL_PIPE.pkl")


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: Optional[Union[str, float]]
    engine: Optional[Union[str, float]]
    max_power: Optional[Union[str, float]]
    torque: Optional[Union[str, float]]
    seats: float
    
    def to_df(self):
        return pd.DataFrame([self.model_dump()])


class Items(BaseModel):
    objects: List[Item]

    @classmethod
    def from_csv(cls, csv_path: str) -> "Items":
        items = pd.read_csv(csv_path, dtype=str)
        objects = []
        fields = list(Item.model_fields.keys())
        for _, row in items.iterrows():
            objects.append(Item(**row[fields].to_dict()))
        return Items(objects=objects)

    def to_df(self):
        return pd.DataFrame([item.model_dump() for item in self.objects])


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    """По описанию объекта возвращает предсказание."""
    df = item.to_df()
    return np.nan_to_num(_BEST_MODEL.predict(df)[0])


@app.post("/predict_items")
def predict_items(items: Items) -> List[float]:
    """Принимает JSON со списком описаний объектов, возврщает список предсказаний."""
    df = pd.DataFrame([item.model_dump() for item in items.objects])
    return np.nan_to_num(_BEST_MODEL.predict(df))


@app.post("/predict_items_csv")
def predict_items_csv(upload_file: UploadFile) -> FileResponse:
    """Принимает CSV, отдаёт csv с колонкой predict."""
    items = Items.from_csv(upload_file.file)
    
    df = items.to_df()
    df["predict"] = np.nan_to_num(_BEST_MODEL.predict(df))
    
    headers = {"Content-Disposition": 'attachment; filename="prediction.csv"'}
    with tempfile.NamedTemporaryFile() as temp:
        df.to_csv(f"{temp.name}.csv")
        return FileResponse(f"{temp.name}.csv", headers=headers)
