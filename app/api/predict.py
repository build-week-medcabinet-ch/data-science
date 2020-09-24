import logging
import random

import pandas as pd
from fastapi import APIRouter
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)
router = APIRouter()

"""read in the data"""
data = pd.read_csv('https://raw.githubusercontent.com/'
                   'build-week-medcabinet-ch/data-science/master/data/final%20(1).csv')


class Item(BaseModel):
    """Use this data model to parse the request body JSON."""

    Effects: str = Field(..., example='Creative,Energetic,Tingly,Euphoric,Relaxed')
    Type: str = Field(..., example='hybrid,sativa,indica')
    Flavor: str = Field(..., example='Earthy,Sweet,Citrus')

    def to_df(self):
        """Convert pydantic object to pandas dataframe with 1 row."""
        return pd.DataFrame([dict(self)])




@router.post('/predict')
async def predict(item: Item):
    """Make random baseline predictions for classification problem."""
    X_new = item.to_df()

    num1 = random.randint(0, 2276)
    index, strain, typ, rating, effects, flavor, description, nearest = data.iloc[num1]

    return {
        'prediction': strain,
        'Description': description,
        'rating': int(rating),
        'Type': typ,
        'Effects': effects,
        'Flavors': flavor
    }