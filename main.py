import pandas as pd
from fastapi import FastAPI, Form
from fastapi.templating import Jinja2Templates

lendmarkdf = pd.read_csv('lendmark.csv')
app = FastAPI()
lendmarkdf.fillna(
    'https://maps.gstatic.com/tactile/pane/default_geocode-2x.png',
    inplace=True)
templates = Jinja2Templates(directory="template")


@app.post("/content")
async def content(title: str = Form(...)):
    item = pd.read_csv('item.csv')
    return item[item['이름'] == title]['설명'].values[0]


@app.post("/lendmark")
async def lendmark(kind: str = Form(...)):
    tf = lendmarkdf[lendmarkdf['종류'] ==
                    kind] if kind != "모두보기" else lendmarkdf.copy()
    return {
        'data': [{
            "name": row[0],
            "lat": row[1],
            "lon": row[2],
            "img": row[4],
            "content": row[5],
        } for row in tf.values]
    }


@app.get("/lendmark")
async def lendmark():
    return {
        'data': [{
            "name": row[0],
            "lat": row[1],
            "lon": row[2],
            "img": row[4],
            "content": row[5],
        } for row in lendmarkdf.values]
    }


@app.get("/kind")
async def kind():
    return {'data': ['모두보기'] + list(lendmarkdf['종류'].unique())}
