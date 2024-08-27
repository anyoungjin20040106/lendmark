import pandas as pd
from fastapi import FastAPI, Form,Query
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import folium
from geopy import distance

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

@app.get("/map")
async def map(lat:str=Query(...),lon:str=Query(...),kind:str=Query("모두보기"),km:str=Query(4)):
    m = folium.Map(location=[lat,lon], zoom_start=12)
    tf = lendmarkdf[lendmarkdf['종류'] ==
                    kind].copy() if kind != "모두보기" else lendmarkdf.copy()
    tf['거리'] = tf.apply(lambda x: distance.distance((x['위도'], x['경도']), (lat, lon)).km, axis=1)
    tf=tf[tf['거리']<=float(km)]
    folium.Marker([lat,lon]).add_to(m)
    for row in tf.values:
        folium.Marker(location=[row[1],row[2]],icon=folium.Icon(color='red'),popup=f"""
                      <h1>{row[0]}</h1><dialog id={row[0]}><form method="dialog"><h1>{row[0]}</h1><img src="{row[4]}" width="100%"><br><h2>{row[5]}</h2><h2>거리 : {row[6]}</h2><button type="button"><h2>ar로 보기</h2></button><button><h2>닫기</h2></button></form></dialog><button onclick="document.getElementById('{row[0]}').showModal();">설명보기</button>""").add_to(m)
    map_html = m._repr_html_()
    return HTMLResponse(content=map_html)


@app.get("/kind")
async def kind():
    return {'data': ['모두보기'] + list(lendmarkdf['종류'].unique())}
