import pandas as pd
from fastapi import FastAPI, Form,Query
from fastapi.responses import HTMLResponse,FileResponse
from fastapi.templating import Jinja2Templates
import folium
from geopy import distance

lendmarkdf = pd.read_csv('lendmark.csv')
navi = pd.read_csv('navi.csv')
app = FastAPI()
lendmarkdf.fillna(
    'https://maps.gstatic.com/tactile/pane/default_geocode-2x.png',
    inplace=True)
templates = Jinja2Templates(directory="template")

@app.get("/")
async def index():
    return FileResponse("index.html")
@app.post("/content")
async def content(title: str = Form(...)):
    item = pd.read_csv('item.csv')
    return item[item['이름'] == title]['설명'].values[0]

@app.post("/map")
async def map(lat:str=Form(...),lon:str=Form(...),km:str=Form(4)):
    m = folium.Map(location=[lat,lon], zoom_start=12)
    tf = lendmarkdf.copy()
    folium.Marker([lat,lon]).add_to(m)
    folium.CircleMarker([lat,lon],radius=km*100).add_to(m)
    for row in tf.values:
        folium.Marker(location=[row[1],row[2]],icon=folium.Icon(color='red'),popup=f"""
                      <h1>{row[0]}</h1><dialog id={row[0]}><form method="dialog"><h1>{row[0]}</h1><img src="{row[4]}" width="100%"><br><h2>{row[5]}</h2><h2>거리 : {row[6]}</h2><button type="button"><h2>ar로 보기</h2></button><button><h2>닫기</h2></button></form></dialog><button onclick="document.getElementById('{row[0]}').showModal();">설명보기</button>""").add_to(m)
    map_html = m._repr_html_()
    return HTMLResponse(content=map_html)


@app.get("/kind")
async def kind():
    return {'data': ['모두보기'] + list(lendmarkdf['종류'].unique())}
@app.post("/place")
async def place(lendmark:str=Form(...)):
    rf=navi[navi['시작점']&(navi['관광지명']==lendmark)]
    return{
    'name':rf['장소명'].values[0],
    'img':rf['이미지주소'].values[0],
    'top':rf['위쪽장소'].values[0],
    'bottom':rf['아랫쪽장소'].values[0],
    'left':rf['왼쪽장소'].values[0],
    'right':rf['오른쪽장소'].values[0],
}
@app.post("/place")
async def place(name:str=Form(...)):
    rf=navi[navi['시작점']&(navi['장소명']==name)]
    return{
    'name':rf['장소명'].values[0],
    'img':rf['이미지주소'].values[0],
    'top':rf['위쪽장소'].values[0],
    'bottom':rf['아랫쪽장소'].values[0],
    'left':rf['왼쪽장소'].values[0],
    'right':rf['오른쪽장소'].values[0],
}
if __name__=="__main__":
    import uvicorn
    uvicorn.run(app,host="0.0.0.0",port=8569,ssl_keyfile="key.pem",ssl_certfile="cert.pem",reload=True)
#uvicorn main:app --host 0.0.0.0 --port 8569 --ssl-keyfile key.pem --ssl-certfile cert.pem --reload
