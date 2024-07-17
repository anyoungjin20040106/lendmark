import os
import folium.map
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from fastapi import FastAPI,Form,HTTPException,Query
from fastapi.responses import HTMLResponse
import httpx
import numpy as np
import folium
url = 'http://api.data.go.kr/openapi/tn_pubr_public_trrsrt_api'
params ={'serviceKey' : os.getenv("publicdata"), 'pageNo' : '1', 'numOfRows' : '10000', 'type' : 'json'}

app=FastAPI()
@app.post("/getInfo")
async def getInfo(x:str=Form(),y:str=Form()):
    x=float(x)
    y=float(y)
    try:
        async with httpx.AsyncClient() as hc:
            response=await hc.get(url,params=params)
        data=response.json()['response']['body']['items']
        df=pd.DataFrame(data)[['trrsrtNm','latitude','longitude','trrsrtIntrcn','phoneNumber']]
        df[['latitude','longitude']]=df[['latitude','longitude']].astype("float64")
        model=KNeighborsClassifier()
        model.fit(df[['latitude','longitude']],df[['trrsrtNm','trrsrtIntrcn','phoneNumber']])
        result=model.predict([[y,x]])
        json=dict(zip(['name','des','ph'],result[0]))
        r=df[df['trrsrtNm']==json['name']]
        json['sx']=x
        json['sy']=y
        json['ex']=r['longitude'].values[0]
        json['ey']=r['latitude'].values[0]
        return json
    except Exception as e:
        raise HTTPException(status_code=429, detail=f"에러 : {e}")
@app.get("/map")
async def getInfo(name:str=Query(""),ex:str=Query(""),ey:str=Query(""),sx:str=Query(""),sy:str=Query("")):
    if name=="":
        return HTMLResponse("<script>alert('비정상적인 접근');window.history.back();</script>")
    ex=float(ex)
    ey=float(ey)
    sx=float(sx)
    sy=float(sy)
    m=folium.Map(location=[get_mid_point(sy,ey),get_mid_point(sx,ex)])
    folium.Marker([sy,sx],tooltip="현재위치",icon=folium.Icon(color="red")).add_to(m)
    folium.Marker([ey,ex],tooltip=name).add_to(m)
    return HTMLResponse(m._repr_html_())
@app.get("/")
async def index():
        return HTMLResponse("<script>alert('비정상적인 접근');window.history.back();</script>")
def get_mid_point(s,e):
    mid=(np.abs(s-e))/2
    return mid+(s if s<e else e)