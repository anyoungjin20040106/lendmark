import folium
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from fastapi import FastAPI,Form,Query
from fastapi.responses import HTMLResponse
import numpy as np
import folium
df=pd.read_csv('data.csv')
app=FastAPI()
@app.post("/getInfo")
async def getInfo(x:str=Form(...),y:str=Form(...)):
     x=float(x)
     y=float(y)
     model=KNeighborsClassifier()
     model.fit(df[['경도','위도']].values,df[['시설명','카테고리1','카테고리2']].values)
     result=model.predict([[x,y]])[0]
     data=dict(zip(['name'],result))
     r=df[df['시설명']==data['name']]
     data['sx']=x
     data['sy']=y
     data['c1']=r['카테고리1'].values[0]
     data['c2']=r['카테고리2'].values[0]
     data['ey']=r['위도'].values[0]
     data['ex']=r['경도'].values[0]
     return data
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