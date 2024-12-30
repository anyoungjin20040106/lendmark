from fastapi import FastAPI, Form, Query, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import sqlite3
import folium
import pandas as pd
import numpy as np
from PIL import Image
import io
from tensorflow.keras.models import load_model
import geopy.distance
app = FastAPI(
    title="객체 탐지 API",
    description="이 API는 관광지 탐색 시스템입니다.",
    version="1.0.0"
)

# 레이블 파일 로드
with open('labels.txt', 'r', encoding="utf-8") as f:
    labels = [line.strip() for line in f.readlines()]
model=load_model('coin_crane_model.h5')
db = sqlite3.Connection('data.db')
cursor=db.cursor()
lendmarkdf = pd.read_sql("select * from lendmark", db)
lendmarkdf['img'].fillna("https://cdn-icons-png.flaticon.com/512/75/75519.png", inplace=True)

# 모델 로드

item = pd.read_sql("select * from item", db)

app.mount("/img", StaticFiles(directory="image"), name="img")

# 관광지 설명 제공 API
@app.post("/content", summary="관광지 설명 제공", description="관광지의 이름을 입력하면 해당 관광지의 설명을 반환합니다.")
async def content(title: str = Form(..., description="조회할 관광지의 이름")):
    item = pd.read_sql("select * from item", db)
    return item[item['name'] == title]['description'].values[0]

# 지도 생성 API
@app.get("/map", summary="지도 생성", description="지정된 위도와 경도에 관광지 정보가 포함된 지도를 반환합니다.")
async def local(lat: str = Query(..., description="위도"), lon: str = Query(..., description="경도"),km:str= Query(..., description="거리")):
    df = pd.read_sql("select * from lendmark", db)
    km=float(km)
    df['km'] = df.apply(lambda x: geopy.distance.distance((float(lat),float(lon)), (x['lat'],x['lon'])).km, axis=1)
    df=df[df['km']<=km]
    
    m = folium.Map(location=[float(lat), float(lon)], zoom_start=12, tiles='stamenterrain')
    folium.Marker(
        [lat, lon],
        icon=folium.Icon(color="blue")
    ).add_to(m)
    for row in lendmarkdf.values:
        index = lendmarkdf[lendmarkdf['name'] == row[0]].index[0]
        folium.Marker(
            [row[1], row[2]],
            popup=f"""
<dialog id="dialog{index}" style="max-width: 800px; max-height: 900px; overflow: auto; padding: 10px;">
    <h1 style="font-size: 16px; margin: 0;">{row[0]}</h1>
    <img src="{row[4]}" alt="{row[0]}" style="width:100%; height:auto; margin-top: 10px;"><br>
    <h2 style="font-size: 14px; margin-top: 10px;">{row[5]}</h2>
    <form method="dialog">
        <button style="margin-top: 10px; padding: 5px 10px;">Close</button>
    </form>
</dialog>
<h1 style="font-size: 14px; margin: 0;">{row[0]}</h1>
<button onclick="document.getElementById('dialog{index}').showModal()" style="padding: 5px 10px; margin-top: 5px;">Read More</button>
            """,
            icon=folium.Icon(color="red")
        ).add_to(m)
    return HTMLResponse(content=m._repr_html_())

# 관광지 종류 조회 API
@app.get("/kind", summary="관광지 종류 조회", description="고유한 관광지 종류 목록을 반환합니다.")
async def kind():
    return {'data': ['모두보기'] + list(lendmarkdf['kind'].unique())}

# 관광지 상세 정보 조회 API
@app.post("/item", summary="관광지 상세 정보 조회", description="관광지 이름으로 상세 정보를 조회합니다.")
async def yujeock(name: str = Form(..., description="관광지 이름")):
    item = pd.read_sql(f"select * from item where name= ? ", db, params=(name,))
    return dict(zip(item.columns, item.values[0]))

# 객체 탐지 API
@app.post("/detect", summary="객체 탐지", description="이미지에서 객체를 탐지하고 탐지 결과를 반환합니다.")
async def detect(file: UploadFile = File(..., description="분석할 이미지 파일 (JPG 또는 PNG 형식 권장)")):
    content = await file.read()
    img=Image.open(io.BytesIO(content)).convert("RGB").resize((244,244))
    img_array=np.array(img)/255
    img_array=np.expand_dims(img_array,axis=0)
    result=model.predict(img_array)
    print(np.argmax(result))
    return {'data':str((np.argmax(result)))}

# 지역 추가 api
@app.post("/visit", summary="방문한 유적 업로드", description="방문한 유적을 업로드 합니다")
async def visit(name: str = Form(..., description="유적이름")):
    result=""
    try:
        cursor.execute("insert  INTO visit (name) VALUES (?)",(name,))
        db.commit()
        print("성공")
        return "성공"
    except Exception as e:
        print(e)
        return "실패"

# 지역 삭제 api
@app.post("/reset", summary="방문한 유적 정보를 삭제함", description="방문한 유적을 삭제 합니다")
async def reset():
    try:
        cursor.execute("delete  from visit ")
        db.commit()
    except:
        pass
# 지역 조회 api
@app.get("/visit", summary="유적 조회", description="방문한 유적을 조회 합니다")
async def visit():
    local=list(pd.read_sql("select * from visit",db)['name'])
    df=pd.read_sql(f"SELECT * FROM local WHERE name IN ({', '.join(['?']*len(local))})",db,params=local)
    return {'data': df.to_dict(orient='records')}