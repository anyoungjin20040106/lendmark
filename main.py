#fastapi 설치하는 방법
#pip install -r requirements.txt
#fastapi 실행하는법
#uvicorn main:app --reload
#엔드포인트 설명 보는법
#127.0.0.1:8000/docs
from fastapi import FastAPI, Form, Query, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import sqlite3
from fastapi.templating import Jinja2Templates
import folium
import pandas as pd
import torch
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
import cv2
import numpy as np
from PIL import Image
import io

app = FastAPI(
    title="객체 탐지 API",
    description="이 API는 관광지 탐색 시스템입니다.",
    version="1.0.0"
)

# 레이블 파일 로드
with open('labels.txt', 'r', encoding="utf-8") as f:
    labels = [line.strip() for line in f.readlines()]
db=sqlite3.Connection('data.db')
lendmarkdf=pd.read_sql("select * from lendmark",db)
lendmarkdf['img'].fillna("https://cdn-icons-png.flaticon.com/512/75/75519.png",inplace=True)
net = create_mobilenetv1_ssd(len(labels), is_test=True)
state_dict = torch.load("model.pth", map_location=torch.device("cpu"))
net.load_state_dict(state_dict, strict=False)
net.eval()
predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)
item=pd.read_sql("select * from item",db)
def preprocess_image(image_bytes):
    """
    이미지 전처리 함수: 업로드된 이미지를 RGB로 변환하고
    OpenCV 형식으로 변환하여 반환합니다.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = np.array(image)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

def detect_objects(image_bytes):
    """
    객체 탐지 함수: 모델을 사용하여 이미지에서 객체를 탐지하고,
    탐지 결과를 반환합니다.
    """
    image = preprocess_image(image_bytes)
    
    # 비동기 예외 처리 (predictor.predict가 비동기 함수일 경우)
    try:
        boxes, labels_idx, scores = predictor.predict(image, 10, 0.4)
    except TypeError as e:
        raise ValueError("predictor.predict 함수 호출 중 문제가 발생했습니다.") from e
    results = []

    for i in range(boxes.size(0)):
        box = boxes[i]

        label = labels[labels_idx[i]]
        confidence = float(scores[i])
        
        # 신뢰도 임계값 확인
        if confidence < 0.4:
            continue

        results.append({
            "classID": int(labels_idx[i]),
            "className": label,
            "box": box.tolist(),
            "confidence": confidence
        })

    return results


app.mount("/img", StaticFiles(directory="image"), name="img")
templates = Jinja2Templates(directory="template")

@app.post("/content", summary="관광지 설명 제공", description="관광지의 이름을 입력하면 해당 관광지의 설명을 반환합니다.")
async def content(title: str = Form(..., description="조회할 관광지의 이름")):
    """
    ### 관광지 설명 제공
    - **title**: 조회할 관광지의 이름
    - 관광지 설명을 반환합니다.
    """
    item = pd.read_sql("select * from item",db)
    #Json으로 반환
    return item[item['name'] == title]['description'].values[0]
@app.get("/map", summary="지도 생성", description="지정된 위도와 경도에 관광지 정보가 포함된 지도를 반환합니다.")
async def map(lat: str = Query(..., description="위도"), lon: str = Query(..., description="경도")):
    """
    ### 지도 생성
    - **lat**: 위도
    - **lon**: 경도
    - 해당 위치와 주변 관광지를 표시하는 지도를 반환합니다.
    """
    m = folium.Map(location=[float(lat), float(lon)], zoom_start=12)
    for row in lendmarkdf.values:
        index=lendmarkdf[lendmarkdf['name']==row[0]].index[0]
        # 각 마커에 다이얼로그와 HTML 생성
        folium.Marker(
            [row[1], row[2]],
            popup=f"""
<dialog id="dialog{index}">
    <h1>{row[0]}</h1>
    <img src="{row[4]}" alt="{row[0]}" style="width:100%;"><br>
    <h2>{row[5]}</h2>
    <form method="dialog">
        <button>닫기</button>
    </form>
</dialog>
    <h1>{row[0]}</h1>
<button onclick="document.getElementById('dialog{index}').showModal()">자세히보기</button>
            """,
            icon=folium.Icon(color="red")
        ).add_to(m)
    # HTML 반환
    return HTMLResponse(content=m._repr_html_())

@app.get("/kind", summary="관광지 종류 조회", description="고유한 관광지 종류 목록을 반환합니다.")
async def kind():
    """
    ### 관광지 종류 조회
    - 고유한 관광지 종류 목록을 반환합니다.
    """
    #Json으로 반환
    return {'data': ['모두보기'] + list(lendmarkdf['kind'].unique())}

@app.post("/item", summary="관광지 상세 정보 조회", description="관광지 이름으로 상세 정보를 조회합니다.")
async def yujeock(name: str = Form(...,description="관광지 이름")):
    """
    ### 관광지 상세 정보 조회
    - **name**: 관광지 이름
    - 관광지의 이름과 설명을 반환합니다.
    """
    item=pd.read_sql(f"select * from item where name= ? ",db,params=(name,))
    #Json으로 반환
    return dict(zip(item.columns,item.values[0]))

@app.post("/detect", summary="객체 탐지", description="이미지에서 객체를 탐지하고 탐지 결과를 반환합니다.")
async def detect(file: UploadFile = File(..., description="분석할 이미지 파일 (JPG 또는 PNG 형식 권장)")):
    """
    ### 객체 탐지
    - **file**: 업로드할 이미지 파일
    - 업로드된 이미지에서 객체를 탐지하여 탐지된 객체 정보와 이미지 경로를 반환합니다.
    """
    contents = await file.read()
    detection_results = detect_objects(contents)
    #Json으로 반환
    return JSONResponse(content={"detections": detection_results})