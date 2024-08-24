import pandas as pd
from fastapi import FastAPI, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import folium

lendmark = pd.read_csv('lendmark.csv')
app = FastAPI()
lendmark.fillna('https://maps.gstatic.com/tactile/pane/default_geocode-2x.png',
          inplace=True)
templates = Jinja2Templates(directory="template")


@app.get("/")
async def kind(request: Request):
    result = ""
    for i in lendmark['종류'].unique():
        result += f'<option value="{i}">{i}</option>'
    return templates.TemplateResponse(request, 'index.html', {'kind': result})


@app.post("/map", response_class=HTMLResponse)
async def getInfo(x: float = Form(...),
                  y: float = Form(...),
                  kind: str = Form(""),
                  distance: int = Form(...)):
    if x == "":
        return HTMLResponse(
            "<script>alert('비정상적인 접근');window.history.back();</script>")

    kind = "" if kind == "모두보기" else kind
    x = float(x)
    y = float(y)
    distance = float(distance)
    rf = lendmark[lendmark['종류'].str.contains(kind)].copy()
    rf['거리'] = rf.apply(lambda d: geopy.distance.distance(
        (y, x), (d['위도'], d['경도'])).km,
                        axis=1)
    rf = rf[rf['거리'] <= distance]

    if rf.empty:
        return HTMLResponse(
            """근처에 관광지가 없습니다<br><button onclick="window.location.href='/'">다시해보기</button><br>"""
        )

    near = rf['거리'].min()
    m = folium.Map(location=[y, x], zoom_start=13)
    folium.Marker([y, x], tooltip="현재위치",
                  icon=folium.Icon(color="blue")).add_to(m)
    folium.Circle([y, x], radius=distance * 1000).add_to(m)
    for _, row in rf.iterrows():
        folium.Marker(
            [row['위도'], row['경도']],
            popup=f'''
            <form action="/content" method="post"  target="_self">
                <input type="hidden" name="title" value="{row['이름']}">
                <input type="hidden" name="content" value="{row['설명']}">
                <input type="hidden" name="img" value="{row['이미지주소']}">
                <button type="submit">
                설명보기
                </button>
            </form>''',
            tooltip=row['이름'],
            icon=folium.Icon(
                color="red",
                icon="star" if row['거리'] == near else "circle")).add_to(m)
    m.get_root().html.add_child(
        folium.Element(
            """<button onclick="window.location.href='/'">다시해보기</button><br>"""
        ))
    code = m._repr_html_().replace(
        '<span style="color:#565656">Make this Notebook Trusted to load map: File -> Trust Notebook</span>',
        "")
    return code


@app.post("/lendmark")
async def lendmark(kind: str = Form(...)):
    tf = lendmark[lendmark['종류'] == kind] if kind != "모두보기" else lendmark.copy()
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
        } for row in lendmark.values]
    }


@app.get("/kind")
async def kind():
    return {'data': ['모두보기'] + list(lendmark['종류'].unique())}

@app.post("/content")
async def content(title:str=Form(...)):
    item=pd.read_csv('item.csv')
    return item[item['이름']==title]['설명'].values[0]