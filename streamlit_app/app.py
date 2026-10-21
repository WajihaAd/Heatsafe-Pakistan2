"""HeatSafe Pakistan | premium Streamlit weather intelligence and grounded assistant."""
from pathlib import Path
import json,re
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
from gemini_rag import guidance_answer, gemini_generate
from data_services import read_published_results

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'/'processed'/'heatsafe_ml_dataset.csv'
MODEL=ROOT/'models'/'heatsafe_live_improved_model.pkl'
RISK={0:'Low',1:'Moderate',2:'High',3:'Extreme'}
st.set_page_config(page_title='HeatSafe Pakistan | Karachi',page_icon='🌤️',layout='wide',initial_sidebar_state='collapsed')
st.markdown('''<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"],.stApp{font-family:'DM Sans',sans-serif!important}
.stApp{background:radial-gradient(ellipse at 70% 0%,#25234b 0%,#0a1225 47%,#08101e 100%);color:#edf4ff}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#171637,#0c1830);border-right:1px solid #343b61}
[data-testid="stSidebar"] *{color:#e6edff!important}
h1,h2,h3{letter-spacing:-.035em!important;color:#f4f6ff!important}
h1{font-size:2.7rem!important} h2{font-size:1.7rem!important}
[data-testid="stMetric"]{background:linear-gradient(135deg,#1a2b4a,#171b3a);border:1px solid #344668;padding:20px;border-radius:20px;box-shadow:0 12px 30px #0002}
[data-testid="stMetricLabel"]{color:#b7c7e3!important}
[data-testid="stMetricValue"]{color:#fff!important;font-weight:800!important}
[data-testid="stChatMessage"]{background:#18233c;border:1px solid #33425c;border-radius:18px;padding:14px;margin:10px 0}
.stButton>button{background:linear-gradient(90deg,#5754d9,#7754c8);color:white;border:0;border-radius:13px;padding:12px 22px;font-weight:700}
.stButton>button:hover{background:#6961e5;color:white;border:0}
.hero{padding:27px 30px;border-radius:25px;background:linear-gradient(110deg,#20254c,#233b5e 65%,#46326a);border:1px solid #4b4e82;margin:10px 0 23px}
.eyebrow{color:#b9b7ff;font-weight:700;letter-spacing:.12em;font-size:.8rem}
.hero-title{font-size:2.4rem;font-weight:800;letter-spacing:-.045em;line-height:1.12;color:white;margin:10px 0}
.hero-desc{color:#d1dafa;max-width:760px}
.insight{background:#182943;border-left:4px solid #b4a2ff;padding:17px 21px;border-radius:13px;margin:14px 0;color:#edf4ff}
.source{font-size:.86rem;color:#aebdd5}
[data-testid="stAlert"]{border-radius:14px}

/* HeatSafe Midnight Violet — weather app inspired desktop treatment */
:root{color-scheme:dark}
.stApp{background:radial-gradient(circle at 78% 8%,rgba(109,59,204,.24),transparent 32%),radial-gradient(circle at 3% 35%,rgba(53,64,143,.26),transparent 40%),linear-gradient(150deg,#070b22 0%,#101039 52%,#090a23 100%)!important}
.block-container{max-width:1380px!important;padding-top:1.25rem!important;padding-bottom:3rem!important}
[data-testid="stSidebar"]{display:none!important}
header[data-testid="stHeader"]{background:transparent!important}
[data-testid="stDecoration"]{display:none!important}
.hs-topbar{display:flex;align-items:center;justify-content:space-between;gap:14px;padding:14px 23px;margin-bottom:14px;border:1px solid rgba(162,128,255,.27);border-radius:22px;background:linear-gradient(110deg,rgba(15,18,54,.96),rgba(35,20,80,.91));box-shadow:0 12px 36px rgba(0,0,0,.22)}
.hs-brand{font-weight:850;color:#fff;font-size:1.28rem;letter-spacing:-.045em}.hs-brand span{color:#c9a5ff}
.hs-topmeta{color:#a8a8cf;font-size:.81rem}
.hs-orb{display:inline-block;width:12px;height:12px;border-radius:50%;background:#c3a5ff;box-shadow:0 0 20px #af80ff;margin-right:8px}
.hero{position:relative;overflow:hidden;min-height:174px;background:radial-gradient(circle at 82% 18%,rgba(146,89,255,.55),transparent 34%),linear-gradient(115deg,#11143e 0%,#211452 62%,#37146d 100%)!important;border:1px solid rgba(175,122,255,.33)!important;box-shadow:0 18px 46px #0004!important}
.hero:after{content:'☾';position:absolute;right:6%;top:1%;font-size:8rem;color:rgba(229,210,255,.16);pointer-events:none}
.eyebrow{color:#c9a4ff!important}
.hero-title{font-size:clamp(2rem,3.8vw,3.35rem)!important}
[data-testid="stMetric"]{background:linear-gradient(145deg,#1c154f,#10163e 72%,#19123d)!important;border:1px solid rgba(148,108,255,.28)!important;box-shadow:inset 0 1px 0 rgba(255,255,255,.06),0 12px 30px #0003!important;min-height:125px}
[data-testid="stMetricValue"]{font-size:clamp(1.6rem,2.2vw,2.4rem)!important}
[data-testid="stMetricLabel"]{color:#c1b7e6!important}
[data-testid="stChatMessage"]{background:linear-gradient(135deg,#18163e,#151b3c)!important;border:1px solid rgba(139,107,236,.35)!important}
.stButton>button,.stFormSubmitButton>button{background:linear-gradient(105deg,#6d3fd0,#7e46d5)!important;color:white!important;border:1px solid rgba(194,158,255,.35)!important;border-radius:15px!important;box-shadow:0 6px 18px #0003}
.stButton>button:hover,.stFormSubmitButton>button:hover{background:linear-gradient(105deg,#8359e3,#9954ec)!important}
.insight{background:linear-gradient(115deg,#20184e,#151c45)!important;border-left:4px solid #a779f8!important;box-shadow:0 8px 25px #0002}
[data-testid="stPlotlyChart"]{border:1px solid rgba(154,110,255,.2);border-radius:19px;background:rgba(14,14,43,.6);padding:7px}
[data-testid="stSegmentedControl"]{margin:0 0 17px}
[data-testid="stSegmentedControl"] button{border-radius:13px!important;color:#e5dcff!important;background:rgba(36,26,84,.7)!important;border:1px solid rgba(147,110,230,.22)!important;font-weight:600!important}
[data-testid="stSegmentedControl"] button[aria-checked="true"], [data-testid="stSegmentedControl"] button[aria-pressed="true"]{background:linear-gradient(110deg,#6738c5,#8a4cd6)!important;border-color:#ae80ff!important}
.stChatInputContainer textarea{background:#18173c!important}
@media(max-width:760px){.hs-topbar{padding:12px 14px}.hs-topmeta{display:none}.hero{padding:21px!important}.block-container{padding-left:.8rem!important;padding-right:.8rem!important}}
</style>''',unsafe_allow_html=True)

@st.cache_data(ttl=600,show_spinner=False)
def live_weather():
    r=requests.get('https://api.open-meteo.com/v1/forecast',params={'latitude':24.8607,'longitude':67.0011,'current':'temperature_2m,relative_humidity_2m,dew_point_2m,apparent_temperature,wind_speed_10m,precipitation,weather_code','hourly':'temperature_2m,apparent_temperature,relative_humidity_2m,precipitation_probability,weather_code,wind_speed_10m','daily':'weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max,sunrise,sunset,uv_index_max,wind_speed_10m_max','forecast_days':7,'timezone':'Asia/Karachi'},timeout=18)
    r.raise_for_status();return r.json()
@st.cache_data(show_spinner=False)
def historical():
    d=pd.read_csv(DATA);d['time']=pd.to_datetime(d['time'],utc=True,errors='coerce');d=d.dropna(subset=['time']).sort_values('time')
    d['row_id']='KHI_'+d['time'].dt.strftime('%Y%m%dT%H%M%SZ')
    return d
@st.cache_resource(show_spinner=False)
def model_bundle():
    import joblib
    return joblib.load(MODEL)
def live_status(w):
    t=pd.Timestamp(w['time'])
    t=t.tz_localize('Asia/Karachi') if t.tzinfo is None else t.tz_convert('Asia/Karachi')
    hours=(pd.Timestamp.now(tz='UTC')-t.tz_convert('UTC')).total_seconds()/3600
    return ('PASS' if -.25<=hours<=2 else 'REVIEW_REQUIRED'),hours

def predict(w):
    try:
        status,_=live_status(w)
        if status!='PASS':return None,'Prediction withheld: live observation failed freshness check.'
        bundle=model_bundle();features=bundle['features'];t=pd.Timestamp(w['time']);t=t.tz_localize('Asia/Karachi') if t.tzinfo is None else t.tz_convert('Asia/Karachi')
        vals={'meteo_temperature':w['temperature_2m'],'meteo_humidity':w['relative_humidity_2m'],'dew_point_2m':w['dew_point_2m'],'apparent_temperature':w['apparent_temperature'],'meteo_wind_speed':w['wind_speed_10m'],'hour_karachi':t.hour,'month':t.month}
        label=int(bundle['model'].predict(pd.DataFrame([vals])[features])[0]);return RISK.get(label,'Unknown'),None
    except Exception as exc:return None,f'Model unavailable ({type(exc).__name__}); no risk prediction displayed.'

def get_live():
    try:return live_weather(),None
    except Exception as exc:return None,f'Live weather unavailable: {type(exc).__name__}. Please retry later.'

def hero(kicker,title,description):
    st.markdown(f'<div class="hero"><div class="eyebrow">{kicker}</div><div class="hero-title">{title}</div><div class="hero-desc">{description}</div></div>',unsafe_allow_html=True)

def chart(x,series,title,ylabel):
    fig=go.Figure()
    for i,(name,values) in enumerate(series.items()):
        fig.add_trace(go.Scatter(x=x,y=values,name=name,mode='lines',line={'color':['#b995ff','#fbacdf','#77d2f7','#ffce8e'][i%4],'width':2.5},fill='tozeroy' if len(series)==1 else None,fillcolor='rgba(105,100,220,0.12)'))
    fig.update_layout(title={'text':title,'font':{'size':17}},height=340,margin=dict(l=15,r=10,t=50,b=25),paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',font={'color':'#cdd9ee'},xaxis={'gridcolor':'#302454'},yaxis={'title':ylabel,'gridcolor':'#302454'},legend={'orientation':'h','y':1.13})
    st.plotly_chart(fig,use_container_width=True)

def answer(question):
    q=question.lower().strip()
    weather_words=['weather','temperature','temp','humidity','wind','feels like','hot right now','karachi now','current conditions','mausam','موسم','درجہ حرارت']
    current_words=['current','now','today','right now','at the moment','karachi is','karachi weather','latest','present','abhi','ابھی','آج']
    history_words=['historical','history','highest','hottest','average','mean','past','since 2022','trend','recorded','maximum','minimum','dataset','2022','2023','2024','2025','2026']
    risk_words=['risk prediction','risk level','risk category','predicted risk','heat risk now','current risk','next hour risk']
    if any(s in q for s in risk_words):
        result,error=get_live()
        if error:return error
        w=result['current'];label,err=predict(w)
        if err:return err+' Source: Open-Meteo live weather and HeatSafe experimental model.'
        return f'**Experimental next-hour risk category: {label}.**\n\nObservation: {w["time"]} Pakistan time. This is a model classification based on weather features, not an official heat alert or medical safety determination.\n\n**Evidence:** Open-Meteo live weather API; HeatSafe trained model.'
    if (any(s in q for s in weather_words) and any(s in q for s in current_words)) or re.search(r'(?:temp|temperature|weather|humidity|wind)\s+(?:of|in|at)\s+karachi',q) or re.search(r'karachi\s+(?:temperature|weather|humidity|temp)',q) or (('karachi' in q) and any(s in q for s in ['temperature','weather','temp','humidity','wind','feels like'])):
        result,error=get_live()
        if error:return error
        w=result['current'];status,age=live_status(w)
        return f'**Karachi weather** (Open-Meteo, {w["time"]} Pakistan time)\n\n🌡️ Temperature: **{w["temperature_2m"]} °C**  \n☀️ Feels like: **{w["apparent_temperature"]} °C**  \n💧 Humidity: **{w["relative_humidity_2m"]}%**  \n🌬️ Wind: **{w["wind_speed_10m"]} km/h**\n\nFreshness check: **{status}** ({age:.1f} hours old). These are API weather estimates, not readings from an on-site sensor.\n\n**Source:** [Open-Meteo](https://open-meteo.com/).'
    if any(s in q for s in history_words) and any(s in q for s in ['temperature','temp','weather','heat','hottest','highest','average','mean','trend','history','historical','dataset']):
        try:
            d=historical();v=d['meteo_temperature'].dropna();row=d.loc[v.idxmax()];start=d['time'].min().date();end=d['time'].max().date()
            if any(s in q for s in ['highest','hottest','maximum']):return f'**Highest temperature in the HeatSafe dataset: {row["meteo_temperature"]:.1f} °C** on {row["time"]} UTC.\n\nSupporting gold-derived row ID: `{row["row_id"]}`. This is a dataset maximum, not Karachi’s official all-time record. Source: Open-Meteo historical data.'
            monthly=d.set_index('time')['meteo_temperature'].resample('MS').mean().dropna();peak=monthly.idxmax()
            return f'Across **{len(d):,} hourly rows** ({start} to {end}), the mean Open-Meteo temperature is **{v.mean():.1f} °C**. The warmest monthly mean is **{monthly.max():.1f} °C** in {peak:%B %Y}.\n\nMethod: arithmetic mean of nonmissing hourly observations; monthly means grouped by UTC month. Example supporting row IDs: `{d.iloc[0]["row_id"]}`, `{d.iloc[len(d)//2]["row_id"]}`, `{d.iloc[-1]["row_id"]}`. The full dataset is required to reproduce the aggregate.'
        except Exception:return 'I cannot load the historical dataset to verify this number.'
    try: key=st.secrets.get('GEMINI_API_KEY','')
    except Exception: key=''
    return guidance_answer(question,api_key=key)


# Enhanced visual weather design inspired by midnight-violet mobile weather applications.
st.markdown("""<style>
.stApp{background:radial-gradient(ellipse at 82% 0%,#48287555,transparent 42%),linear-gradient(160deg,#080a22,#18123e 52%,#0b0d27)!important}
.hs-topbar{background:linear-gradient(110deg,#141032,#291554,#1b174b)!important;border:1px solid #7756bb70!important}
.hero{background:radial-gradient(circle at 84% 40%,#7652c980,transparent 34%),linear-gradient(110deg,#11112e,#302067,#19143e)!important}
[data-testid="stMetric"]{border-radius:24px!important;background:linear-gradient(135deg,#33236e,#19153d)!important}
[data-testid="stChatMessage"]{border-radius:24px!important;background:#21194b!important}
.hs-weather-card{padding:22px;border-radius:24px;background:linear-gradient(135deg,#392179,#17183e);border:1px solid #8e65d955;margin:10px 0;box-shadow:0 14px 36px #0003}
.hs-weather-icon{font-size:3rem}.hs-weather-card h3{margin:0;color:#fff}.hs-weather-card p{color:#ddd2ff}
[data-testid="stTabs"] button{color:#e8dfff!important}
</style>""",unsafe_allow_html=True)

# Reference-inspired visual-only theme override; backend and application logic unchanged.
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@400;500;600&display=swap');
:root{color-scheme:dark;--hs-violet:#a779ff;--hs-pink:#ed57c7}
.stApp{background:radial-gradient(circle at 52% 0%,rgba(98,61,167,.38),transparent 38%),radial-gradient(circle at 90% 65%,rgba(87,39,139,.22),transparent 40%),radial-gradient(1px 1px at 10% 12%,#ffffff88 98%,transparent),radial-gradient(1px 1px at 37% 27%,#ffffff66 98%,transparent),radial-gradient(1px 1px at 76% 17%,#ffffff88 98%,transparent),radial-gradient(1px 1px at 82% 62%,#ffffff66 98%,transparent),radial-gradient(1px 1px at 24% 74%,#ffffff66 98%,transparent),linear-gradient(145deg,#08071c,#151034 55%,#08091d)!important;background-attachment:fixed!important;color:#f6f0ff!important}
.block-container{max-width:1300px!important;padding-top:1.5rem!important}
.hs-topbar{background:linear-gradient(120deg,rgba(24,15,57,.93),rgba(42,19,87,.82))!important;border:1px solid #9d7bd047!important;border-radius:28px!important;backdrop-filter:blur(18px)}
.hs-brand{font-size:1.45rem!important}.hs-brand span{color:#d8b8ff!important}.hs-orb{background:#f38fe1!important;box-shadow:0 0 19px #dc5fc6!important}
.hs-topmeta{color:#c8b9e4!important}
.hero{min-height:220px!important;padding:35px 38px!important;border-radius:32px!important;background:radial-gradient(circle at 85% 26%,#a375e544,transparent 31%),linear-gradient(115deg,#171035,#2d165c 66%,#39176b)!important;border:1px solid #b997ef4d!important;box-shadow:0 22px 55px #0006!important}
.hero:after{content:'☾'!important;font-family:serif;font-size:10rem!important;color:#f6ddff20!important;top:-12%!important}
.hero-title{font-family:'Playfair Display',Georgia,serif!important;font-weight:500!important;font-size:clamp(2.4rem,4.5vw,4.1rem)!important;letter-spacing:-.025em!important}
.hero-desc{color:#ddd2f5!important;font-size:1.02rem}.eyebrow{color:#e4b5ff!important}
h1,h2,h3{color:#fff8ff!important}h2,h3{font-weight:650!important}
[data-testid='stMetric']{background:linear-gradient(150deg,#35176b,#231047 65%,#1a123e)!important;border:1px solid #a97de755!important;border-radius:26px!important;box-shadow:inset 0 1px #ffffff14,0 15px 30px #0004!important;min-height:137px!important;padding:24px!important}
[data-testid='stMetricValue']{font-family:'Playfair Display',Georgia,serif!important;font-weight:500!important;font-size:clamp(2rem,3vw,3rem)!important;color:#fff!important}
[data-testid='stMetricLabel']{color:#d3c0f0!important;font-weight:500!important}
[data-testid='stChatMessage'],.hs-weather-card{background:linear-gradient(145deg,#31165f,#1b123d)!important;border:1px solid #9d70e04d!important;border-radius:26px!important;box-shadow:0 12px 35px #0003!important}
.insight{background:linear-gradient(120deg,#2d175d,#1b143e)!important;border-left:3px solid #ec69c8!important;border-radius:20px!important;color:#f4eaff!important}
[data-testid='stPlotlyChart']{background:#160f33c9!important;border:1px solid #a77ce13b!important;border-radius:25px!important;padding:12px!important}
.stButton>button,.stFormSubmitButton>button{background:linear-gradient(105deg,#6630bd,#a049cb)!important;border:1px solid #c393ed77!important;border-radius:18px!important;color:white!important;box-shadow:0 9px 22px #0004!important;transition:transform .2s,filter .2s}
.stButton>button:hover,.stFormSubmitButton>button:hover{filter:brightness(1.18);transform:translateY(-1px)}
[data-testid='stSegmentedControl'] button{background:#2a1556!important;border:1px solid #9163cb55!important;border-radius:18px!important;color:#e8dcff!important}
[data-testid='stSegmentedControl'] button[aria-checked='true'],[data-testid='stSegmentedControl'] button[aria-pressed='true']{background:linear-gradient(105deg,#6330b9,#a047ca)!important;border-color:#c99cf8!important}
[data-testid='stTextInput'] input,[data-testid='stSelectbox']>div>div,[data-testid='stChatInput'] textarea{background:#21133f!important;color:#fff!important;border:1px solid #956bd277!important;border-radius:16px!important}
[data-testid='stDataFrame'],[data-testid='stExpander']{border-radius:20px!important;overflow:hidden}
[data-testid='stAlert']{border-radius:20px!important}
@media(max-width:760px){.hero{min-height:180px!important;padding:25px 20px!important}.hero-title{font-size:2.5rem!important}.block-container{padding-left:1rem!important;padding-right:1rem!important}[data-testid='stMetric']{min-height:115px!important;padding:15px!important}}
</style>""",unsafe_allow_html=True)

WEATHER_ICONS={0:'☀️',1:'🌤️',2:'⛅',3:'☁️',45:'🌫️',48:'🌫️',51:'🌦️',53:'🌦️',55:'🌧️',61:'🌧️',63:'🌧️',65:'🌧️',71:'🌨️',73:'🌨️',75:'❄️',80:'🌦️',81:'🌧️',82:'⛈️',95:'⛈️',96:'⛈️',99:'⛈️'}

def weather_icon(code):
    try:return WEATHER_ICONS.get(int(code),'🌤️')
    except (TypeError,ValueError):return '🌤️'

def forecast_card(date,code,high,low,rain):
    return f'<div class="hs-weather-card"><div class="hs-weather-icon">{weather_icon(code)}</div><h3>{date}</h3><p><b>{high} °C</b> / {low} °C<br>Chance of rain: {rain}%</p></div>'

@st.cache_data(ttl=1800,show_spinner=False)
def air_quality():
    r=requests.get('https://air-quality-api.open-meteo.com/v1/air-quality',params={'latitude':24.8607,'longitude':67.0011,'current':'pm2_5,us_aqi','timezone':'Asia/Karachi'},timeout=15)
    r.raise_for_status();return r.json().get('current',{})

def gemini_weather_explanation(question, evidence):
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
    except Exception as e:
        st.error(f"Could not read Streamlit Secrets: {type(e).__name__}")
        return None

    if not key or not key.strip():
        st.warning(
            "GEMINI_API_KEY was not found. "
            "Check the Secrets settings of this exact Streamlit app."
        )
        return None

    text, error = gemini_generate(question, evidence, key.strip())

    if error:
        st.error(f"Gemini API error: {error}")

    return text if text else None

st.markdown("""<div class="hs-topbar"><div class="hs-brand"><span class="hs-orb"></span>☀️ HeatSafe <span>Pakistan</span></div><div class="hs-topmeta">KARACHI · WEATHER INTELLIGENCE · EXPERIMENTAL HEAT-RISK RESEARCH</div></div>""",unsafe_allow_html=True)
NAV=['🏠 Overview','📈 Weather Trends','🌦️ Forecast','💬 Ask HeatSafe AI','🛡️ Data Quality','ℹ️ About']
page=st.segmented_control('Navigate',NAV,default=NAV[0],selection_mode='single',label_visibility='collapsed',key='hs_top_navigation')
page=page or NAV[0]

if page.startswith('🏠'):
    hero('KARACHI · WEATHER INTELLIGENCE','Weather, explained.','Live weather conditions, experimental heat-risk intelligence and evidence-based guidance — in one place.')
    data,error=get_live()
    if error:st.warning(error)
    else:
        w=data['current'];status,age=live_status(w)
        cols=st.columns(4)
        for c,(label,value,unit) in zip(cols,[('🌡️ Temperature',w.get('temperature_2m'),'°C'),('☀️ Feels like',w.get('apparent_temperature'),'°C'),('💧 Humidity',w.get('relative_humidity_2m'),'%'),('🌬️ Wind',w.get('wind_speed_10m'),'km/h')]):c.metric(label,f'{value} {unit}')
        st.caption(f'Open-Meteo • {w["time"]} Pakistan time • Quality: {status} • {age:.1f} hours old')
        try:
            aq=air_quality()
            if aq:
                a,b,c=st.columns(3)
                a.metric('🌫️ PM2.5 · model estimate',f'{aq.get("pm2_5","N/A")} µg/m³')
                b.metric('🌬️ US AQI · model estimate',str(aq.get('us_aqi','N/A')))
                c.metric('🌧️ Current precipitation',f'{w.get("precipitation","N/A")} mm')
                st.caption(f'Air quality source: Open-Meteo Air Quality API · {aq.get("time","time unavailable")} PKT. These are not live OpenAQ sensor measurements.')
        except Exception:
            st.caption('Air-quality service currently unavailable; no air-quality value is shown.')

        label,err=predict(w)
        if err:st.warning(err)
        else:st.info(f'**Experimental next-hour heat-risk category: {label}.** This is not an official warning or a guarantee of safety.')
        if status=='PASS':
            with st.expander('✨ Explain these weather readings with Gemini'):
                if st.button('Generate grounded weather insight',key='gemini_live_insight'):
                    ev=(f'[LIVE-OPENMETEO] Open-Meteo API model estimates, observed time {w["time"]} Asia/Karachi: '
                        f'temperature {w["temperature_2m"]} °C; feels-like {w["apparent_temperature"]} °C; '
                        f'humidity {w["relative_humidity_2m"]}%; wind {w["wind_speed_10m"]} km/h; '
                        f'precipitation {w.get("precipitation")} mm. Freshness {status}.')
                    reply=gemini_weather_explanation('Explain the weather conditions in 3 practical sentences. Do not make unsupported safety claims.',ev)
                    if reply:st.markdown(reply)
                    else:st.info('Configure a new GEMINI_API_KEY in Streamlit Secrets to enable this feature.')
            diff=w['apparent_temperature']-w['temperature_2m']
            st.markdown(f'<div class="insight"><b>✦ What the readings mean</b><br>The feels-like temperature is <b>{abs(diff):.1f} °C {"higher" if diff>=0 else "lower"}</b> than the air temperature. Humidity is <b>{w["relative_humidity_2m"]}%</b>. Heat strain depends on exposure, activity, hydration and other factors; these readings alone cannot determine individual safety.</div>',unsafe_allow_html=True)
        hourly=data.get('hourly',{})
        if hourly.get('time'):
            times=pd.to_datetime(hourly['time']);chart(times[:36],{'Air temperature':hourly['temperature_2m'][:36],'Feels like':hourly['apparent_temperature'][:36]},'Next 36 hours · Open-Meteo forecast','°C')
            st.caption('Forecast estimates, not measured future conditions.')
    try:
        d=historical();month=d.set_index('time')['meteo_temperature'].resample('MS').mean().dropna();last=month.iloc[-1];overall=month.mean()
        st.markdown(f'<div class="insight"><b>✦ Historical perspective</b><br>Across the available dataset, the average of monthly mean temperatures is <b>{overall:.1f} °C</b>. The final complete month in the dataset averages <b>{last:.1f} °C</b>. Compare equivalent seasons before inferring climate trends.</div>',unsafe_allow_html=True)
    except Exception:pass
    st.subheader('💬 Ask HeatSafe AI')
    st.write('Ask about Karachi weather, heat-risk predictions, heat precautions or historical records.')
    with st.form('overview_ask'):
        question=st.text_input('Your question',placeholder='What is the temperature in Karachi now?');submitted=st.form_submit_button('Ask HeatSafe AI →')
    if submitted and question:st.markdown(answer(question))
elif page.startswith('🌦️'):
    hero('KARACHI · WEATHER OUTLOOK','Forecast','A practical short-range weather outlook, based on Open-Meteo model estimates.')
    data,error=get_live()
    if error:st.warning(error)
    else:
        daily=data.get('daily',{})
        if daily.get('time'):
            st.subheader('📅 7-day Karachi outlook')
            for row_start in (0,4):
                indices=list(range(row_start,min(row_start+4,len(daily['time']))))
                cols=st.columns(len(indices))
                for col,i in zip(cols,indices):
                    with col:
                        st.markdown(forecast_card(daily['time'][i],daily['weather_code'][i],daily['temperature_2m_max'][i],daily['temperature_2m_min'][i],daily['precipitation_probability_max'][i]),unsafe_allow_html=True)
            a,b,c=st.columns(3)
            a.metric('☀️ UV index · today',str(daily.get('uv_index_max',['N/A'])[0]))
            b.metric('🌅 Sunrise · PKT',str(daily.get('sunrise',['N/A'])[0]).split('T')[-1])
            c.metric('🌇 Sunset · PKT',str(daily.get('sunset',['N/A'])[0]).split('T')[-1])
        hourly=data.get('hourly',{})
        if hourly.get('time'):
            times=pd.to_datetime(hourly['time'])
            chart(times,{'Temperature':hourly['temperature_2m'],'Feels like':hourly['apparent_temperature']},'Next 48 hours · forecast temperatures','°C')
            chart(times,{'Humidity':hourly['relative_humidity_2m']},'Next 48 hours · forecast humidity','%')
            frame=pd.DataFrame({'Time (PKT)':hourly['time'],'Temperature (°C)':hourly['temperature_2m'],'Feels like (°C)':hourly['apparent_temperature'],'Humidity (%)':hourly['relative_humidity_2m']})
            st.dataframe(frame,use_container_width=True,hide_index=True)
            st.caption('Forecast values are model estimates, not observations. Source: Open-Meteo.')
        else:st.warning('Forecast data is not available at the moment.')
elif page.startswith('📈'):
    hero('HISTORICAL ANALYTICS','Weather Trends','Explore hourly historical weather from January 2022 through September 2026.')
    try:
        d=historical();a,b,c=st.columns(3);a.metric('Hourly observations',f'{len(d):,}');b.metric('Mean temperature',f'{d["meteo_temperature"].mean():.1f} °C');c.metric('Maximum in dataset',f'{d["meteo_temperature"].max():.1f} °C')
        choice=st.selectbox('Explore a measure',{'Temperature':'meteo_temperature','Feels like':'apparent_temperature','Humidity':'meteo_humidity','Wind speed':'meteo_wind_speed'}.keys());mapping={'Temperature':'meteo_temperature','Feels like':'apparent_temperature','Humidity':'meteo_humidity','Wind speed':'meteo_wind_speed'};freq=st.selectbox('Time resolution',['Monthly','Daily']);v=d.set_index('time')[mapping[choice]].resample('MS' if freq=='Monthly' else 'D').mean().dropna()
        chart(v.index,{choice:v.values},f'{freq} average {choice.lower()}','°C' if choice in ('Temperature','Feels like') else '%' if choice=='Humidity' else 'km/h')
        peak=v.idxmax();st.markdown(f'<div class="insight"><b>✦ Data insight</b><br>Highest {freq.lower()} mean in this view: <b>{v.max():.1f}</b> during <b>{peak:%B %Y}</b>. Lowest: <b>{v.min():.1f}</b>. These are descriptive statistics from the committed dataset, not a prediction.</div>',unsafe_allow_html=True)
        st.caption('Source: HeatSafe processed dataset from Open-Meteo / NASA POWER. Chart series uses the selected Open-Meteo measure.')
    except Exception as exc:st.error(f'Historical data could not be loaded: {exc}')
elif page.startswith('💬'):
    hero('EVIDENCE-BASED ASSISTANT','Ask HeatSafe AI','Ask about real Karachi weather, historical measurements and trusted heat-health guidance. Answers show evidence and limitations.')
    if 'messages' not in st.session_state:st.session_state.messages=[]
    samples=['What is the temperature in Karachi now?','What is the current heat risk prediction?','What precautions should outdoor workers take?','What is the highest historical temperature?']
    st.caption('Suggested questions')
    cols=st.columns(2)
    chosen=None
    for i,q in enumerate(samples):
        if cols[i%2].button(q,key=f'sample{i}',use_container_width=True):chosen=q
    for msg in st.session_state.messages:
        with st.chat_message(msg['role']):st.markdown(msg['content'])
    q=st.chat_input('Ask about Karachi weather or heat safety...') or chosen
    if q:
        st.session_state.messages.append({'role':'user','content':q})
        with st.chat_message('user'):st.markdown(q)
        with st.chat_message('assistant'):
            with st.spinner('Checking data and trusted sources...'):reply=answer(q)
            st.markdown(reply)
        st.session_state.messages.append({'role':'assistant','content':reply})
    if st.button('Clear conversation'):st.session_state.messages=[];st.rerun()
    st.caption('Gemini uses retrieved guidance for health questions; current weather and historical numbers come from real HeatSafe data. Set GEMINI_API_KEY in Streamlit Secrets. Answers are informational, not official alerts.')
elif page.startswith('🛡️'):
    hero('TRANSPARENCY','Data Quality & Pipeline','See what is validated, what is stale, and what has not been connected yet.')
    try:
        d=historical();a,b=st.columns(2);a.metric('Historical hourly rows',f'{len(d):,}');b.metric('Flagged temperature disagreements',int(pd.to_numeric(d['temperature_quality_flag'],errors='coerce').fillna(0).sum()))
        st.caption('Historical flags are from the committed processed dataset; they are not live quality checks.')
    except Exception as exc:st.warning(f'Historical quality unavailable: {exc}')
    data,error=get_live()
    if error:st.warning(error)
    else:
        status,age=live_status(data['current']);st.metric('Live observation freshness',status);st.caption(f'Observed {data["current"]["time"]} Pakistan time • age {age:.1f} h')
    try:url=st.secrets.get('PUBLISHED_RESULTS_URL','')
    except Exception:url=''
    pub=read_published_results(url)
    if pub['available']:
        st.success('Persistent published Prefect results are available');st.json(pub['prediction'])
        with st.expander('Published quality report'):st.json(pub['quality'])
    else:st.warning('Prefect-to-dashboard publishing not connected: '+pub['reason'])
    st.caption('A successful manual Prefect run does not prove scheduled hourly runs or durable publishing.')
else:
    hero('PROJECT INFORMATION','About HeatSafe Pakistan','A transparent research prototype that turns climate and weather data into understandable insights.')
    st.markdown('''### Data sources
- **Open-Meteo:** historical weather and independent current weather API.
- **NASA POWER:** historical climate/weather cross-checks.
- **OpenAQ:** October 2026 PM2.5 measurements (a different date range; not merged as concurrent observations).

### Model and limitations
The trained HistGradientBoostingClassifier predicts a **proxy next-hour heat-risk category** based on weather features. Risk labels were derived from apparent-temperature thresholds rather than health outcomes. Rare extreme-risk cases had weaker recall. This is not a validated medical or official meteorological alert.

### AI assistant
Current weather answers come from Open-Meteo; historical numeric answers come from the processed dataset; guidance answers come from a small source-linked document collection. The knowledge base is not yet a complete verified corpus, and generative AI is optional.

### Automation
Prefect Cloud Serverless is scheduled hourly. Durable publishing to this dashboard is **not yet configured**. This dashboard's current observations are fetched independently.

**Trusted links:** [Open-Meteo](https://open-meteo.com/) · [NASA POWER](https://power.larc.nasa.gov/) · [OpenAQ](https://openaq.org/) · [WHO](https://www.who.int/news-room/fact-sheets/detail/climate-change-heat-and-health) · [CDC/NIOSH](https://www.cdc.gov/niosh/heat-stress/about/index.html)''')
