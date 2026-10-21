"""Gemini grounded response composer. Never embed API keys in source."""
import re
import requests
from rag_engine import retrieve

API='https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent'

def gemini_generate(question, evidence, api_key, model='gemini-3.5-flash'):
    if not api_key:
        return None, 'Gemini is not dfd configured. Add GEMINI_API_KEY in Streamlit app secretskkk.'
    prompt=(
        'You are Ask HeatSafe AI, a careful weather and heat-health assistant for Karachi. '
        'Answer the question ONLY using EVIDENCE below. If evidence is insufficient, say so. '
        'Never invent temperatures, forecasts, document quotes, citations, medical diagnoses, or historical records. '
        'Cite evidence IDs exactly as provided, e.g. [WHO-HEAT-1] or [LIVE-OPENMETEO]. '
        'For health questions give general preventive guidance, not a personal medical diagnosis. '
        'For live weather do not assert official station observations; Open-Meteo is a model/API estimate. '
        'Use clear simple English, or Urdu if the user asks in Urdu. Keep answers concise.\n\n'
        f'QUESTION: {question}\n\nEVIDENCE:\n{evidence[:13000]}'
    )
    try:
        response=requests.post(API.format(model=model), params={'key':api_key},json={
            'contents':[{'role':'user','parts':[{'text':prompt}]}],
            'generationConfig':{'temperature':0.15,'maxOutputTokens':1100}
        },timeout=35)
        response.raise_for_status()
        data=response.json()
        answer=''.join(part.get('text','') for c in data.get('candidates',[]) for part in c.get('content',{}).get('parts',[])).strip()
        if not answer:return None,'Gemini returned no answer.'
        permitted=set(re.findall(r'\[([A-Za-z0-9_-]+)\]',evidence))
        cited=set(re.findall(r'\[([A-Za-z0-9_-]+)\]',answer))
        if cited-permitted:return None,'Gemini returned an unsupported citation; answer withheld.'
        return answer,None
    except requests.HTTPError as exc:
        status=exc.response.status_code if exc.response is not None else 'unknown'
        return None,f'Gemini request failed (HTTP {status}). Check the API key, model access, and quota in Streamlit logs.'
    except requests.RequestException:
        return None,'Gemini connection timed out or failed.'

def guidance_answer(question,api_key='',model='gemini-2.5-flash'):
    docs=retrieve(question,k=4)
    if not docs:
        return "I couldn't find supporting information in HeatSafe's trusted guidance documents, so I can't verify an answer."
    evidence='\n\n'.join(f"[{d['id']}] {d['title']} | {d['section']} | {d['url']}\n{d['text']}" for d in docs)
    if api_key:
        answer,error=gemini_generate(question,evidence,api_key,model)
        if answer:
            sources='\n'.join(f"- [{d['id']}] [{d['title']} — {d['section']}]({d['url']})" for d in docs)
            return answer+'\n\n**Retrieved sources**\n'+sources
        return (error or 'Gemini is unavailable.')+'\n\n**Retrieved evidence:**\n'+evidence
    return '**Evidence-based summary (Gemini not configured):**\n\n'+ '\n\n'.join(f"**[{d['id']}]** {d['text']} ([source]({d['url']}))" for d in docs)
