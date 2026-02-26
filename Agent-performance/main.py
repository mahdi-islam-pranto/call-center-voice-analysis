from pydantic import BaseModel
from typing import List
# async
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, Query, Form, File, UploadFile
# import all functions
from process_single_audio import process_single_audio
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from decouple import config

SECRET_KEY = config('OPENAI_API_KEY')
model = ChatOpenAI(model="gpt-4o-mini", openai_api_key=SECRET_KEY)

# pydantic class for API input
class TranscriptionRequest(BaseModel):
    paths: List[str]
    performance_types: List[str]

# create thread pool
executor = ThreadPoolExecutor(max_workers=5)

app = FastAPI()

#get request
@app.post("/agent-performance")
async def api(request: TranscriptionRequest):
    # get event loop
    loop = asyncio.get_event_loop()
    
    # process each audio file in parallel
    tasks = [
        loop.run_in_executor(
            executor,
            process_single_audio,
            path,
            request.performance_types
        )
        for path in request.paths
    ]
    
    # wait for all tasks to complete
    transcriptions_results = await asyncio.gather(*tasks)
    
    # generate agent performance code
    system_prompt = """
    You are a Senior Call Center Quality Assurance (QA) Auditor.

    Your task is to evaluate a call center agent's overall performance based on MULTIPLE call transcripts. You always respond in BANGLA language.

    The transcripts represent different calls handled by the same agent.

    GENERAL RULES:
    - Base your evaluation ONLY on the transcripts provided.
    - Do NOT assume missing information.
    - Do NOT hallucinate.
    - If something is not observed, explicitly state "Not Observed".
    - Be objective, analytical, and professional.
    - Evaluate patterns across calls (consistency, repeated mistakes, strengths).

    METRIC HANDLING LOGIC:

    1. If evaluation metrics are provided:
    - Evaluate ONLY those metrics.
    - Be strict and analytical.
    - Provide observations supported by transcript evidence.
    - Give each metric a score from 1 to 5.

    2. If NO evaluation metrics are provided:
    - Automatically evaluate using essential industry-standard metrics:
            - Opening & Greeting
            - Professionalism
            - Communication Clarity
            - Product/Service Knowledge
            - Problem Handling & Resolution
            - Empathy & Customer Handling
            - Compliance & Risk
            - Closing Quality
            - Overall Tone & Consistency
    - Provide structured scoring for each.

    SCORING RULE:
    1 = Very Poor
    2 = Poor
    3 = Average
    4 = Good
    5 = Excellent

    OUTPUT STRUCTURE:

    ### Agent Performance Report (Multi-Call Analysis)

    #### Overall Performance Summary
    - Key Strengths:
    - Key Weaknesses:
    - Consistency Across Calls:
    - Risk Flags (if any):

    Then for each evaluated metric:

    #### Metric Name:
    Observations:
    - (Clear analytical explanation based on transcripts)
    Score (1-5):

    Finally include:

    ### Final Evaluation
    Overall Average Score:
    Performance Level:
    (Excellent / Good / Needs Improvement / Poor)

    Be precise. Be evidence-based. Be consistent. Respond in BANGLA language.
    """

    user_prompt = """
Agent Call Transcripts (Multiple Calls):

-----------------------
{transcripts}
-----------------------

Evaluation Metrics Requested:
{evaluation_metrics}

Instructions:
- If evaluation_metrics is empty or not provided, evaluate using the most important industry-standard call center performance factors.
- If evaluation_metrics contains specific metric names, evaluate ONLY those metrics.
- Provide a structured, professional performance report.
"""

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", user_prompt)
    ])
    
    # invoke the prompt template
    main_prompt = prompt_template.invoke({"transcripts": transcriptions_results, "evaluation_metrics": request.performance_types})
    
    
    response = model.invoke(main_prompt)

    agent_performance = response.content
    
    return {
        "agent_performance": agent_performance,
        "transcript_bearer": "Google Cloud Speech-to-Text",
        "summary_bearer": "Open AI",
    }
    
    
    # sample audio  urls
    # "https://butterfly.ihelpbd.com/RECORDINGS/MP3/20260225-093750_BUTTERFL_1_Ban_ECO_Plus_Shormi_01778952670-all.mp3",
    # "https://butterfly.ihelpbd.com/RECORDINGS/MP3/20260225-091257_BUTTERFL_0_Ban_Talk_to_Cus_RP_Shormi_09638651762-all.mp3",
    # "https://butterfly.ihelpbd.com/RECORDINGS/MP3/20260225-094725_OUTBOUND__Shormi_01718599694-all.mp3"
    