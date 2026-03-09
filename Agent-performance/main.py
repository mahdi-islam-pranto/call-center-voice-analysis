from pydantic import BaseModel
from typing import List
# async
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, Query, Form, File, UploadFile
# import all functions
from process_single_audio import process_single_audio
from pydantic import Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from decouple import config

SECRET_KEY = config('OPENAI_API_KEY')
model = ChatOpenAI(model="gpt-4o-mini", openai_api_key=SECRET_KEY)

# pydantic class for API input
class TranscriptionRequest(BaseModel):
    paths: List[str]
    performance_types: List[str]
    
    
# pydantic class for Output
class MetricEvaluation(BaseModel):
    metric_name: str = Field(description="Name of the evaluation metric")
    score: int = Field(ge=0, le=10, description="Score for this metric out of 10")
    observation: str = Field(description="Explanation for the given score")


class AgentPerformanceReport(BaseModel):
    metrics: List[MetricEvaluation] = Field(
        description="Evaluation results for each metric"
    )

    total_score: int = Field(
        description="Sum of all metric scores"
    )

    max_possible_score: int = Field(
        description="Maximum possible score based on number of metrics"
    )

    performance_summary: str = Field(
        description="Overall analysis of the agent's performance"
    )

    strengths: List[str] = Field(
        description="Key strengths observed in the agent's performance"
    )

    weaknesses: List[str] = Field(
        description="Key weaknesses observed in the agent's performance"
    )

    improvement_suggestions: List[str] = Field(
        description="Actionable suggestions to train the agent"
    )

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

Your task is to evaluate the performance of a call center agent based on multiple call transcripts.

These transcripts represent different calls handled by the same agent.

GENERAL RULES:
- Only use information present in the transcripts.
- Do NOT assume or hallucinate missing information.
- If a behavior is not observed, mark it as "Not Observed".
- Be objective, analytical, and professional.
- Look for patterns across calls (repeated strengths or repeated mistakes).

SCORING RULE:
Each evaluation metric must be scored from 0 to 10.

0-2 = Very Poor / Not Performed  
3-4 = Poor  
5-6 = Average  
7-8 = Good  
9-10 = Excellent

METRIC HANDLING LOGIC:

If evaluation metrics are provided by the user:
- Evaluate ONLY those metrics.

If NO evaluation metrics are provided:
Use the following default metrics:

1. Opening Greetings  
Evaluate whether the agent started the call professionally. A proper greeting usually includes welcoming the customer, introducing the company or service, and optionally the agent's name.

2. Active Listening  
Evaluate whether the agent carefully listens to the customer without interrupting, acknowledges the customer's concerns, asks clarifying questions, and responds appropriately to what the customer says.

3. Check Resource  
Evaluate whether the agent checks necessary internal systems, documents, or knowledge bases before providing information to ensure accuracy.

4. Hold  
Evaluate how the agent manages placing the customer on hold. The agent should ask permission before putting the customer on hold, explain the reason clearly, and return within a reasonable time.

5. Correct Info  
Evaluate whether the agent provides accurate and correct information based on the customer's request or issue.

6. Complete Info  
Evaluate whether the agent provides all necessary details to fully answer the customer's question instead of giving partial or incomplete information.

7. Empathy/Tone  
Evaluate the agent's tone and emotional intelligence. The agent should sound polite, calm, respectful, and empathetic, especially when the customer has a problem or complaint.

8. Taking Ownership  
Evaluate whether the agent takes responsibility for helping the customer and demonstrates willingness to solve the problem instead of shifting responsibility.

9. Further Assistance  
Evaluate whether the agent asks the customer if they need any additional help before ending the conversation.

10. Ending Greetings  
Evaluate whether the agent closes the call professionally by thanking the customer, confirming resolution if applicable, and ending the call politely.

For each metric you must:
- Provide observations
- Provide a score out of 10

ANALYSIS REQUIREMENTS:
Your evaluation should help an admin train the agent.

Therefore include:
- Key strengths of the agent
- Key weaknesses
- Repeated mistakes across calls
- Training recommendations

OUTPUT REQUIREMENTS:
- Return structured data matching the required JSON schema.
- Ensure scores are integers between 0 and 10.
- Calculate the total score by summing all metric scores.
- Provide a professional performance summary.
- Provide practical improvement suggestions for the agent.

Be strict, consistent, and evidence-based.
"""

# user/ main prompt for AI

    user_prompt = """
Below are multiple call transcripts from the same call center agent.

-------------------------
CALL TRANSCRIPTS
{transcripts}
-------------------------

Evaluation Metrics Provided by Admin:
{evaluation_metrics}

Instructions:

1. If evaluation_metrics contains metrics, evaluate ONLY those metrics.
2. If evaluation_metrics is empty, evaluate using the default 10 QA metrics.
3. Score each metric out of 10.
4. Provide observations explaining the score.
5. Calculate the total score.
6. Provide a professional performance summary.
7. Provide improvement suggestions to help train the agent.
"""

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", user_prompt)
    ])
    
    # invoke the prompt template
    main_prompt = prompt_template.invoke({"transcripts": transcriptions_results, "evaluation_metrics": request.performance_types})
    
    model_with_structured_output = model.with_structured_output(AgentPerformanceReport)
    
    response = model_with_structured_output.invoke(main_prompt)

    agent_performance = response
    
    return {
        "agent_performance": agent_performance,
        "transcript_bearer": "Google Cloud Speech-to-Text",
        "summary_bearer": "Open AI",
    }
    
    
    # sample audio  urls
    # "https://butterfly.ihelpbd.com/RECORDINGS/MP3/20260225-093750_BUTTERFL_1_Ban_ECO_Plus_Shormi_01778952670-all.mp3",
    # "https://butterfly.ihelpbd.com/RECORDINGS/MP3/20260225-091257_BUTTERFL_0_Ban_Talk_to_Cus_RP_Shormi_09638651762-all.mp3",
    # "https://butterfly.ihelpbd.com/RECORDINGS/MP3/20260225-094725_OUTBOUND__Shormi_01718599694-all.mp3"