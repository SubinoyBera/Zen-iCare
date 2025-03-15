import os
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool
from langchain_groq import ChatGroq
from tools import RetrievalTool
import google.generativeai as genai

from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
llm1= genai.GenerativeModel('gemini-pro')

os.environ["GROQ_API_KEY"] ="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"

llm2= ChatGroq(
    model="llama3-70b-8192",
    api_key=os.getenv('GROQ_API_KEY')
)

DoctorAgent= Agent(
    role="Medical Doctor AI Agent",
    goal="Diagnose disease, symptoms and suggest possible remedies based on knowledge base. If you cannot find answer, ask the Web Search Agent.",
    backstory="You are a professional AI Doctor that first uses the internal knowledge base. If no answer is found, you consult the Web Search Agent to get the answer.",
    llm=llm1,
    tools=[RetrievalTool]
)

WebSearchAgent= Agent(
    role="Web Search Agent",
    goal="Search and provide the latest medical information from the web",
    backstory="You are an expert web researcher focused on gathering relevant medical from the web when asked.",
    llm=llm2,
    tools=[SerperDevTool()]
)

doctor_task= Task(
    description=(
        "Analyze patient's query, symptoms, etc. and predict disease using knowledge base. Also provide relevent treatment suggestions how to get cure."
        "If unsure or no answer, consult Web Search Agent."
    ),
    agent= DoctorAgent
)

med_crew= Crew(
    agents=[DoctorAgent, WebSearchAgent],
    tasks=[doctor_task]
)


if __name__=="main":
    user_query= input("User: "),
    
    result= med_crew.kickoff(inputs={"query": user_query})
    print("Result: ")
    
