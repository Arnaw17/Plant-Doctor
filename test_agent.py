from app.core.agent import plant_doctor_agent
from langchain_core.messages import HumanMessage

initial_state = {
    "messages": [
        HumanMessage(content=(
            "I have a Tomato plant. It has clear dark brown concentric circles on its lower leaves. "
            "My agriculture teacher confirmed it is Early Blight disease. Can you write a copper fungicide treatment plan "
            "and save this final diagnosis report to my account history?"
        ))
    ],
    "species": "",
    "symptoms": [],
    "diagnosis_ready": False
}

print("Running live End-to-End Extraction & Storage Node...")
final_output = plant_doctor_agent.invoke(initial_state)