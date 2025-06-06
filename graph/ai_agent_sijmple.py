import os
import re
from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()


class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]


llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-001", temperature=0.7)


def process(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    state["messages"].append(AIMessage(content=response.content))
    return state


graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

file_name_starts = "Soimething something"


def natural_sort_key(s):
    """Natural sort key function for numerical ordering of filenames."""
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split("([0-9]+)", s)
    ]


def get_ordered_subtitle_files(folder_path):
    """Get subtitle files sorted in numerical order from Player.srt to Player_148.srt"""
    files = []
    for f in os.listdir(folder_path):
        if f.startswith(file_name_starts) and f.endswith(".srt"):
            files.append(f)

    # Special case: "filename.srt" should come first
    base_file = f"{file_name_starts}.srt"
    if base_file in files:
        files.remove(base_file)
        files.sort(key=natural_sort_key)
        files.insert(0, base_file)
    else:
        files.sort(key=natural_sort_key)

    return [os.path.join(folder_path, f) for f in files]


def extract_code_samples(content):
    """Extract potential code samples from subtitle content"""
    # This pattern looks for text that appears to be code
    code_pattern = r"(?:^|\n)((?:\s{4,}|\t).+?)(?=\n\S|\Z)"
    return re.findall(code_pattern, content, re.DOTALL)


def process_subtitle_file(file_path, conversation_history):
    """Process a subtitle file with conversation context"""
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    code_samples = extract_code_samples(content)
    code_context = (
        "\n\nExtracted code samples:\n"
        + "\n".join(f"```\n{sample}\n```" for sample in code_samples)
        if code_samples
        else "No code samples found"
    )

    prompt = f"""Analyze this subtitle content and create a technical lesson focusing on:
    1. Key concepts with clear explanations
    2. Code samples that illustrate these concepts
    3. Practical applications
    4. Connections to previous lessons
    
    Maintain a professional, tutorial-style tone .
    
    Subtitle content:
    {content[:5000]}{code_context}
    """

    # Maintain exactly 5 items in conversation history
    if len(conversation_history) >= 5:
        conversation_history = conversation_history[-4:]

    conversation_history.append(HumanMessage(content=prompt))
    result = agent.invoke({"messages": conversation_history})
    response = result["messages"][-1].content

    # Add the AI response to history
    conversation_history.append(AIMessage(content=response))
    return response, conversation_history


def generate_lesson(output_folder, file_path, lesson_number, content):
    """Generate a lesson file with standardized naming"""
    base_name = os.path.basename(file_path)
    output_name = f"lesson_{lesson_number:03d}_{base_name.replace('.srt', '.md')}"
    output_path = os.path.join(output_folder, output_name)

    with open(output_path, "w", encoding="utf-8") as out_file:
        out_file.write(f"# Lesson {lesson_number}: {base_name.replace('.srt', '')}\n\n")
        out_file.write(content)

    return output_path


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_folder = os.path.join(script_dir, "subtitles")
    output_folder = os.path.join(script_dir, "readme_output")

    os.makedirs(input_folder, exist_ok=True)
    os.makedirs(output_folder, exist_ok=True)

    subtitle_files = get_ordered_subtitle_files(input_folder)
    conversation_history = []

    for i, file_path in enumerate(subtitle_files, 1):
        try:
            print(
                f"Processing {i}/{len(subtitle_files)}: {os.path.basename(file_path)}"
            )

            response, conversation_history = process_subtitle_file(
                file_path, conversation_history
            )

            output_path = generate_lesson(output_folder, file_path, i, response)
            print(f"Saved: {os.path.basename(output_path)}")

        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")


if __name__ == "__main__":
    main()
    print("Lesson generation complete!")
