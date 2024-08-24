from ollama_asssistant import OllamaAssistant
import json

system_prompt = "Pretend you are Paul Graham, the founder of Y Combinator."
model = "chat-pg-q8"


def chat():
    ollama_assistant = OllamaAssistant(
        system_prompt=system_prompt,
        model_name=model,
        is_chat_mode=True,
        print_messages=True,
    )
    while True:
        question = input("Ask a question: ")
        if question == "exit":
            break
        ollama_assistant.generate_answer(question)


def evaluate_assistant():
    ollama_assistant = OllamaAssistant(
        system_prompt=system_prompt, model_name=model, is_chat_mode=False
    )
    # load json file
    with open("evaluation/evaluation_dataset/pg_set.json") as f:
        data = json.load(f)
        questions = data["questions"]

    question_answers = []
    for question in questions:
        print(f"Evaluation Question: {question}")
        answer = ollama_assistant.generate_answer(question)
        question_answers.append({"question": question, "answer": answer})

    # write to results.json
    output_file = f"evaluation/evaluation_results/{model}_pg_results.json"
    with open(output_file, "w") as f:
        json.dump(question_answers, f, indent=4)


evaluate_assistant()
