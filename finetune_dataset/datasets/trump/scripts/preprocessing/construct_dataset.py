import json
import os
import random


def open_file(file_path):
    with open(file_path, "r") as file:
        return file.read()


def get_file_paths_from_directory(directory):
    return [os.path.join(directory, filename) for filename in os.listdir(directory)]


def parse_turn_from_line(line, user_tag="User:", assistant_tag="Assistant:"):
    if user_tag in line:
        return {"role": "user", "content": line.replace(user_tag, "").strip()}
    elif assistant_tag in line:
        return {"role": "assistant", "content": line.replace(assistant_tag, "").strip()}
    else:
        return {"role": "same", "content": line.strip()}


def construct_conversations_from_interview(interview_transcript):
    conversations = []

    # split interview on new lines
    interview_lines = interview_transcript.split("\n")

    min_assistant_turns = random.randint(1, 2)

    current_assistant_turns = 0
    current_user_turns = 0
    # sample conversation with 1 turn
    # [
    #    {"role": "user", "content": question},
    #    {"role": "assistant", "content": answer},
    # ]

    current_conversation = []
    for line in interview_lines:
        if (
            current_assistant_turns >= min_assistant_turns
            and current_user_turns >= 1
            and current_user_turns >= 1
        ):
            min_assistant_turns = random.randint(1, 2)
            conversations.append(current_conversation)
            current_conversation = []
            current_assistant_turns = 0
            current_user_turns = 0

        turn = parse_turn_from_line(
            line, user_tag="INTERVIEWER:", assistant_tag="TRUMP:"
        )
        if turn["role"] == "same":
            if len(current_conversation) > 0:
                current_conversation[-1]["content"] += " " + turn["content"]
            continue
        elif turn["role"] == "assistant":
            current_assistant_turns += 1
            if current_user_turns > 0:
                current_conversation.append(turn)
        elif turn["role"] == "user":
            current_user_turns += 1
            current_conversation.append(turn)

    return conversations


# save conversations to jsonl file
def save_conversations_to_jsonl(conversations, outdir, file_name):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    file_path = os.path.join(outdir, file_name)
    with open(file_path, "w") as file:
        for conversation in conversations:
            item = {"conversations": conversation}
            json_string = json.dumps(
                item, ensure_ascii=False
            )  # Serialize to a JSON formatted string
            file.write(f"{json_string}\n")


def main():

    file_paths = get_file_paths_from_directory(
        "finetune_dataset/datasets/trump/clean_interviews"
    )

    for file_path in file_paths:
        interview_text = open_file(file_path)
        conversations = construct_conversations_from_interview(interview_text)
        outdir = "finetune_dataset/datasets/trump/conversation"
        file_name = file_path.split("/")[-1].replace(".txt", ".jsonl")
        save_conversations_to_jsonl(conversations, outdir, file_name)


main()
