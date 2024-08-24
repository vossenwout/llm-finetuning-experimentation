import os
import re
from bs4 import BeautifulSoup
import html2text
import requests

ALL_ESSAYS_URL = "https://www.paulgraham.com/articles.html"

def get_pg_essays():
    response = requests.get(ALL_ESSAYS_URL)
    soup = BeautifulSoup(response.text, "html.parser")
    links = soup.find_all("a")
    pg_essays = []
    for link in links:
        href = link.get("href")
        if "rss" in href or "https:/" in href or "index" in href:
            continue
        pg_essays.append("https://www.paulgraham.com" + "/" + href)
    return pg_essays


def get_essay_text2(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    essay = soup.find("font", {"face": "verdana"})
    essay_text = ""

    for br in essay.find_all("br"):
        br.replace_with("\n")
    
    for i, p in enumerate(essay):
        try:
            if i == 0 or (p.text[0] == "[" and p.text[-1] == "]"):
                continue
            if p.text == "Notes":
                break
            essay_text += p.text
        except Exception as e:
            print(f"Exception at line {i}: {e}")    
    # remove newlines at the beginning and end of the essay
    essay_text = essay_text.strip()


    # remove double spaces
    essay_text = essay_text.replace("  ", " ")
    # remove extra newlines

    essay_text = essay_text.replace("\n\n\n", "\n\n")
    return essay_text


def clean_essay_text(essay_text):
    try:
        # Remove everything after "Notes"
        essay_text = essay_text.split("Notes")[0]
    except Exception as e:
        pass

    try:
        # Replace triple newlines with double newlines
        essay_text = essay_text.replace("\n\n\n", "\n\n")
    except Exception as e:
        pass

    try:
        # Replace [number] with an empty string
        essay_text = re.sub(r'\[\d+\]', '', essay_text)
    except Exception as e:
        pass

    try:
        # Remove leading and trailing whitespace
        essay_text = essay_text.strip()
    except Exception as e:
        pass

    try:
        # Remove the first line
        essay_text = essay_text.split("\n", 1)[1]
    except Exception as e:
        pass

    try:
        # Remove leading and trailing whitespace again (after removing the first line)
        essay_text = essay_text.strip()
    except Exception as e:
        pass

    return essay_text
def get_essay_text(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    # Convert HTML to Markdown-like text
    html_content = str(soup)
    text_maker = html2text.HTML2Text()
    text_maker.ignore_links = True  # Optional: ignore links in the output
    text_maker.ignore_images = True  # Optional: ignore images in the output
    text_maker.bypass_tables = False  # Optional: convert tables to text
    formatted_text = text_maker.handle(html_content)
    return clean_essay_text(formatted_text)


def download_pg_essays():
    pg_essays = get_pg_essays()
    # create directory if it doesn't exist
    if not os.path.exists("finetune_dataset/data/pg_essays"):
        os.makedirs("finetune_dataset/data/pg_essays")
    for i, url in enumerate(pg_essays):
        essay_text = get_essay_text(url)
        # save in finetune_dataset/data/pg_essays
        with open(f"finetune_dataset/data/pg_essays/pg_essay_{i}.txt", "w") as f:
            f.write(essay_text)
        print(f"Downloaded essay {i}")

download_pg_essays()

#example_essay = "https://www.paulgraham.com/future.html"

#print(get_essay_text(example_essay))