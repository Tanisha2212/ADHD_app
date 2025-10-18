import requests
from bs4 import BeautifulSoup

def fetch_lessons():
    search_queries = [
        "Force and Motion 8th grade science",
        "Atoms and Molecules 8th grade",
        "Light and Reflection 8th grade",
        "Sound 8th grade science",
        "Solar System 8th grade"
    ]

    lessons = []

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/114.0.0.0 Safari/537.36"
    }

    try:
        for query in search_queries:
            search_url = f"https://www.youtube.com/results?search_query={query.replace(' ', '+')}"
            response = requests.get(search_url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')

            video_tags = soup.select("a#video-title")
            if not video_tags:
                continue

            vid = video_tags[0]
            title = vid.get('title', '8th Grade Science Lesson')
            href = "https://www.youtube.com" + vid['href']
            lessons.append({
                "title": title,
                "url": href,
                "description": f"Learn about {title} in this interactive science video designed for 8th-grade students."
            })

    except Exception as e:
        print("Scraping failed:", e)

    # Fallback default data if YouTube scraping fails
    if not lessons:
        lessons = [
            {
                "title": "Force and Motion - 8th Grade Science",
                "url": "https://www.youtube.com/watch?v=ur0hCdne2Ew",
                "description": "Learn the basics of force, motion, and Newton’s laws."
            },
            {
                "title": "Atoms and Molecules - Class 8",
                "url": "https://www.youtube.com/watch?v=fLrL4MZL0uw",
                "description": "Understand atoms, molecules, and how elements combine."
            },
            {
                "title": "Light and Reflection - Class 8 Science",
                "url": "https://www.youtube.com/watch?v=MMkLqnE9QkQ",
                "description": "Explore light, reflection, and how mirrors work."
            },
            {
                "title": "Sound - Class 8 Science",
                "url": "https://www.youtube.com/watch?v=4VB2jkoD-f8",
                "description": "Discover how sound travels and how we hear."
            },
            {
                "title": "The Solar System - Class 8",
                "url": "https://www.youtube.com/watch?v=libKVRa01L8",
                "description": "Explore the planets, moons, and the mysteries of our solar system."
            }
        ]

    return lessons
