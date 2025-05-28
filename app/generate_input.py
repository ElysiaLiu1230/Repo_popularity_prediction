import json
import random
import os

# 不同规模的数据样本数
sizes = [10, 100, 500, 1000, 2000]

# 可选语言池
languages = ["Python", "JavaScript", "TypeScript", "Java", "Go", "Rust", "C++"]

def generate_repo(index):
    return {
        "full_name": f"example{index}/project-{index}",
        "html_url": f"https://github.com/example{index}/project-{index}",
        "forks_count": random.randint(1000, 50000),
        "open_issues_count": random.randint(0, 1000),
        "size": random.randint(1000, 500000),
        "language": random.choice(languages),
        "has_wiki": random.randint(0, 1),
        "has_pages": random.randint(0, 1),
        "has_downloads": random.randint(0, 1)
    }

# 创建输出目录（可选）
os.makedirs("inputs", exist_ok=True)

for size in sizes:
    data = [generate_repo(i) for i in range(1, size + 1)]
    filename = f"inputs/input_{size}.json"
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    print(f"✅ Generated: {filename}")

