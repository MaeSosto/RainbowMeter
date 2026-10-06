from transformers import pipeline

print("start")

pipe = pipeline("image-text-to-text", model="Qwen/Qwen3.5-2B")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What animal is on the candy?"}
        ]
    },
]
res = pipe(text=messages)
print(res)