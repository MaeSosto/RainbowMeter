# lsof -ti :1234 | xargs kill -9
# lms server start
lms get qwen/qwen3-4b-2507
ollama pull qwen3:8b
lms get qwen/qwen3-30b-a3b-2507
lms get google/gemma-3-4b
lms get google/gemma-3-12b
lms get google/gemma-3-27b
lms get mistralai/ministral-3-3b
lms get mistralai/ministral-3-8b
lms get mistralai/ministral-3-14b
ollama pull deepseek-r1:1.5b
ollama pull deepseek-r1:8b
ollama pull deepseek-r1:32b