#!/usr/bin/env python3
import argparse
import json
import random


def make_example():
    mode = random.choice(["translate", "summarize", "classify", "rewrite", "qa"])
    if mode == "translate":
        a = random.randint(1, 1000)
        return {
            "instruction": f"Translate the following sentence into English and mention the number {a}.",
            "input": f"今天天气很好，编号是 {a}。",
            "output": f"The weather is nice today, and the number is {a}.",
        }
    if mode == "summarize":
        n = random.randint(3, 9)
        text = " ".join([f"item{idx}" for idx in range(n)])
        return {
            "instruction": "Summarize the following list in one short English sentence.",
            "input": text,
            "output": f"This list contains {n} items.",
        }
    if mode == "classify":
        label = random.choice(["positive", "negative", "neutral"])
        return {
            "instruction": "Classify the sentiment of the sentence in one English word.",
            "input": f"This synthetic sample should be treated as {label}.",
            "output": label,
        }
    if mode == "rewrite":
        seed = random.randint(100, 999)
        return {
            "instruction": "Rewrite the sentence in simple English.",
            "input": f"The synthetic identifier for this sample is {seed}.",
            "output": f"This sample ID is {seed}.",
        }
    a = random.randint(1, 50)
    b = random.randint(1, 50)
    return {
        "instruction": "Answer the arithmetic question with a short English sentence.",
        "input": f"What is {a} plus {b}?",
        "output": f"The answer is {a + b}.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    with open(args.output, "w", encoding="utf-8") as f:
        for _ in range(args.num_samples):
            ex = make_example()
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(args.output)


if __name__ == "__main__":
    main()
