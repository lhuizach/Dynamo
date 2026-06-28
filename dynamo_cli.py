from __future__ import annotations
import argparse
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Dynamo compound coding AI — generate code from natural-language requests"
    )
    p.add_argument("prompt", nargs="?", default=None, help="Coding request (single-shot mode)")
    p.add_argument("--interactive", action="store_true", help="Start an interactive REPL")
    p.add_argument("--show-router", action="store_true", help="Print the router's precise instruction")
    p.add_argument("--checkpoint", default="dynamo/final.pt", help="Path to Dynamo .pt checkpoint")
    p.add_argument("--tokenizer", default="tokenizer/dynamo.json", help="Path to tokenizer JSON")
    p.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    p.add_argument("--max-tokens", type=int, default=512, help="Max new tokens to generate")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.prompt is None and not args.interactive:
        print("Provide a prompt or pass --interactive", file=sys.stderr)
        sys.exit(1)

    from inference.pipeline import DynamoPipeline

    pipeline = DynamoPipeline(
        checkpoint=args.checkpoint,
        tokenizer_path=args.tokenizer,
        temperature=args.temperature,
        max_new_tokens=args.max_tokens,
    )

    if args.interactive:
        print("Dynamo REPL — Ctrl-C or type 'exit' to quit")
        while True:
            try:
                user_input = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if user_input.lower() in ("exit", "quit"):
                break
            if not user_input:
                continue
            print(pipeline.run(user_input, show_router=args.show_router))
    else:
        print(pipeline.run(args.prompt, show_router=args.show_router))


if __name__ == "__main__":
    main()
