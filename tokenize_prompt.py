#!/usr/bin/env python3
"""
tokenize_prompt.py — Chat template + tokenizer for warm_forward.swift

Two modes:
  1. Tokenize: Apply Qwen3.5 chat template to a filing text + schema, output token JSON
  2. Decode:   Read token IDs from JSON, decode back to text

Usage:
  # Tokenize a filing with extraction schema
  python3 tokenize_prompt.py --schema entity_extract filing.txt -o prompt_tokens.json

  # Tokenize with a custom system message
  python3 tokenize_prompt.py --system "Extract all entities" filing.txt -o prompt_tokens.json

  # Tokenize raw text (no schema, just user message)
  python3 tokenize_prompt.py filing.txt -o prompt_tokens.json

  # Decode output tokens from warm_forward
  python3 tokenize_prompt.py --decode warm_output_tokens.json

  # Set max generation tokens (default 400)
  python3 tokenize_prompt.py --max-generate 800 filing.txt -o prompt_tokens.json

Requires: transformers (installed in ~/.mlx-env/)
Model path: ~/models/Qwen3.5-27B-MLX-4bit/
"""

import argparse
import json
import sys
import os

MODEL_PATH = os.path.expanduser("~/models/Qwen3.5-27B-MLX-4bit")

# ============================================================================
# extraction Schema Definitions
# ============================================================================

SCHEMAS = {
    "entity_extract": """\
Extract the following fields for each entity listed in this SEC filing exhibit.
Output a JSON array where each element has these fields:
- entity_name: string, the legal name of the entity
- jurisdiction: string, the state/country of incorporation
- dba_names: list of strings, any "doing business as" names
- ownership_pct: number or null, percentage owned by parent
- parent_entity: string or null, the direct parent entity name
- entity_type: string, one of: "subsidiary", "branch", "joint_venture", "partnership", "other"
- status: string, one of: "active", "inactive", "dissolved", "unknown"
- notes: string or null, any additional relevant information""",

    "risk_factor": """\
Extract structured risk factor data from this SEC filing section.
Output a JSON array where each element has these fields:
- risk_title: string, short title summarizing the risk
- risk_category: string, one of: "market", "credit", "operational", "regulatory", "liquidity", "strategic", "other"
- severity: string, one of: "high", "medium", "low"
- description: string, 1-2 sentence summary of the risk
- mitigants: list of strings, any mentioned mitigating factors
- regulatory_refs: list of strings, any referenced regulations or agencies""",

    "financial_summary": """\
Extract key financial metrics from this SEC filing section.
Output a JSON object with these fields:
- revenue: number or null, total revenue in millions USD
- net_income: number or null, net income in millions USD
- total_assets: number or null, total assets in millions USD
- total_liabilities: number or null, total liabilities in millions USD
- eps: number or null, earnings per share
- period: string, the fiscal period (e.g., "FY2024", "Q3 2024")
- currency: string, reporting currency (default "USD")
- notes: list of strings, any notable items or adjustments""",
}

DEFAULT_SYSTEM = (
    "You are an expert SEC filing analyst. "
    "Extract structured data from documents. "
    "Output valid JSON only — no markdown, no explanation, no code fences."
)


def get_tokenizer():
    """Load tokenizer from model path."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)


def build_chat_prompt(user_text: str, system_msg: str = DEFAULT_SYSTEM) -> str:
    """
    Build Qwen3.5 chat-format prompt string with enable_thinking=false.

    Format:
      <|im_start|>system\n{system}<|im_end|>\n
      <|im_start|>user\n{user}<|im_end|>\n
      <|im_start|>assistant\n<think>\n\n</think>\n\n
    """
    prompt = f"<|im_start|>system\n{system_msg}<|im_end|>\n"
    prompt += f"<|im_start|>user\n{user_text}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    return prompt


def tokenize_prompt(args):
    """Tokenize a filing text with optional schema into token JSON."""
    tokenizer = get_tokenizer()

    # Read the filing text
    if args.text_file == "-":
        filing_text = sys.stdin.read()
    else:
        with open(args.text_file, "r") as f:
            filing_text = f.read()

    # Build user message
    if args.schema and args.schema in SCHEMAS:
        schema_text = SCHEMAS[args.schema]
        user_msg = f"{schema_text}\n\nFiling text:\n{filing_text}"
    elif args.schema:
        # Custom schema from file
        with open(args.schema, "r") as f:
            schema_text = f.read()
        user_msg = f"{schema_text}\n\nFiling text:\n{filing_text}"
    else:
        user_msg = filing_text

    # System message
    system_msg = args.system if args.system else DEFAULT_SYSTEM

    # Build chat prompt
    prompt_str = build_chat_prompt(user_msg, system_msg)

    # Tokenize
    token_ids = tokenizer.encode(prompt_str, add_special_tokens=False)

    # Build source name
    source_name = os.path.basename(args.text_file)
    if args.schema:
        source_name = f"{source_name}_{args.schema}"
    source_name += "_nothink"

    # Output
    output = {
        "tokens": token_ids,
        "max_generate": args.max_generate,
        "source": source_name,
        "prompt_length": len(token_ids),
        "system_message": system_msg[:100] + "..." if len(system_msg) > 100 else system_msg,
    }

    output_path = args.output or "/Users/midas/Desktop/cowork/inference-across-metal/overnight_prompt_tokens.json"
    with open(output_path, "w") as f:
        json.dump(output, f)

    print(f"Tokenized: {len(token_ids)} tokens")
    print(f"Source: {source_name}")
    print(f"Max generate: {args.max_generate}")
    print(f"Saved to: {output_path}")
    print(f"\nRun inference:")
    print(f"  ./warm_forward {output_path}")


def decode_tokens(args):
    """Decode token IDs from JSON file back to text."""
    tokenizer = get_tokenizer()

    with open(args.decode, "r") as f:
        data = json.load(f)

    # Support both formats: {"tokens": [...]} and {"generated_token_ids": [...]}
    if "tokens" in data:
        token_ids = data["tokens"]
    elif "generated_token_ids" in data:
        token_ids = data["generated_token_ids"]
    else:
        print("ERROR: JSON file must contain 'tokens' or 'generated_token_ids' key")
        sys.exit(1)

    text = tokenizer.decode(token_ids, skip_special_tokens=False)

    # Print metadata if available
    if "hit_eos" in data:
        print(f"EOS hit: {data['hit_eos']}")
    if "prompt_tokens" in data:
        print(f"Prompt tokens: {data['prompt_tokens']}")
    if "generated_tokens" in data:
        print(f"Generated tokens: {data['generated_tokens']}")

    print(f"\n{'='*60}")
    print("DECODED OUTPUT")
    print(f"{'='*60}")
    print(text)
    print(f"{'='*60}")

    # Also try to parse as JSON to validate structured output
    # Strip any remaining special tokens for JSON parsing
    clean = text.strip()
    for tag in ["<think>", "</think>", "<|im_end|>", "<|endoftext|>", "<|im_start|>"]:
        clean = clean.replace(tag, "")
    clean = clean.strip()

    if clean.startswith("[") or clean.startswith("{"):
        try:
            parsed = json.loads(clean)
            print(f"\nValid JSON ({type(parsed).__name__})")
            if isinstance(parsed, list):
                print(f"  {len(parsed)} entries")
            # Save pretty-printed JSON
            json_path = args.decode.replace("_tokens.json", "_parsed.json").replace(".json", "_parsed.json")
            with open(json_path, "w") as f:
                json.dump(parsed, f, indent=2)
            print(f"  Saved parsed JSON to: {json_path}")
        except json.JSONDecodeError as e:
            print(f"\nJSON parse failed: {e}")
            print(f"  First 200 chars: {clean[:200]}")


def list_schemas(args):
    """List available extraction schemas."""
    print("Available extraction schemas:")
    for name, desc in SCHEMAS.items():
        first_line = desc.split("\n")[0]
        print(f"  {name:20s} — {first_line}")


def main():
    parser = argparse.ArgumentParser(
        description="Tokenize prompts for warm_forward.swift / decode output tokens"
    )
    parser.add_argument("text_file", nargs="?", help="Filing text file to tokenize")
    parser.add_argument("--decode", metavar="JSON", help="Decode token IDs from JSON file")
    parser.add_argument("--schema", "-s", help="extraction schema name or custom schema file path")
    parser.add_argument("--system", help="Custom system message (overrides default)")
    parser.add_argument("--max-generate", type=int, default=400, help="Max tokens to generate (default: 400)")
    parser.add_argument("-o", "--output", help="Output JSON path (default: overnight_prompt_tokens.json)")
    parser.add_argument("--list-schemas", action="store_true", help="List available extraction schemas")

    args = parser.parse_args()

    if args.list_schemas:
        list_schemas(args)
        return

    if args.decode:
        decode_tokens(args)
        return

    if not args.text_file:
        parser.print_help()
        sys.exit(1)

    tokenize_prompt(args)


if __name__ == "__main__":
    main()
