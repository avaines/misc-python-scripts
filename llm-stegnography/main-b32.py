#!/usr/bin/env python3
"""
LLM Steganography CLI
Encode or decode secret messages in seemingly natural text using LLM token selection.
"""
import argparse
import dotenv
import os
import sys
import torch  # PyTorch for running the model
from tqdm.auto import tqdm  # Nice progress bars
from transformers import AutoTokenizer, AutoModelForCausalLM  # HuggingFace transformers and tools
from transformers import logging as transformers_logging

dotenv.load_dotenv()

# Suppress transformer warnings (like tied weights warnings)
transformers_logging.set_verbosity_error()


# Define the character set with direct 1-to-1 mapping to top 32 tokens
char_set = " abcdefghijklmnopqrstuvwxyz.,!?"
char_to_index = {c: i for i, c in enumerate(char_set)}  # 0-30 for chars
index_to_char = {i: c for c, i in char_to_index.items()}
UPPERCASE_MARKER = 31  # Index 31 for uppercase marker


def encode_secret_message(txt):
    """Convert message to list of indices (0-31) for direct token selection."""
    encoded = []
    for c in txt:
        if c.lower() in char_set:
            if c != c.lower():
                # Add uppercase marker
                encoded.append(UPPERCASE_MARKER)
            encoded.append(char_to_index[c.lower()])

    return encoded


def decode_secret_message(indices):
    """Convert list of indices back to message."""
    decoded = ""
    uppercase_next = False
    for idx in indices:
        if idx == UPPERCASE_MARKER:
            uppercase_next = True
        else:
            char = index_to_char.get(idx, "")
            if uppercase_next:
                char = char.upper()
                uppercase_next = False
            decoded += char
    return decoded


def encode_steganographic_text(prompt, secret_message):
    # Convert prompt text to token IDs (PyTorch tensor)
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Convert secret message to list of indices (0-31)
    encoded_indices = encode_secret_message(secret_message)

    for char_index in tqdm(encoded_indices): # One token per character
        # Get model predictions without computing gradients (faster)
        with torch.no_grad():
            logits = model(input_ids).logits

        next_token_logits = logits[:, -1, :]  # Predictions for next token
        probabilities = torch.nn.functional.softmax(next_token_logits, dim=-1)

        # Get top 32 token candidates (for direct 1-to-1 mapping)
        top_token_ids = torch.topk(probabilities, 32).indices

        # Choose token based on character index (0-31)
        next_token = top_token_ids[0, char_index]

        # Append chosen token to the sequence
        input_ids = torch.cat([input_ids, next_token[None, None]], dim=-1)

    # Convert final token IDs back to readable text
    return tokenizer.decode(input_ids[0], skip_special_tokens=True)


def decode_steganographic_text(text, start=0):
    # Convert text to token IDs
    ids = tokenizer.encode(text, return_tensors="pt")
    decoded_indices = []

    # Start with just the prompt tokens (first 'start' tokens)
    context_ids = ids[:, :start]

    # Process each token one at a time slowly building the context up, same as encoding
    for i in tqdm(range(start, len(ids[0]))):
        with torch.no_grad():
            logits = model(context_ids).logits
        next_token_logits = logits[:, -1, :] # Get predictions for next position
        top_token_ids = torch.topk(next_token_logits, 32).indices

        # Get the actual token that appears in the text
        actual_token = ids[0][i]

        # Find which of the top 32 positions this token was in
        position = -1
        for pos in range(32):
            if actual_token == top_token_ids[0, pos]:
                position = pos
                break

        if position == -1:
            position = 0  # Not in top 32, default to 0

        decoded_indices.append(position)

        # Add this token to context for next iteration (this is why it must be incremental!)
        context_ids = torch.cat([context_ids, actual_token[None, None]], dim=-1)

    # Convert indices back to the secret message
    return decode_secret_message(decoded_indices)


def read_file(filepath):
    """Read text from a file."""
    with open(filepath, 'r') as f:
        return f.read().strip()


def write_file(filepath, content):
    """Write text to a file."""
    with open(filepath, 'w') as f:
        f.write(content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Encode or decode secret messages in AI-generated text"
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('-e', '--encode', metavar='MESSAGE', help='Encode secret MESSAGE (text)')
    mode.add_argument('-d', '--decode', metavar='FILE', help='Decode steganographic text from FILE')

    parser.add_argument('-o', '--output', help='Output file (required for encoding)')
    parser.add_argument('--model', default='meta-llama/Llama-3.1-8B',
                       help='HF Model to use (default: meta-llama/Llama-3.1-8B)')

    args = parser.parse_args()

    # Validate: encoding requires -o
    if args.encode and not args.output:
        parser.error('-e/--encode requires -o/--output')

    # Check for HuggingFace token
    if not os.environ.get('HF_HUB_TOKEN'):
        print("Error: HF_HUB_TOKEN not found in environment", file=sys.stderr)
        sys.exit(1)

    hf_token = os.environ['HF_HUB_TOKEN']
    model_id = args.model

    # Load model and tokenizer
    print(f"Loading model: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=hf_token
    )

    print("Model loaded successfully")

    # prompt =  "When I was little, my neighbor had a garden full of sunflowers that seemed to tower over me like giants.",
    # prompt = "Every summer, my family would drive to the lake house where my uncle taught us how to fish from the old wooden dock.",
    prompt =  "On rainy afternoons, my father would pull out his old record player and we'd listen to jazz music in the living room.",

    # Encoding mode
    if args.encode:
        secret_message = args.encode

        print(f"Encoding message: {secret_message}")

        stego_text = encode_steganographic_text(prompt, secret_message)
        write_file(args.output, stego_text)

        print(f"\nEncoded text written to: {args.output}")
        print(f"\nSteganographic text:\n{stego_text}")


    # Decoding mode
    elif args.decode:
        stego_text = read_file(args.decode)

        print(f"Decoding message from: {args.decode}")

        # Calculate start position using the same static prompt
        prompt_ids = tokenizer.encode(prompt, return_tensors="pt")
        start = prompt_ids.size(1)

        decoded = decode_steganographic_text(stego_text, start)

        # Print decoded message to stdout
        print(f"\nDecoded message:\n{decoded}")
