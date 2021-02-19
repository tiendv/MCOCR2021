from evaluate import evaluate

cer_full = evaluate("results.txt", "FULL")
cer_address = evaluate("results.txt", "ADDRESS")

print(f"FULL CER: {cer_full}")
print(f"ADDRESS CER: {cer_address}")