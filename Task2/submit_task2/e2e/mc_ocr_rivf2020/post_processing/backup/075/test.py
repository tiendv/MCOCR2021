from evaluate import evaluate

cer_full = evaluate("train_df/results.txt", "FULL")
# cer_address = evaluate("train_df/results.txt", "ADDRESS")

print(f"FULL CER: {cer_full}")
# print(f"ADDRESS CER: {cer_address}")