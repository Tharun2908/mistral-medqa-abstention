from scripts.common.medqa_data import load_medqa

train = load_medqa("train")
dev = load_medqa("dev")

print("\n=== SPLIT CHECK ===")
print("train:", len(train))
print("dev:  ", len(dev))

print("\n=== FIRST DEV EXAMPLE ===")
print("question:", dev[0]["question"][:150])
print("options:", dev[0]["options"])
print("answer:", dev[0]["answer_idx"])

print("\n=== TEST LOCK CHECK ===")
try:
    load_medqa("test")
    print("ERROR: test was not locked!")
except RuntimeError as e:
    print("PASS: test correctly blocked")
    print(e)

print("\n=== EXPLICIT FINAL-TEST ACCESS CHECK ===")
test = load_medqa("test", allow_test=True)
print("test:", len(test))