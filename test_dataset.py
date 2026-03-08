from data.dataset_loader import load_dataset

docs = load_dataset()

print("Example document:\n")
print(docs[0][:500])