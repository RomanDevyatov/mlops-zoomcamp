import os

path = "./artifacts/1/0b8ab2bed886413dbabda1ce6009d5ee/artifacts/preprocessor/preprocessor.b"
size_bytes = os.path.getsize(path)
print(f"Model size: {size_bytes} bytes ({size_bytes/1024:.2f} KB)")
