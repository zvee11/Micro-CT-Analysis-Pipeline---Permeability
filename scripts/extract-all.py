from pathlib import Path
import shutil

source = Path(r"C:\Users\20200864\Desktop\SHELL\THESIS")
output = Path(r"C:\Users\20200864\Desktop\SHELL\THESIS\pythons")
print("Source:", source.resolve())
print("Output:", output.resolve())

if not source.exists():
    raise FileNotFoundError(f"Source folder does not exist: {source}")

py_files = list(source.rglob("*.py"))

print(f"Found {len(py_files)} Python files.")

if not py_files:
    raise RuntimeError("No .py files were found in the source folder.")

output.mkdir(parents=True, exist_ok=True)

for py_file in py_files:
    # Avoid copying files already inside the output folder
    if output in py_file.parents:
        continue

    relative_path = py_file.relative_to(source)
    destination = output / relative_path

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(py_file, destination)

    print(f"Copied: {py_file} -> {destination}")

print("Finished.")