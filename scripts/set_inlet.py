import re, sys

if len(sys.argv) != 3:
    print("Usage: python3 set_inlet.py path/to/U U_value"); sys.exit(1)
path, U = sys.argv[1], sys.argv[2]
text = open(path).read()
text = re.sub(r"internalField\s+uniform\s*\([^)]*\)",
              f"internalField   uniform ({U} 0 0)", text, count=1)
text = re.sub(r"(inlet\s*{[^}]*value\s+uniform\s*\()[^)]*(\))",
              r"\g<1>" + U + r" 0 0\g<2>", text, count=1, flags=re.S)
open(path, "w").write(text)
print(f"Set inlet velocity to {U} m/s in {path}")
