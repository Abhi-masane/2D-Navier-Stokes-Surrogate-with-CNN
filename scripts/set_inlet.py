import re
import sys


def update_inlet_file(path, velocity):
    with open(path, "r") as f:
        text = f.read()

    text = re.sub(
        r"internalField\s+uniform\s*\([^)]*\)",
        f"internalField   uniform ({velocity} 0 0)",
        text,
        count=1,
    )

    text = re.sub(
        r"(inlet\s*{[^}]*value\s+uniform\s*\()[^)]*(\))",
        r"\g<1>" + velocity + r" 0 0\g<2>",
        text,
        count=1,
        flags=re.S,
    )

    with open(path, "w") as f:
        f.write(text)


def main():
    if len(sys.argv) != 3:
        print("usage: python3 set_inlet.py path/to/U velocity")
        sys.exit(1)

    path = sys.argv[1]
    velocity = sys.argv[2]
    update_inlet_file(path, velocity)
    print(f"set inlet velocity to {velocity} m/s in {path}")


if __name__ == "__main__":
    main()
