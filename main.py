from isa import Opcode


def main():
    s = "add"
    s2opdcode = Opcode(s)
    print(s2opdcode.value)


if __name__ == "__main__":
    main()
