import argparse
from typing import Optional

from isa import DataType, Opcode, write_code


def symbol2opcode(symbol):
    return {
        "load": Opcode.load,
        "store": Opcode.store,
        "add": Opcode.add,
        "sub": Opcode.sub,
        "mod": Opcode.mod,
        "jmp": Opcode.jmp,
        "cmp": Opcode.cmp,
        "jz": Opcode.jz,
        "push": Opcode.push,
        "pop": Opcode.pop,
        "iret": Opcode.iret,
        "ei": Opcode.ei,
        "di": Opcode.di,
        "hlt": Opcode.hlt,
        "jnz": Opcode.jnz,
        "jn": Opcode.jn,
        "jnn": Opcode.jnn,
    }.get(symbol)


def symbol2datatype(symbol):
    return {"num": DataType.num, "string": DataType.string, "char": DataType.char}.get(symbol)


class Translator:
    out_instructions_file_name: str
    out_data_file_name: str

    file_data: str
    labels: list[dict]
    instructions: list[dict]
    label_pos: dict

    def __init__(
        self, path_to_program: str, out_instructions_file_name: str, out_data_file_name: Optional[str] = None
    ) -> None:
        self.labels = []
        self.instructions = []
        self.label_pos = dict()
        self.out_instructions_file_name = out_instructions_file_name
        self.out_data_file_name = out_data_file_name
        self.file_data = self.read_data_from_file(path_to_program)

    def read_data_from_file(self, path_to_file: str) -> str:
        with open(path_to_file, encoding="utf-8") as f:
            return f.read()

    def translate(self):
        self.parse()
        write_code(self.labels, self.out_data_file_name)
        write_code(self.instructions, self.out_instructions_file_name)

    def parse(self):
        self.skip()
        programm_lines = self.delete_spaces_comments()
        self.parse_labels(programm_lines)
        self.parse_instructions(programm_lines)

    def skip(self):
        d = self.file_data.find(".data")
        t = self.file_data.find(".text")
        if t == -1:
            raise ValueError
        if d == -1:
            self.file_data = self.file_data[t:]
        else:
            self.file_data = self.file_data[d:]

    def delete_spaces_comments(self) -> list:
        lines = self.file_data.split("\n")
        updated_lines = []
        for i in range(len(lines)):
            if lines[i] == "" or lines[i] == ".text:" or lines[i] == ".data:":
                continue
            cur_line = lines[i].strip()
            pos = cur_line.find(";")
            if pos != -1:
                cur_line = cur_line[:pos]
            updated_lines.append(cur_line)
        return updated_lines

    def parse_labels(self, lines: list) -> None:
        # Count position of instruction to which label points
        data_mem_pointer = 0
        instr_mem_pointer = 0
        for i in range(len(lines)):
            pos = lines[i].find(":")
            string = None
            if pos == -1:
                instr_mem_pointer += 1
                continue
            quotes = lines[i].find("'")
            if quotes != -1:
                string = lines[i][quotes:]
                lines[i] = lines[i][:quotes]
            line = lines[i].split(" ")
            name = line[0][:pos]
            # Points to the memory cell
            if len(line) == 1: # значит метка и стоит сохранить ее позицию
                self.label_pos[name] = instr_mem_pointer
            if (len(line) == 3) or string:
                l2l = False
                datatype = symbol2datatype(line[1]).name
                if datatype is None:
                    raise ValueError
                var_type = datatype
                val = line[2]
                # Если нужно получить в переменную адрес другой переменной
                # word: string 'hello'
                # pointer: num word
                if val in self.label_pos:
                    val = str(self.label_pos[val])
                    l2l = True
                if var_type == DataType.char.name:
                    val = self.save_char_in_mem(string)
                # Строка - набор char, каждый char храниться в отдельной ячейке
                if var_type is DataType.string.name:
                    self.label_pos[name] = data_mem_pointer
                    offset = self.save_string_in_mem(string)
                    data_mem_pointer += offset # смещаем память на длину строки
                    continue
                self.labels.append({"name": f"{name}", "type": f"{var_type}", "val": val, "l2l": l2l})
                self.label_pos[name] = data_mem_pointer
                data_mem_pointer += 1

    def save_char_in_mem(self, val: str) -> int:
        quotes = val.find("'")
        if quotes != -1:
            val = val.replace("'", "")
        if val == "\\n":
            val = "\n"
        if val == "\\0":
            val = "\0"
        return ord(val)

    def save_string_in_mem(self, string: str) -> int:
        quotes = string.find("'")
        if quotes != -1:
            string = string.replace("'", "")
        nul_char = string.find("\\0")
        if nul_char == -1:
            raise SyntaxError
        string = string[:nul_char]
        string += "\0"
        for char in string:
            self.labels.append({"name": f"{char}", "type": "char", "val": ord(char), "l2l": False})
        return len(string)

    def parse_instructions(self, lines: list):
        counter = 0
        for i in range(len(lines)):
            line = lines[i].split(" ")
            pos = lines[i].find(":")
            if pos != -1:
                continue
            opcode = symbol2opcode(line[0]).name
            # If address_type == False => direct addressing
            address_type = False
            if opcode is None:
                raise ValueError
            # instructions with no operand
            if len(line) == 1:
                arg = None
            else:
                if line[1].isnumeric():
                    arg = int(line[1])
                else:
                    # Replace label with position in data memory or instruction memory
                    sq_brackets = line[1].find("[")
                    if sq_brackets != -1:
                        line[1] = line[1].replace("[", "")
                        line[1] = line[1].replace("]", "")
                        # Indirect addressing
                        address_type = True
                    arg = self.label_pos[line[1]]
            self.instructions.append({"opcode": f"{opcode}", "arg": f"{arg}", "address_type": address_type})
            counter += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "inputs",
        metavar="INPUT",
        nargs="*",
        help="<file_with_programm> <file for encoded instructions> <file for encoded data>",
    )
    args = parser.parse_args().inputs
    assert len(args) > 1, "The number of arguments have to be at least 2"

    t = Translator(*args)
    t.translate()


if __name__ == "__main__":
    main()
