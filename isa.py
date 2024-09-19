import json
from enum import Enum

"""
Модуль ISA расшифровывается как Instruction Set Architecture (Архитектура набора команд).
"""


class Opcode(str, Enum):
    load = "load"  # Команда загрузки данных
    store = "store"  # Команда сохранения данных
    add = "add"  # Команда сложения
    sub = "sub"  # Команда вычитания
    mod = "mod"  # Команда вычисления остатка от деления
    jmp = "jmp"  # Команда безусловного перехода
    cmp = "cmp"  # Команда сравнения
    jz = "jz"  # Команда перехода, если ноль
    push = "push"  # Команда добавления данных в стек
    pop = "pop"  # Команда удаления данных из стека
    iret = "iret"  # Команда возврата из прерывания
    ei = "ei"  # Команда разрешения прерываний
    di = "di"  # Команда запрещения прерываний
    hlt = "hlt"  # Команда остановки выполнения
    jnz = "jnz"  # Команда перехода, если не ноль
    interrupt = ("interrupt",)  # Команда прерывания
    indirect = ("indirect",)  # Команда косвенного адреса
    jn = ("jn",)  # Команда перехода, если отрицательно
    jnn = "jnn"  # Команда перехода, если не отрицательно


class DataType(Enum):
    num = "num"
    string = ("string",)
    char = "char"


def write_code(code: list, filename: str) -> None:
    with open(filename, "w") as f:
        f.write(json.dumps(code, indent=4))


def load_code_data(inst, data):
    with open(inst, encoding="utf-8") as f:
        instructions = json.loads(f.read())
        for inst in instructions:
            inst["opcode"] = Opcode[inst["opcode"]]
            if inst["arg"] != "None":
                inst["arg"] = int(inst["arg"])
            else:
                inst["arg"] = None
    with open(data, encoding="utf-8") as f:
        data = json.loads(f.read())
        for d in data:
            d["type"] = DataType(d["type"])
    return instructions, data


def encode_data(name: str, val, d_type: DataType) -> dict:
    return {"name": name, "type": d_type, "val": f"{val}"}
