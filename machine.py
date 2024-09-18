import argparse  # библиотека для парсинга аргументов командной строки
import logging
from collections import deque  # очередь для работы с данными внешних устройств
from enum import Enum
from typing import Optional  # для создания опциональных типов(None или значение)

import yaml

from isa import DataType, Opcode, encode_data, load_code_data  # импорт инструкций и данных

# Множества для классификации инструкций процессора
instructions = {Opcode.add, Opcode.cmp, Opcode.load, Opcode.mod, Opcode.di, Opcode.ei, Opcode.hlt, Opcode.store}
control_flow = {Opcode.jmp, Opcode.jnz, Opcode.jz, Opcode.jn, Opcode.jnn}
arithmetic_ops = {Opcode.add, Opcode.sub, Opcode.mod}
address_instructions = {Opcode.add, Opcode.cmp, Opcode.load, Opcode.mod, Opcode.store}
stack_instructions = {Opcode.push, Opcode.pop, Opcode.iret}

"""
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler(f"{__name__}.log", mode="w")
formatter = logging.Formatter("%(name)s %(levelname)s %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
"""


# Перечисление типов памяти
class MemType(Enum):
    instruction_mem = 0
    data_mem = 1


# Класс для работы со внешними устройствами
class ExternalDevice:
    in_data: Optional[deque]  # Очередь с входными данными
    output_data: list  # Выходные данные
    interrupt_vector_address: int  # Адрес вектора прерываний

    def __init__(self, input_data: Optional[deque] = None) -> None:
        self.in_data = input_data  # Входные данные , если они есть
        self.output_data = []  # Инициализируем список для выходных данных
        self.interrupt_vector_address = 2  # Адрес расположения вектора прерываний

    # Возвращение текущего символа из входных данных
    def get_cur_char(self) -> tuple[int, str]:
        if len(self.in_data) == 0:
            return (-1, "")  # если данных нет, вернет вот это
        return self.in_data[0]

    # Принять символа с последующим удалением его из очереди
    def send_char(self) -> dict:
        if len(self.in_data) == 0:
            raise BufferError  # Если данных нет, то кидаем ошибку
        char_s = self.in_data[0][1]
        if char_s == "\x00":
            char_s = "\0"
        char = ord(self.in_data[0][1])  # Вернем числовое значение нашего символа
        self.in_data.popleft()  # Удаление символа из очереди
        ch_for_log = chr(char)  # Получаем удаленный символ для последующего логирования
        if ch_for_log == "\0":
            ch_for_log = "null"
        logging.debug(f"CHAR_IN: {ch_for_log}")
        return {"name": "char_from_input_device", "type": DataType.char, "val": char, "pos_in_data_mem": "0"}

    # Чтение символа и добавление его в выходные данные
    def read_char(self, char):
        self.output_data.append(chr(int(char)))  # добавляем символ в выходные данные
        logging.debug(f"CHAR_OUT: {chr(int(char))}")
        if chr(int(char)) == "\0":  # если был символ конца строки, то выводим всю строку
            word = "".join(self.output_data)
            logging.debug(f"THE WHOLE WORD: {word}")

    # Чтение числа и добавление его в выходные данные
    def read_int(self, num):
        self.output_data.append(num)
        logging.debug(f"INT_OUT: {num}")


# Класс арифметико-логического устройства
class ALU:
    operation_res: Optional[int]  # Результат выполнения операции
    left_in: Optional[int]  # Левый операнд
    right_in: Optional[int]  # Правый операнд
    zero_flag: bool  # Флаг нуля
    negative_flag: bool  # Флаг отрицательного значения

    def __init__(self) -> None:
        self.operation_res = 0
        self.left_in = 0
        self.right_in = 0
        self.zero_flag = False
        self.negative_flag = False

    # Выполнение арифметических операций
    def do_arithmetic(self, opcode: Opcode):
        if opcode is Opcode.add:
            self.add_op()
        elif opcode is Opcode.sub:
            self.sub_op()
        elif opcode is Opcode.mod:
            self.mod_op()
        elif opcode is Opcode.cmp:
            self.sub_op()  # Операция сравнения сводится к вычитанию
        self.check_if_zero()  # Проверка на ноль
        self.check_if_negative()  # Проверка на отрицательное значение

    def check_if_zero(self):
        if self.operation_res == 0:
            self.zero_flag = True
        else:
            self.zero_flag = False

    def check_if_negative(self):
        if self.operation_res < 0:
            self.negative_flag = True
        else:
            self.negative_flag = False

    def add_op(self):
        self.operation_res = self.left_in + self.right_in

    def sub_op(self):
        self.operation_res = self.left_in - self.right_in

    def mod_op(self):
        self.operation_res = self.left_in % self.right_in


# Класс для управления потоками данных
class DataPath:
    pc: int  # Регистр счетчика команд
    ar: int  # Регистр адреса
    acc: dict  # Регистр аккумулятора
    sp: int  # Регистр указателя стека
    ir: dict  # Регистр команд
    dr: dict  # Регистр данных

    inst_mem: list  # Память инструкций
    data_mem: list  # Память ALU

    in_dev: ExternalDevice  # Входное устройство
    out_dev: ExternalDevice  # Выходное устройство

    def __init__(
            self,
            start_cell_isr: int,
            isr_prog: list,
            isr_data: list,
            input_device: ExternalDevice,
            output_device: ExternalDevice,
    ) -> None:
        # Sp указывает на 2048, поскольку изначально в стеке нет данных
        self.sp = 2048
        self.acc = encode_data("empty acc", 0, DataType.num)
        self.ar = 0
        self.ir = 0
        self.dr = 0

        # Инициализируем память инструкций и данных размером 2048 ячеек (2**11)
        self.inst_mem = [0] * (2 ** 11)
        self.data_mem = [0] * (2 ** 11)

        # Устанавливаем начальную ячейку для обработки прерываний
        # start_cell_isr - начальная ячейка для обработки прерываний
        self.inst_mem[2] = start_cell_isr
        self.data_mem[2] = encode_data("interrupt vector", start_cell_isr, DataType.num)

        # Указывают на первую свободную ячейку памяти, нужны для загрузки нескольких программ в память
        self.instr_empty_cell = 0

        # Указывает на первую свободную ячейку для данных
        self.data_empty_cell = 3

        # Загрузка программы и данных в память
        self.load_program_in_mem(isr_prog, isr_data)

        # В instr_empty_cell запишется первая инструкция основной программы
        # Установим в счетчик команд указатель на первую инструкцию
        self.pc = self.instr_empty_cell

        # Инициализация арифметико-логического устройства
        self.alu = ALU()

        # Инициализация устройств ввода-вывода
        self.in_dev = input_device
        self.out_dev = output_device

    # Загрузка данных в память
    def load_program_in_mem(self, instr: list, data: list):
        counter = 0
        d_offset = self.data_empty_cell  # начальная ячейка для данных
        instr_offset = self.instr_empty_cell  # начальная ячейка для инструкций

        # Загрузка данных в память данных
        for i in range(d_offset, d_offset + len(data)):
            if data[counter]["l2l"] is True:
                # Если выяснилось, что косвенная адресация, то смещаемся
                val = int(data[counter]["val"]) + d_offset
                data[counter]["val"] = val
            self.data_mem[i] = data[counter]  # записываем данные в память данных
            counter += 1

        # Обновляем смещение для следующего блока данных
        self.data_empty_cell = self.data_empty_cell + counter
        counter = 0
        for i in range(instr_offset, instr_offset + len(instr)):
            if instr[counter]["opcode"] in {Opcode.add, Opcode.sub, Opcode.cmp, Opcode.load, Opcode.store, Opcode.mod}:
                # При загрузке данных в память мы смещаем их адреса, чтобы они не конфликтовали
                # с уже существующими данными.
                instr[counter]["arg"] += d_offset
            elif instr[counter]["opcode"] in {Opcode.jmp, Opcode.jnz, Opcode.jz, Opcode.jn, Opcode.jnn}:
                # Инструкции тоже смещены, если загружаемая программа загружается не первой.
                instr[counter]["arg"] += instr_offset
            self.inst_mem[i] = instr[counter]  # записать в память
            counter += 1
        # Обновляем смещение для следующего блока команд
        self.instr_empty_cell = self.instr_empty_cell + counter

    # Установка значения регистра счетчика команд PC в зависимости от выбранной операции
    def latch_pc(self, sel: Opcode, arg: Optional[int] = None):
        # Если операция не управляющая и не прерывание
        if (sel not in control_flow) and (sel is not Opcode.iret):
            self.pc += 1
        elif sel is Opcode.iret:
            # Для операции iret устанавливаем PC на значение, содержащееся в DR(должно быть сохранено при прерываниях)
            self.pc = int(self.dr["val"])
        else:
            # Для всех остальных случаев устанавливаем PC в значение аргумента
            self.pc = arg

    # Установка значения регистра адреса AR в зависимости от выбранной операции
    def latch_ar(self, sel):
        if sel is Opcode.interrupt:
            # Для прерываний AR устанавливается на значение вектора прерываний
            self.ar = self.in_dev.interrupt_vector_address
        elif sel in address_instructions:
            # Для операций, работающих с адресами, AR устанавливается на значение аргумента в IR
            self.ar = int(self.ir["arg"])
        elif sel is Opcode.indirect:
            # Для косвенной адресации AR устанавливается на значение из DR
            self.ar = int(self.dr["val"])
        elif sel in stack_instructions:
            # Для стековых инструкций AR устанавливается из SP(указатель стека)
            self.ar = self.sp

    # Установка значения регистра аккумулятора ACC в зависимости от выбранной операции
    def latch_acc(self, sel):
        if sel is Opcode.interrupt:
            # При прерывании в ACC устанавливается значение PC
            self.acc = encode_data("saved_pc", self.pc, DataType.num)
        elif sel in {Opcode.load, Opcode.pop}:
            # При операции загрузки или извлечения из стека, ACC устанавливается на значение DR
            self.acc = self.dr
        elif sel in arithmetic_ops:
            # При арифметических операциях в AC устанавливается результат операции
            self.acc = encode_data(f"{sel} operation res", self.alu.operation_res, DataType.num)

    # Установка значения
    def latch_sp(self, sel):
        if sel in {Opcode.push, Opcode.interrupt}:
            # Для операций push или прерываний, стек растет вверх, поэтому уменьшаем SP
            # 2045: новейшее значение  ^
            # 2046:                 |
            # 2047: старейшее значение |
            self.sp -= 1
        elif sel in {Opcode.pop, Opcode.iret}:
            # Для операции pop и iret стек наоборот идет вниз, поэтому прибавляем
            self.sp += 1

    # Передача значения из ACC и DR в ALU
    def latch_alu(self):
        self.alu.left_in = int(self.acc["val"])
        self.alu.right_in = int(self.dr["val"])

    # Чтение данных из памяти в зависимости от типа памяти
    def read_from_mem(self, mem_type):
        # Если идет чтение инструкций, то записываем в регистр IR из PC
        if mem_type is MemType.instruction_mem:
            self.ir = self.inst_mem[self.pc]
        elif mem_type is MemType.data_mem:
            # Если читаем из памяти данных
            if self.ar == 0:
                # Если адрес равен 0, получаем символ с входного устройства
                self.dr = self.in_dev.send_char()
            else:
                # Если нет, то читаем с указанного адреса
                self.dr = self.data_mem[self.ar]

    # Запись данных в память из ACC
    def write_to_data_mem(self):
        if self.ar == 1:
            # Если адрес равен 1, это может означать что нужно вывести данные на выходное устройство
            if self.acc["type"] == DataType.char:
                self.out_dev.read_char(self.acc["val"])
            elif self.acc["type"] == DataType.num:
                self.out_dev.read_int(self.acc["val"])
        else:
            # Если нет, то запишем данные из acc в память
            self.data_mem[self.ar] = self.acc


class ControUnit:
    input_device: ExternalDevice
    datapath: DataPath

    ei: bool  # флаг разрешения прерываний
    interrupt: bool  # флаг запроса прерываний
    _tick: int  # счетчик тиков или циклов выполнения

    def __init__(self, input_device, datapath) -> None:
        self.input_device = input_device
        self.datapath = datapath
        self._tick = 0
        self.ei = True
        self.interrupt = False

    """
    Основная функция выполнения инструкций. Проводит этапы выборки, декодирования и выполнения инструкций.
    """

    def execute(self):
        # Первый этап , выборка инструкции: instruction fetch(inst_mem[pc] -> ir)
        self.datapath.read_from_mem(MemType.instruction_mem)  # Инициируем чтение команды
        cur_inst = self.datapath.ir["opcode"]  # Текущая операция
        arg = self.datapath.ir["arg"]  # Аргумент операции
        ad_type = self.datapath.ir["address_type"]  # Тип адресации
        self.tick()  # Производим такт
        # Этап декодирования и выполнения
        if cur_inst in instructions:
            # если текущая инструкция является базовой, то выполняем ее
            self.execute_basic_instructions(cur_inst, ad_type)
            self.datapath.latch_pc(cur_inst)  # обновляем регистр PC
        elif cur_inst in control_flow:
            # если текущая инструкция связана с управлением
            self.execute_control_flow_instruction(cur_inst, arg)
        elif cur_inst in stack_instructions:
            # если текущая инструкция связана со стеком
            self.excute_stack_instructions(cur_inst)
            if cur_inst is not Opcode.iret:
                # Обновляем PC для всех операций, кроме iret
                self.datapath.latch_pc(cur_inst)
        # Проверка на наличие запроса прерывания
        self.check_for_interrupt()

    """
    Выполняет инструкции(прыжки, условные переходы)
    """

    def execute_control_flow_instruction(self, instr, arg):
        if instr is Opcode.jmp:
            # безусловный прыжок, устанавливает PC в значение аргумента
            self.datapath.latch_pc(instr, arg)
            self.tick()
        elif instr is Opcode.jz:
            # Тик для просмотра флага
            self.tick()
            if self.datapath.alu.zero_flag:
                # Если флаг нуля установлен, то изменяем значение PC на значение аргумента
                self.datapath.latch_pc(instr, arg)
            else:
                # костыль, чтобы произошёл простой инкремент PC, который происходит после выполнения базовой операции
                self.datapath.latch_pc(Opcode.add)
            self.tick()
        elif instr is Opcode.jnz:
            # аналогично, но переход если флаг не установлен
            self.tick()
            if self.datapath.alu.zero_flag is False:
                self.datapath.latch_pc(instr, arg)
            else:
                self.datapath.latch_pc(Opcode.add)
            self.tick()
        elif instr is Opcode.jn:
            # переход если установлен негатив флаг
            self.tick()
            if self.datapath.alu.negative_flag:
                self.datapath.latch_pc(instr, arg)
            else:
                self.datapath.latch_pc(Opcode.add)
            self.tick()
        elif instr is Opcode.jnn:
            # переход если не установлен негатив флаг
            self.tick()
            if self.datapath.alu.negative_flag is False:
                self.datapath.latch_pc(instr, arg)
            else:
                self.datapath.latch_pc(Opcode.add)
            self.tick()

    """
    Выполнение инструкций связанных стеком
    """

    def excute_stack_instructions(self, instr):
        if instr is Opcode.pop:
            # извлечение данных из стека
            # берем и SP -> AR
            self.datapath.latch_ar(instr)
            self.tick()
            # далее читаем значение
            self.datapath.read_from_mem(MemType.data_mem)
            # сохраняем его в аккумулятор
            self.datapath.latch_acc(instr)
            # делаем инкремент стека, ибо один элемент ушел
            self.datapath.latch_sp(instr)
            self.tick()
        elif instr is Opcode.push:
            # кладем на стек
            # делаем инкремент стека
            self.datapath.latch_sp(instr)
            # SP -> AR
            self.datapath.latch_ar(instr)
            # записываем новые данные в стек
            self.datapath.write_to_data_mem()
            self.tick()
        elif instr is Opcode.iret:
            # операция возврата из прерывания
            # читаем PC из стека
            self.datapath.latch_ar(instr)
            self.tick()
            self.datapath.read_from_mem(MemType.data_mem)
            # увеличиваем указатель стека, так как одно из значений ушло
            self.datapath.latch_sp(instr)
            self.tick()
            # обновляем PC, чтобы вернуть к выполнению нашей проги после прерывания
            self.datapath.latch_pc(instr)
            self.ei = True
            self.interrupt = False
            self.tick()
            logging.debug("-----------Interrupt-Ended-----------")

    """
    Выполнение базовой инструкции: базовые инструкции, арифметические, загрузка/сохранение данных
    """

    def execute_basic_instructions(self, instr, ad_type):
        if ad_type is True:
            # если используется косвенная адресация, загружаем адрес
            self.load_indirect_address(instr)
        else:
            # если косвенная не используется, то просто латчим ar
            self.datapath.latch_ar(instr)
            self.tick()

        if instr is Opcode.load:
            # операция загрузки считывает данные и обновляет аккумулятор
            self.datapath.read_from_mem(MemType.data_mem)
            self.datapath.latch_acc(instr)
            self.tick()

        elif instr is Opcode.store:
            # операция записи, сохраняет данные из аккумулятора в память
            self.datapath.write_to_data_mem()
            self.tick()
        elif instr in arithmetic_ops:
            # арифметические операции, загружаем операнды, выполняем вычисления
            self.datapath.read_from_mem(MemType.data_mem)
            # латчим правый и левых входы
            self.datapath.latch_alu()
            self.tick()
            # выполняем вычисления
            self.datapath.alu.do_arithmetic(instr)
            self.datapath.latch_acc(instr)
            self.tick()
        elif instr is Opcode.cmp:
            # операция сравнения
            self.datapath.read_from_mem(MemType.data_mem)
            self.datapath.latch_alu()
            self.tick()
            self.datapath.alu.do_arithmetic(instr)
            self.tick()
        elif instr in {Opcode.ei, Opcode.di}:
            # управление прерываниями
            if instr is Opcode.ei:
                self.ei = True
            elif instr is Opcode.di:
                self.ei = False
            self.tick()
        elif instr is Opcode.hlt:
            raise SystemExit

    def load_indirect_address(self, instr):
        """Загрузить в AR косвенный адрес, после завершиния функции в AR лежит нужный адрес"""
        self.datapath.latch_ar(instr)
        self.tick()
        self.datapath.read_from_mem(MemType.data_mem)
        self.datapath.latch_ar(Opcode.indirect)
        self.tick()

    def check_for_interrupt(self):
        # One tick to check for interrupt request
        self.tick()
        if (self.ei is True) and (self.interrupt is True):
            logging.debug("-----------Interrupt-Started-----------")
            self.do_interrupt()

    def do_interrupt(self):
        self.ei = False
        logging.debug("EI switched to False")
        self.save_context()
        # Находим и загружаем ISR (Interrupt Service Routine) для обработки прерывания
        self.find_isr()

    def save_context(self):
        # Сохраняем текущее значение счетчика команд (PC) в стек
        self.save_pc()

    def save_acc(self):
        # Для push необходимо сначала инкрементировать SP, поэтому сначала вызываем latch_sp
        self.datapath.latch_sp(Opcode.push)
        self.tick()
        self.datapath.latch_ar(Opcode.push)
        self.datapath.write_to_data_mem()
        self.tick()

    def save_pc(self):
        # Сохраняем текущее значение PC (счетчика команд) в аккумулятор (ACC) для дальнейшей записи в стек
        self.datapath.latch_acc(Opcode.interrupt)
        # Инкрементируем указатель стека (SP) перед записью PC в стек
        self.datapath.latch_sp(Opcode.push)
        self.tick()
        # Копируем значение SP в адресный регистр (AR)
        self.datapath.latch_ar(Opcode.push)
        # Записываем значение PC (из аккумулятора) в стек по адресу, указанному в AR
        self.datapath.write_to_data_mem()
        self.tick()
        logging.debug(f"save_pc: ar:{self.datapath.ar} mem[ar]:{self.datapath.data_mem[self.datapath.ar]}")

    def find_isr(self):
        # Загружаем в AR адрес, по которому хранится вектор прерываний
        self.datapath.latch_ar(Opcode.interrupt)
        self.tick()
        # Считываем адрес ISR из памяти по текущему значению AR
        self.datapath.read_from_mem(MemType.data_mem)
        self.tick()
        # Загружаем считанный адрес ISR в аккумулятор (ACC)
        self.datapath.latch_acc(Opcode.load)
        self.tick()
        # Загружаем адрес ISR из ACC в PC, чтобы начать выполнение обработчика прерывания
        self.datapath.latch_pc(Opcode.iret)
        self.tick()
        logging.debug(f"find_isr: ar:{self.datapath.ar} mem[ar]:{self.datapath.data_mem[self.datapath.ar]}")
        logging.debug("-----------Execute-ISR-----------")

    def tick(self):
        self._tick += 1

    def __repr__(self) -> str:
        state_repr = "tick:{} pc:{} ar:{} acc:{} ei:{} interrupt:{}".format(
            self._tick, self.datapath.pc, self.datapath.ar, self.datapath.acc["val"], self.ei, self.interrupt
        )
        cur_instr = self.datapath.inst_mem[self.datapath.pc]
        opcode = cur_instr["opcode"].name
        arg = cur_instr["arg"]
        if (arg is not None) and (opcode not in {"jmp", "jz", "jnz", "jn", "jnn"}):
            arg_in_data_mem = self.datapath.data_mem[arg]
        else:
            arg_in_data_mem = "null"
        if arg is None:
            arg = "null"
        instr_repr = "Opcode:{} Arg:{} Mem[arg]:{}".format(opcode, arg, arg_in_data_mem)
        return "{} \t{}".format(state_repr, instr_repr)


def simulation(limit: int, inst_mem: list, data_mem: list, inst_isr, data_isr, input_data: list):
    in_d: deque = deque(input_data)
    in_dev = ExternalDevice(input_data=in_d)
    out_dev = ExternalDevice()
    datapath = DataPath(
        start_cell_isr=0, isr_prog=inst_isr, isr_data=data_isr, input_device=in_dev, output_device=out_dev
    )
    datapath.load_program_in_mem(inst_mem, data_mem)
    controlunit = ControUnit(input_device=in_dev, datapath=datapath)
    try:
        c = 0
        while c < limit:
            logging.debug("%s", controlunit)
            if len(in_dev.in_data) != 0:
                if in_dev.get_cur_char()[0] <= c:
                    controlunit.interrupt = True
            controlunit.execute()
            c += 1
    except SystemExit:
        logging.debug(f"Simulation stopted by HLT command Total ticks: {controlunit._tick}")
        if len(out_dev.output_data) != 0:
            print(*out_dev.output_data)
    except BufferError:
        logging.debug("Input buffer is empty")
        return


def main(instr_f: str, data_f: str, input_f: str):
    inst_p, data_p = load_code_data(instr_f, data_f)

    inst_isr, data_isr = load_code_data("static/isr/instr.json", "static/isr/data.json")

    with open(input_f, encoding="utf-8") as f:
        ym = yaml.safe_load(f.read())
    data = ym
    simulation(limit=100000, inst_mem=inst_p, data_mem=data_p, inst_isr=inst_isr, data_isr=data_isr, input_data=data)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "inputs",
        metavar="INPUT",
        nargs="*",
        help="<path to file with instruction memory> <path to file with data memory> <path to input file>",
    )
    args: list[str] = parser.parse_args().inputs
    assert len(args) == 3, "There should be two arguments"
    main(args[0], args[1], args[2])
