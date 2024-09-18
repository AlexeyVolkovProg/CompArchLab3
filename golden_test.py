import contextlib
import io
import logging
import os
import tempfile

import pytest
import yaml

import machine
import translator


@pytest.mark.golden_test("golden/*.yml")
def test_translator_and_machine(golden, caplog):
    caplog.set_level(logging.DEBUG)

    with tempfile.TemporaryDirectory() as tmpdirname:
        prog = os.path.join(tmpdirname, "prog.txt")
        input_stream = os.path.join(tmpdirname, "input.yml")
        data = os.path.join(tmpdirname, "data.json")
        instr = os.path.join(tmpdirname, "instr.json")

        with open(prog, "w", encoding="utf-8") as file:
            file.write(golden["in_source"])
        with open(input_stream, "w", encoding="utf-8") as file:
            yaml.dump(golden["in_stdin"], file)

        t = translator.Translator(prog, instr, data)

        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            t.translate()
            print("============================================================")
            machine.main(instr_f=instr, data_f=data, input_f=input_stream)

        with open(instr, encoding="utf-8") as file:
            code = file.read()

        with open(data, encoding="utf-8") as file:
            data = file.read()

        assert code == golden.out["out_code"]
        assert data == golden.out["out_data"]
        assert stdout.getvalue() == golden.out["out_stdout"]
        assert caplog.text == golden.out["out_log"]
