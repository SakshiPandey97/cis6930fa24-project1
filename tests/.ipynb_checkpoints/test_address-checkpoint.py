import pytest
import subprocess
import os
import tempfile

def test_address_redaction():
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt') as tmp_input:
        tmp_input.write("The event is located at 1600 Pennsylvania Avenue NW, Washington, D.C.")
        tmp_input_name = tmp_input.name

    with tempfile.TemporaryDirectory() as tmp_output_dir:
        subprocess.run(['python', 'redactor.py',
                        '--input', tmp_input_name,
                        '--output', tmp_output_dir,
                        '--address'])

        output_file_name = os.path.basename(tmp_input_name) + '.censored'
        output_file_path = os.path.join(tmp_output_dir, output_file_name)

        with open(output_file_path, 'r') as f:
            output_content = f.read()

        assert "1600 Pennsylvania Avenue NW" not in output_content
        assert "Washington, D.C." not in output_content
        assert "The event is located at" in output_content
        assert "████" in output_content

    os.remove(tmp_input_name)
