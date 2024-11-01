import pytest
import subprocess
import os
import tempfile

def test_names_redaction():
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt') as tmp_input:
        tmp_input.write("Hello, my name is John Doe. Nice to meet you.")
        tmp_input_name = tmp_input.name


    with tempfile.TemporaryDirectory() as tmp_output_dir:

        result = subprocess.run(['python', 'redactor.py',
                                 '--input', tmp_input_name,
                                 '--output', tmp_output_dir,
                                 '--names'],
                                capture_output=True,
                                text=True)

        output_file_name = os.path.basename(tmp_input_name) + ".censored"
        output_file_path = os.path.join(tmp_output_dir, output_file_name)


        assert os.path.exists(output_file_path), f"Output file {output_file_path} was not created."

        with open(output_file_path, 'r') as f:
            output_content = f.read()


        assert "John Doe" not in output_content, "Name 'John Doe' was not properly redacted."
        assert "Hello, my name is ████████." in output_content, "Redacted output did not match expected content."

    os.remove(tmp_input_name)

