import pytest
import subprocess
import os
import tempfile

def test_names_redaction():
    test_text = "Hello, my name is Jack O'Lantern. Welcome to the haunted house."
    
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt') as tmp_input:
        tmp_input.write(test_text)
        tmp_input_name = tmp_input.name

    with tempfile.TemporaryDirectory() as tmp_output_dir:
        subprocess.run(['python', 'redactor.py',
                        '--input', tmp_input_name,
                        '--output', tmp_output_dir,
                        '--names'])

        output_file_name = os.path.basename(tmp_input_name) + ".censored"
        output_file_path = os.path.join(tmp_output_dir, output_file_name)

        assert os.path.exists(output_file_path), f"Output file {output_file_path} was not created."

        with open(output_file_path, 'r') as f:
            output_content = f.read()

        assert "Jack O'Lantern" not in output_content
        assert "Hello, my name is ██████████████. Welcome to the haunted house." in output_content

    os.remove(tmp_input_name)
