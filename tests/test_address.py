import pytest
import subprocess
import os
import tempfile

def test_address_redaction():
    test_text = "The annual vampire society gathering will be held at Mockingbird Lane, Transylvania."
    
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt') as tmp_input:
        tmp_input.write(test_text)
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

        assert "Mockingbird Lane" not in output_content
        assert "Transylvania" not in output_content
        assert "The annual vampire society gathering will be held at ████████████████, ████████████." in output_content

    os.remove(tmp_input_name)
