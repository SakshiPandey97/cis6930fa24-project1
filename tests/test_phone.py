import pytest
import subprocess
import os
import tempfile

def test_phones_redaction():
    test_text = "If you see a ghost, call the ghostbuster hotline at 666-123-4567 or (666) 765-4321."
    
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt') as tmp_input:
        tmp_input.write(test_text)
        tmp_input_name = tmp_input.name

    with tempfile.TemporaryDirectory() as tmp_output_dir:
        subprocess.run(['python', 'redactor.py',
                        '--input', tmp_input_name,
                        '--output', tmp_output_dir,
                        '--phones'])

        output_file_name = os.path.basename(tmp_input_name) + '.censored'
        output_file_path = os.path.join(tmp_output_dir, output_file_name)

        with open(output_file_path, 'r') as f:
            output_content = f.read()

        assert "666-123-4567" not in output_content
        assert "(666) 765-4321" not in output_content
        assert "If you see a ghost, call the ghostbuster hotline at ████████████ or ██████████████." in output_content

    os.remove(tmp_input_name)
