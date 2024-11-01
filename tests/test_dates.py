import pytest
import subprocess
import os
import tempfile

def test_dates_redaction():
    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt') as tmp_input:
        tmp_input.write("Our meeting is scheduled for March 15, 2021.")
        tmp_input_name = tmp_input.name

    with tempfile.TemporaryDirectory() as tmp_output_dir:
        subprocess.run(['python', 'redactor.py',
                        '--input', tmp_input_name,
                        '--output', tmp_output_dir,
                        '--dates'])

        output_file_name = os.path.basename(tmp_input_name) + '.censored'
        output_file_path = os.path.join(tmp_output_dir, output_file_name)

        with open(output_file_path, 'r') as f:
            output_content = f.read()

        assert "March 15, 2021" not in output_content
        assert "Our meeting is scheduled for" in output_content
        assert "████" in output_content

    os.remove(tmp_input_name)
