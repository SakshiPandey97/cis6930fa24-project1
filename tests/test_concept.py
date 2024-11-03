import pytest
import subprocess
import os
import tempfile

def test_concept_redaction():
    test_text = ("On Halloween night, the ghost made a deposit at the haunted bank. "
                 "Everyone feared their trust funds in the phantom bank might vanish. "
                 "The vampire manager assured everyone that this would not happen.")

    with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt') as tmp_input:
        tmp_input.write(test_text)
        tmp_input_name = tmp_input.name

    with tempfile.TemporaryDirectory() as tmp_output_dir:
        subprocess.run(['python', 'redactor.py',
                        '--input', tmp_input_name,
                        '--output', tmp_output_dir,
                        '--concept', 'banking'])

        output_file_name = os.path.basename(tmp_input_name) + '.censored'
        output_file_path = os.path.join(tmp_output_dir, output_file_name)

        with open(output_file_path, 'r') as f:
            output_content = f.read()

        #Concepts related to banking that should be redacted
        assert "deposit" not in output_content
        assert "trust" not in output_content
        assert "savings" not in output_content

        #This doesn't have banking mentioned and isn't relevant to the concept.
        assert "The vampire manager assured everyone that this would not happen." in output_content

        assert "████" in output_content

    os.remove(tmp_input_name)
