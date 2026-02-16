with open('test_automl_tab_full_output.log', 'r', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()
    for line in lines[-10:]:
        print(line.strip())
