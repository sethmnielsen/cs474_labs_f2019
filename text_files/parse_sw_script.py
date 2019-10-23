with open('combined_dialogues.txt', 'r') as f:
    with open('starwars_data2.txt', 'w') as new_f:
        for line in f:
            if line.isupper():
                continue
            else:
                s = ''
                for word in line.split():
                    if word.isupper() or word[0] == '#':
                        if word != 'I':
                            continue
                    if word.startswith('Artoo-Detoo'):
                        word = 'R2-D2' + word[11:]
                    elif word.startswith('Artoo'):
                        word = 'R2' + word[5:]
                    elif word.startswith('See-Threepio'):
                        word = 'C-3PO' + word[12:]
                    elif word.startswith('Threepio'):
                        word = '3PO' + word[8:]
                    elif word == '(cont)':
                        continue
                    s += word + ' '
                new_f.write(s)
