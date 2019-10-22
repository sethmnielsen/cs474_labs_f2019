with open('combined_dialogues.txt', 'r') as f:
    with open('starwars_data.txt', 'w') as new_f:
        for line in f:
            if line.isupper():
                continue
            else:
                if line[0] == '(' and line[-2] == ')':
                    continue
                else:
                    s = ''
                    for word in line.split():
                        if word.isupper() or word[0] == '(' or word[-1] == ')':
                            continue
                        if word.startswith('Artoo-Detoo'):
                            word = 'R2-D2'
                        elif word.startswith('Artoo'):
                            word = 'R2'
                        elif word.startswith('See-Threepio'):
                            word = 'C-3PO'
                        elif word[4:12] == 'Threepio':
                            word = '3PO'
                        s += word + ' '
                    new_f.write(s)
