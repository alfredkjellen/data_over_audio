def create_ascii_dict():
    characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz ,.!?"
    ascii_to_bin = {char: format(index, '06b') for index, char in enumerate(characters)}
    bin_to_ascii = {v: k for k, v in ascii_to_bin.items()}
    return ascii_to_bin, bin_to_ascii

def text_to_binary(text):
    ascii_to_bin, _ = create_ascii_dict()
    return ''.join(ascii_to_bin.get(char, '######') for char in text)

def binary_to_text(binary_list):
    _, bin_to_ascii = create_ascii_dict()
    binary_string = ''.join(map(str, binary_list))
    return ''.join(bin_to_ascii.get(''.join(map(str, binary_string[i:i+6])), '#') 
                   for i in range(0, len(binary_string) - (len(binary_string) % 6), 6))

def file_to_binary(file_path):
    with open(file_path, 'r') as file:
        return text_to_binary(file.read())