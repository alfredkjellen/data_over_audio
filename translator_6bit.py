def create_ascii_dict():
    characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz ,.!?"
    ascii_to_bin = {char: format(index, '06b') for index, char in enumerate(characters)}
    bin_to_ascii = {v: k for k, v in ascii_to_bin.items()}
    return ascii_to_bin, bin_to_ascii

def file_to_binary(file_path):
    ascii_to_bin, _ = create_ascii_dict()
    with open(file_path, 'r') as file:
        text = file.read()
    binary_output = []
    for char in text:
        if char in ascii_to_bin:
            binary_output.append(ascii_to_bin[char])
        else:
            raise ValueError(f"Tecken '{char}' finns inte i teckenuppsättningen.")
    return ''.join(binary_output)

def text_to_binary(text):
    ascii_to_bin, _ = create_ascii_dict()
    binary_output = []
    for char in text:
        if char in ascii_to_bin:
            binary_output.append(ascii_to_bin[char])
        else:
            raise ValueError(f"Tecken '{char}' finns inte i teckenuppsättningen.")
    return ''.join(binary_output)

def binary_to_text(binary_data):
    _, bin_to_ascii = create_ascii_dict()
    text_output = []
    for i in range(0, len(binary_data), 6):
        bin_chunk = binary_data[i:i+6]
        if bin_chunk in bin_to_ascii:
            text_output.append(bin_to_ascii[bin_chunk])
        else:
            raise ValueError(f"Binär sekvens '{bin_chunk}' finns inte i teckenuppsättningen.")
    return ''.join(text_output)
